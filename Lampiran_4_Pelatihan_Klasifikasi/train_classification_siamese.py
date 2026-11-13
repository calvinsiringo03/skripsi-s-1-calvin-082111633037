# -*- coding: utf-8 -*-
"""
Pelatihan Model Klasifikasi Dua Tahap (Stage 1 Linear Probing & Stage 2 Full Fine-Tuning)
Arsitektur: ConvNeXt V2-Tiny (timm) dengan Shared-Weights Siamese Network,
Augmentasi Warna RandStainNA Sinkron, dan Normalized Feature Consistency Loss.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from timm.optim import create_optimizer_v2
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm


@dataclass
class Cfg:
    TRAIN_DATA_ROOT: str = r"C:\Calvin_082111633037\training\NCT-CRC\train"
    VAL_DATA_ROOT: str = r"C:\Calvin_082111633037\training\NCT-CRC\val"
    RANDSTAINNA_OFFLINE_ROOT: str = r"C:\Calvin_082111633037\training\NCT_RandStainNA"
    VAL_RANDSTAINNA_ROOT: str | None = None
    PRETRAIN_CHECKPOINT_PATH: str = r"G:\My Drive\SIRL3D_Workspace\checkpoints\phase1_pretraining\convnext_v2_tiny_ssl.pth"
    OUTPUT_DIR: str = r"G:\My Drive\SIRL3D_Workspace\checkpoints\phase2_training\lambda07"

    CLASS_NAMES: tuple = ("TUM", "STR", "LYM", "DEB", "ADI", "BACK", "NORM")
    INPUT_SIZE: int = 224

    # Konfigurasi Arsitektur & Regularisasi Konsistensi Fitur (Tabel 4.12 & 4.13)
    DROP_PATH_RATE: float = 0.0
    MIN_CHECKPOINT_COVERAGE: float = 0.80
    LAMBDA_CONSISTENCY: float = 0.7

    # Stage 1: Linear Probing (Frozen Backbone)
    LP_LEARNING_RATE: float = 5e-4
    LP_MIN_LR: float = 1e-5
    LP_WARMUP_EPOCHS: int = 1
    LP_EPOCHS: int = 10

    # Stage 2: Full Fine-Tuning (Unfrozen Backbone + Layer-wise LR Decay)
    FT_LEARNING_RATE: float = 1e-6
    FT_MIN_LR: float = 1e-7
    FT_WARMUP_EPOCHS: int = 1
    FT_EPOCHS: int = 10
    LAYER_DECAY: float = 0.9

    # Parameter Optimisasi Global
    WEIGHT_DECAY: float = 0.05
    GRAD_CLIP_NORM: float | None = None
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    USE_AMP: bool = True
    CHANNELS_LAST: bool = True
    EARLY_STOPPING_PATIENCE: int = 3
    SEED: int = 42

    # Parameter Augmentasi Spasial & Normalisasi (Tabel 4.8)
    RRC_SCALE: tuple = (0.4, 1.0)
    RRC_RATIO: tuple = (3 / 4, 4 / 3)
    IMAGENET_MEAN: tuple = (0.485, 0.456, 0.406)
    IMAGENET_STD: tuple = (0.229, 0.224, 0.225)


cfg = Cfg()
NUM_CLASSES = len(cfg.CLASS_NAMES)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect_samples(root: str) -> tuple[list[str], list[int]]:
    root_p = Path(root).expanduser().resolve()
    paths, labels = [], []
    for idx, name in enumerate(cfg.CLASS_NAMES):
        class_dir = root_p / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        files = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        paths += [str(p) for p in files]
        labels += [idx] * len(files)
    return paths, labels


def class_counts(labels: list[int]) -> dict[str, int]:
    c = Counter(labels)
    return {name: c[i] for i, name in enumerate(cfg.CLASS_NAMES)}


class RandomRot90(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, random.randint(0, 3), dims=(-2, -1))


class PairedDataset(torch.utils.data.Dataset):
    """
    Menyiapkan pasangan citra asli (x) dan citra teraugmentasi RandStainNA (x_aug)
    dengan transformasi spasial sinkron melalui torch.cat() dan torch.split() (Subbab 3.2.6).
    """
    def __init__(self, paths: list[str], labels: list[int], *, train: bool) -> None:
        self.paths, self.labels, self.train = paths, labels, train
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.normalize = v2.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)

        if train:
            root = Path(cfg.RANDSTAINNA_OFFLINE_ROOT)
            self.stain_paths = [root / cfg.CLASS_NAMES[y] / Path(p).name for p, y in zip(paths, labels)]
            self.spatial = v2.Compose([
                v2.RandomResizedCrop(
                    cfg.INPUT_SIZE, scale=cfg.RRC_SCALE, ratio=cfg.RRC_RATIO,
                    interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                v2.RandomHorizontalFlip(p=0.5),
                RandomRot90(),
            ])
        else:
            self.resize = v2.Resize(cfg.INPUT_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as im:
            return self.to_tensor(im.convert("RGB"))

    def __getitem__(self, i: int):
        y = self.labels[i]
        x = self._load(self.paths[i])
        if not self.train:
            return self.normalize(self.resize(x)), y
        x_aug = self._load(self.stain_paths[i])
        both = self.spatial(torch.cat([x, x_aug], dim=0))
        x_sync, x_aug_sync = torch.split(both, 3, dim=0)
        return self.normalize(x_sync), self.normalize(x_aug_sync), y


def load_ssl_weights(model: nn.Module, path: str) -> None:
    ckpt = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt.get("encoder_q", ckpt)))
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}

    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        clean_k = k.split("backbone.")[-1].split("encoder.")[-1].split("module.")[-1]
        if clean_k in target and not clean_k.startswith("head") and v.shape == target[clean_k].shape:
            mapped[clean_k] = v

    target_keys = [k for k in target if not k.startswith("head")]
    coverage = len(mapped) / len(target_keys) if target_keys else 0.0
    print(f"[CKPT] Matched {len(mapped)}/{len(target_keys)} backbone tensors ({coverage:.2%})")
    model.load_state_dict(mapped, strict=False)


def set_trainable_state(model: nn.Module, stage: int) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = ("head" in name) if stage == 1 else True


def set_lr_dynamic(optimizer: torch.optim.Optimizer, epoch_frac: float,
                   base_lr: float, min_lr: float, warmup: int, total_epochs: int) -> float:
    if warmup > 0 and epoch_frac < warmup:
        lr = base_lr * epoch_frac / warmup
    else:
        prog = min(max((epoch_frac - warmup) / max(1e-12, total_epochs - warmup), 0.0), 1.0)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
    return lr


def autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=cfg.USE_AMP and device.type == "cuda")


def to_dev(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    t = t.to(device, non_blocking=True)
    return t.contiguous(memory_format=torch.channels_last) if cfg.CHANNELS_LAST and t.ndim == 4 else t


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, stage_name,
                    base_lr, min_lr, warmup_epochs, total_epochs) -> dict[str, float]:
    if stage_name == "LP":
        model.eval()
        model.head.train()
    else:
        model.train()

    meters = {k: torchmetrics.MeanMetric().to(device) for k in ("loss", "ce", "cons")}
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(device)
    n_batches, lr = len(loader), 0.0

    bar = tqdm(loader, desc=f"[{stage_name}] Epoch {epoch + 1:02d}/{total_epochs:02d}", dynamic_ncols=True)
    for step, (x, x_aug, y) in enumerate(bar):
        x, x_aug, y = to_dev(x, device), to_dev(x_aug, device), to_dev(y, device)
        lr = set_lr_dynamic(optimizer, epoch + step / n_batches, base_lr, min_lr, warmup_epochs, total_epochs)

        with autocast(device):
            features = model.forward_features(torch.cat([x, x_aug], dim=0))
            z_all = model.forward_head(features, pre_logits=True)
            logits_all = model.get_classifier()(z_all)

            # Dual Cross-Entropy Loss (L_CE)
            ce = F.cross_entropy(logits_all.float(), torch.cat([y, y], dim=0))

            # Normalized Feature Consistency Loss (L_cons) - Persamaan 2.4
            z, z_aug = z_all.float().chunk(2, dim=0)
            z_norm = F.normalize(z, p=2, dim=1)
            z_aug_norm = F.normalize(z_aug, p=2, dim=1)
            cons = (1.0 - F.cosine_similarity(z_norm, z_aug_norm, dim=1)).mean()

            # Multi-Objective Loss (L_total) - Persamaan 2.5
            loss = ce + cfg.LAMBDA_CONSISTENCY * cons

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.GRAD_CLIP_NORM is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        meters["loss"].update(loss.detach())
        meters["ce"].update(ce.detach())
        meters["cons"].update(cons.detach())
        f1.update(logits_all[: y.numel()].argmax(1), y)

    return {
        "loss": meters["loss"].compute().item(),
        "ce": meters["ce"].compute().item(),
        "cons": meters["cons"].compute().item(),
        "macro_f1": f1.compute().item(),
        "lr": lr,
    }


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict[str, float], dict[str, Any], np.ndarray]:
    model.eval()
    loss_m = torchmetrics.MeanMetric().to(device)
    cm_m = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES).to(device)
    for x, y in loader:
        x, y = to_dev(x, device), to_dev(y, device)
        with autocast(device):
            logits = model(x)
        loss_m.update(F.cross_entropy(logits.float(), y), weight=y.numel())
        cm_m.update(logits.argmax(1), y)

    cm = cm_m.compute().float()
    tp = cm.diag()
    prec = tp / (cm.sum(0)).clamp(min=1)
    rec = tp / (cm.sum(1)).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    report = {
        n: {
            "precision": prec[i].item(), "recall": rec[i].item(),
            "f1-score": f1[i].item(), "support": int(cm.sum(1)[i].item())
        }
        for i, n in enumerate(cfg.CLASS_NAMES)
    }
    metrics = {
        "loss": loss_m.compute().item(),
        "macro_f1": f1.mean().item(),
        "accuracy": (tp.sum() / cm.sum()).item(),
        "macro_precision": prec.mean().item(),
        "macro_recall": rec.mean().item(),
    }
    return metrics, report, cm.cpu().numpy()


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_confusion_matrix(path: Path, cm: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *cfg.CLASS_NAMES])
        for name, row in zip(cfg.CLASS_NAMES, cm):
            w.writerow([name, *(int(v) for v in row)])


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)


def run_stage(model, stage_name: str, stage_dir: Path,
              optimizer, scaler, train_loader, val_loaders, device,
              epochs: int, base_lr: float, min_lr: float, warmup_epochs: int):
    stage_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score, best_epoch, bad_epochs = (-1.0, -float("inf")), -1, 0

    for epoch in range(epochs):
        t_start = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            stage_name, base_lr, min_lr, warmup_epochs, epochs
        )
        row: dict[str, Any] = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in tr.items()}}
        results = {}
        for name, loader in val_loaders.items():
            m, rep, cm = evaluate(model, loader, device)
            results[name] = (rep, cm)
            row.update({f"val_{name}_{k}": v for k, v in m.items()})

        row["val_macro_f1"] = float(np.mean([row[f"val_{n}_macro_f1"] for n in val_loaders]))
        row["val_loss"] = float(np.mean([row[f"val_{n}_loss"] for n in val_loaders]))
        row["epoch_time_min"] = (time.time() - t_start) / 60.0
        history.append(row)
        save_history(stage_dir / "history.csv", history)

        score = (row["val_macro_f1"], -row["val_loss"])
        print(
            f"[{stage_name}] Epoch {epoch + 1:02d}/{epochs:02d} | LR: {tr['lr']:.2e} | "
            f"Train CE: {tr['ce']:.4f} | Cons: {tr['cons']:.4f} | Total: {tr['loss']:.4f} | "
            f"Train F1: {tr['macro_f1']*100:.2f}% | Val Loss: {row['val_loss']:.4f} | "
            f"Val F1: {row['val_macro_f1']*100:.2f}% | Val Acc: {row['val_orig_accuracy']*100:.2f}%"
        )

        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            atomic_save(
                {"model": model.state_dict(), "epoch": epoch + 1, "val_macro_f1": row["val_macro_f1"]},
                stage_dir / "best_model.pth"
            )
            for name, (rep, cm) in results.items():
                save_json(stage_dir / f"best_val_{name}_classification_report.json", rep)
                save_confusion_matrix(stage_dir / f"best_val_{name}_confusion_matrix.csv", cm)
        else:
            bad_epochs += 1

        atomic_save({"model": model.state_dict(), "epoch": epoch + 1}, stage_dir / "last_model.pth")
        if bad_epochs >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"[{stage_name}] Early stopping triggered ({bad_epochs} epochs without improvement).")
            break

    print(f"[{stage_name}] Memuat checkpoint terbaik dari Epoch {best_epoch + 1} (Val F1: {best_score[0]*100:.2f}%)")
    best_ckpt = torch.load(stage_dir / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])


def main() -> None:
    out = Path(cfg.OUTPUT_DIR).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    timm.utils.random_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True

    train_p, train_y = collect_samples(cfg.TRAIN_DATA_ROOT)
    val_sets = {"orig": collect_samples(cfg.VAL_DATA_ROOT)}

    save_json(out / "split_summary.json", {
        "train": {"n": len(train_p), "per_class": class_counts(train_y)},
        **{f"val_{k}": {"n": len(p), "per_class": class_counts(y)} for k, (p, y) in val_sets.items()},
    })
    save_json(out / "experiment_config.json", asdict(cfg))

    common = dict(num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=cfg.NUM_WORKERS > 0)
    train_loader = torch.utils.data.DataLoader(
        PairedDataset(train_p, train_y, train=True),
        batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, **common
    )
    val_loaders = {
        k: torch.utils.data.DataLoader(
            PairedDataset(p, y, train=False), batch_size=cfg.BATCH_SIZE * 8, shuffle=False, **common
        )
        for k, (p, y) in val_sets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "convnextv2_tiny", pretrained=False, num_classes=NUM_CLASSES,
        drop_path_rate=cfg.DROP_PATH_RATE, head_init_scale=1e-3
    )
    load_ssl_weights(model, cfg.PRETRAIN_CHECKPOINT_PATH)
    model = model.to(device)
    if cfg.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    # Stage 1: Linear Probing
    set_trainable_state(model, stage=1)
    opt_lp = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LP_LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scaler_lp = torch.amp.GradScaler(device.type, enabled=cfg.USE_AMP and device.type == "cuda")
    run_stage(
        model=model, stage_name="LP", stage_dir=out / "stage1_lp",
        optimizer=opt_lp, scaler=scaler_lp, train_loader=train_loader, val_loaders=val_loaders, device=device,
        epochs=cfg.LP_EPOCHS, base_lr=cfg.LP_LEARNING_RATE, min_lr=cfg.LP_MIN_LR, warmup_epochs=cfg.LP_WARMUP_EPOCHS
    )

    # Stage 2: Full Fine-Tuning
    set_trainable_state(model, stage=2)
    opt_ft = create_optimizer_v2(
        model, opt="adamw", lr=cfg.FT_LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY, layer_decay=cfg.LAYER_DECAY
    )
    scaler_ft = torch.amp.GradScaler(device.type, enabled=cfg.USE_AMP and device.type == "cuda")
    run_stage(
        model=model, stage_name="FT", stage_dir=out / "stage2_ft",
        optimizer=opt_ft, scaler=scaler_ft, train_loader=train_loader, val_loaders=val_loaders, device=device,
        epochs=cfg.FT_EPOCHS, base_lr=cfg.FT_LEARNING_RATE, min_lr=cfg.FT_MIN_LR, warmup_epochs=cfg.FT_WARMUP_EPOCHS
    )

    # Ekspor model terbaik akhir sebagai best_model_sirl3d.pth (Subbab 3.5.2 & 4.5.2)
    shutil.copyfile(out / "stage2_ft" / "best_model.pth", out / "best_model_sirl3d.pth")
    print(f"[DONE] Model terbaik diekspor ke: {out / 'best_model_sirl3d.pth'}")


if __name__ == "__main__":
    main()
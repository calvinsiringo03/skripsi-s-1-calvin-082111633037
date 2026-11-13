import os
import glob
import random
import time
import multiprocessing
import sys
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

cv2.setNumThreads(0)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

RANDSTAINNA_DIR = r'C:\Calvin_082111633037\RandStainNA-master'
if RANDSTAINNA_DIR not in sys.path:
    sys.path.insert(0, RANDSTAINNA_DIR)
from randstainna import RandStainNA

# =====================================================================
# 1. KONFIGURASI PARAMETER MO-COV2 & OPTIMISASI (TABEL 4.10)
# =====================================================================
RESUME_TRAINING = False
EPOCHS_TO_RUN   = 50
MOCO_DIM        = 128
MOCO_K          = 65536
MOCO_M          = 0.999
MOCO_T          = 0.2
BATCH_SIZE      = 32
BASE_LR         = 2e-4
MIN_LR          = 1e-5
WARMUP_EPOCHS   = 3
WEIGHT_DECAY    = 1e-4
TOTAL_EPOCHS    = 50
CHECKPOINT_FREQ = 5

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

WORKSPACE_DIR = r'G:\My Drive\SIRL3D_Workspace'
LOCAL_DATA_DIR = r'C:\Calvin_082111633037\local_tcga_patches'
PATHS = {
    'tcga_patches': LOCAL_DATA_DIR,
    'yaml_stats': os.path.join(WORKSPACE_DIR, 'metadata', 'TCGA_LAB_randomTrue_n0.yaml'),
    'checkpoints': os.path.join(WORKSPACE_DIR, 'checkpoints', 'phase1_pretraining'),
    'logs': os.path.join(WORKSPACE_DIR, 'results', 'logs'),
}


# =====================================================================
# 2. AUGMENTASI DUA CABANG (SUBBAB 3.2.5 & TABEL 4.7)
# =====================================================================
class TwoViewTransform:
    def __init__(self, yaml_path, input_size=224):
        self.stain_aug = RandStainNA(
            yaml_file=yaml_path,
            std_hyper=0,
            probability=0.5,
            distribution="normal",
            is_train=True,
        )
        self.spatial_noise_aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=input_size, scale=(0.4, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ]),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        view1_bgr = self.stain_aug(x)
        view2_bgr = self.stain_aug(x)
        view1_rgb = cv2.cvtColor(view1_bgr, cv2.COLOR_BGR2RGB)
        view2_rgb = cv2.cvtColor(view2_bgr, cv2.COLOR_BGR2RGB)
        q = self.spatial_noise_aug(view1_rgb)
        k = self.spatial_noise_aug(view2_rgb)
        return [q, k]


class TCGAPreTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        for folder in os.scandir(root_dir):
            if folder.is_dir():
                for file in os.scandir(folder.path):
                    if file.is_file() and file.name.endswith('.png') and file.stat().st_size > 0:
                        self.image_paths.append(file.path)
        self.transform = transform
        print(f"[Dataset] Total {len(self.image_paths):,} patch WSI TCGA siap dilatih.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# =====================================================================
# 3. ARSITEKTUR MOCO V2 DENGAN BACKBONE CONVNEXT V2-TINY (SUBBAB 3.4.1)
# =====================================================================
class MoCoV2_ConvNeXt(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.2):
        super(MoCoV2_ConvNeXt, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=0, drop_path_rate=0.0)
        self.encoder_k = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=0, drop_path_rate=0.0)
        hidden_dim = 2048
        self.projector_q = nn.Sequential(
            nn.Linear(768, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        self.projector_k = nn.Sequential(
            nn.Linear(768, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.m).add_(param_q.data, alpha=1.0 - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.mul_(self.m).add_(param_q.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k):
        feat_q = self.encoder_q(im_q)
        q = F.normalize(self.projector_q(feat_q), dim=1)
        with torch.no_grad():
            feat_k = self.encoder_k(im_k)
            k = F.normalize(self.projector_k(feat_k), dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        self._dequeue_and_enqueue(k)
        return logits, q, feat_q


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


# =====================================================================
# 4. SIKLUS PELATIHAN PRE-TRAINING (SUBBAB 3.4.2 & 4.4)
# =====================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.set_num_threads(2)
    os.makedirs(PATHS['checkpoints'], exist_ok=True)
    os.makedirs(PATHS['logs'], exist_ok=True)

    g = torch.Generator()
    g.manual_seed(SEED)
    dataset = TCGAPreTrainDataset(
        root_dir=PATHS['tcga_patches'],
        transform=TwoViewTransform(PATHS['yaml_stats'], input_size=224)
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
        prefetch_factor=4, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g, persistent_workers=True
    )

    torch.set_float32_matmul_precision("high")
    model = MoCoV2_ConvNeXt(dim=MOCO_DIM, K=MOCO_K, m=MOCO_M, T=MOCO_T).cuda().to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(get_param_groups(model, WEIGHT_DECAY), lr=BASE_LR)
    steps_per_epoch = len(dataloader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = TOTAL_EPOCHS * steps_per_epoch

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(total_steps - warmup_steps), eta_min=MIN_LR
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss().cuda()

    log_path = os.path.join(PATHS['logs'], 'ssl_pretraining_log.csv')
    log_cols = ['epoch', 'loss', 'acc', 'q_std', 'feat_std', 'pos_sim', 'neg_sim', 'margin', 'lr']
    pd.DataFrame(columns=log_cols).to_csv(log_path, index=False)

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_loss, total_acc, total_q_std = 0.0, 0.0, 0.0
        total_feat_std, total_pos_sim, total_neg_sim, total_margin = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", leave=False, dynamic_ncols=True)
        for images in pbar:
            optimizer.zero_grad(set_to_none=True)
            im_q = images[0].cuda(non_blocking=True).to(memory_format=torch.channels_last)
            im_k = images[1].cuda(non_blocking=True).to(memory_format=torch.channels_last)

            with torch.amp.autocast('cuda'):
                logits, q, feat_q = model(im_q, im_k)
                logits = logits.to(torch.float32)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
                loss = criterion(logits, labels)

            with torch.no_grad():
                acc_t = (logits.argmax(dim=1) == 0).float().mean()
                pos_sim_t = (logits[:, 0] * MOCO_T).mean()
                neg_sim_t = (logits[:, 1:] * MOCO_T).mean()
                hard_neg_sim_i = logits[:, 1:].max(dim=1).values * MOCO_T
                margin_t = ((logits[:, 0] * MOCO_T) - hard_neg_sim_i).mean()
                feat_std_t = feat_q.to(torch.float32).std(dim=0).mean()
                q_std_t = q.to(torch.float32).std(dim=0).mean()
                acc, pos_sim, neg_sim, margin, feat_std, q_std = torch.stack([
                    acc_t, pos_sim_t, neg_sim_t, margin_t, feat_std_t, q_std_t
                ]).cpu().numpy()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            model._momentum_update_key_encoder()
            scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            total_q_std += q_std
            total_feat_std += feat_std
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            total_margin += margin

        N = len(dataloader)
        avg_loss = total_loss / N
        current_lr = scheduler.get_last_lr()[0]
        elapsed = (time.time() - start_time) / 60.0
        print(f"Epoch {epoch+1:02d}/{TOTAL_EPOCHS} | InfoNCE Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.2f}m")

        pd.DataFrame({
            'epoch': [epoch + 1], 'loss': [avg_loss], 'acc': [total_acc / N],
            'q_std': [total_q_std / N], 'feat_std': [total_feat_std / N],
            'pos_sim': [total_pos_sim / N], 'neg_sim': [total_neg_sim / N],
            'margin': [total_margin / N], 'lr': [current_lr]
        }).to_csv(log_path, mode='a', header=False, index=False)

        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            ckpt_name = os.path.join(PATHS['checkpoints'], f'checkpoint_epoch_{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'loss': avg_loss,
            }, ckpt_name)

        if (epoch + 1) == TOTAL_EPOCHS:
            final_weights_path = os.path.join(PATHS['checkpoints'], 'convnext_v2_tiny_ssl.pth')
            torch.save(model.encoder_q.state_dict(), final_weights_path)
            print(f"[DONE] Bobot backbone disimpan ke: {final_weights_path}")
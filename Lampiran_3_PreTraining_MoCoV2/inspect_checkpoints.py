import os
import glob
import math
import torch
import numpy as np
import pandas as pd

WORKSPACE_DIR = r'G:\My Drive\SIRL3D_Workspace'
CKPT_DIR = os.path.join(WORKSPACE_DIR, 'checkpoints', 'phase1_pretraining')
OUTPUT_CSV = os.path.join(WORKSPACE_DIR, 'results', 'logs', 'raw_checkpoint_audit_summary.csv')


def compute_effective_rank(queue_tensor):
    """
    Menghitung Effective Rank (Roy & Vetterli, 2007) dari matriks Memory Queue [128, 65536]
    menggunakan Singular Value Decomposition (SVD).
    """
    q_float = queue_tensor.float()
    S = torch.linalg.svdvals(q_float)
    p = S / torch.sum(S)
    p = p[p > 1e-12]
    entropy = -torch.sum(p * torch.log(p)).item()
    return math.exp(entropy)


def inspect_single_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state' not in ckpt:
        total_params = sum(p.numel() for p in ckpt.values() if isinstance(p, torch.Tensor))
        print(f"[Pure Backbone Weights] {os.path.basename(ckpt_path)} | Parameters: {total_params:,}")
        return None

    epoch = ckpt.get('epoch', -1)
    loss = ckpt.get('loss', float('nan'))
    model_state = ckpt['model_state']
    opt_state = ckpt.get('optimizer_state', {})
    scaler_state = ckpt.get('scaler_state', {})

    q_k_l2_diff_sq = 0.0
    enc_q_weight_norm_sq = 0.0
    for k, v in model_state.items():
        if k.startswith('encoder_q.'):
            enc_q_weight_norm_sq += torch.sum(v.float() ** 2).item()
            k_key = k.replace('encoder_q.', 'encoder_k.', 1)
            if k_key in model_state:
                diff = v.float() - model_state[k_key].float()
                q_k_l2_diff_sq += torch.sum(diff ** 2).item()

    enc_q_l2_norm = math.sqrt(enc_q_weight_norm_sq)
    q_k_l2_dist = math.sqrt(q_k_l2_diff_sq)

    queue = model_state['queue'].float()
    queue_ptr = int(model_state['queue_ptr'].item())
    queue_feat_std = queue.std(dim=1).mean().item()
    eff_rank = compute_effective_rank(queue)

    sample_idx = torch.randperm(queue.shape[1])[:2048]
    q_sub = queue[:, sample_idx]
    sim_matrix = torch.matmul(q_sub.T, q_sub)
    mask = ~torch.eye(2048, dtype=torch.bool)
    queue_offdiag_sim_mean = sim_matrix[mask].mean().item()
    queue_offdiag_sim_std = sim_matrix[mask].std().item()

    param_groups = opt_state.get('param_groups', [])
    current_lr = param_groups[0]['lr'] if param_groups else float('nan')
    opt_steps = 0
    if 'state' in opt_state and len(opt_state['state']) > 0:
        first_param_state = next(iter(opt_state['state'].values()))
        opt_steps = int(first_param_state.get('step', 0))

    return {
        'epoch': epoch,
        'global_step': opt_steps,
        'loss': round(loss, 6),
        'lr': current_lr,
        'amp_scale': scaler_state.get('scale', float('nan')),
        'enc_q_l2_norm': round(enc_q_l2_norm, 4),
        'ema_q_k_l2_dist': round(q_k_l2_dist, 6),
        'queue_ptr': queue_ptr,
        'queue_feat_std': round(queue_feat_std, 6),
        'queue_effective_rank': round(eff_rank, 2),
        'queue_neg_sim_mean': round(queue_offdiag_sim_mean, 6),
        'queue_neg_sim_std': round(queue_offdiag_sim_std, 4),
        'file_size_mb': round(os.path.getsize(ckpt_path) / (1024**2), 2)
    }


if __name__ == '__main__':
    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, 'checkpoint_epoch_*.pth')))
    records = [res for path in ckpt_files if (res := inspect_single_checkpoint(path)) is not None]
    if records:
        df_summary = pd.DataFrame(records)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_summary.to_csv(OUTPUT_CSV, index=False)
        print(df_summary.to_string(index=False))
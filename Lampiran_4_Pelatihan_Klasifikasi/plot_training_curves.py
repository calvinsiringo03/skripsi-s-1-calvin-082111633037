from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
LP_HISTORY_PATH = BASE_DIR / "stage1_lp_history.csv"
FT_HISTORY_PATH = BASE_DIR / "stage2_ft_history.csv"
OUTPUT_IMAGE_PATH = BASE_DIR / "Gambar_4_9_Kurva_Pelatihan_Klasifikasi.png"

df_lp = pd.read_csv(LP_HISTORY_PATH)
df_ft = pd.read_csv(FT_HISTORY_PATH)

n_lp = len(df_lp)
n_ft = len(df_ft)
epochs_total = np.arange(1, n_lp + n_ft + 1)

train_loss_all = np.concatenate([df_lp["train_loss"].values, df_ft["train_loss"].values])
train_ce_all = np.concatenate([df_lp["train_ce"].values, df_ft["train_ce"].values])
train_cons_all = np.concatenate([df_lp["train_cons"].values, df_ft["train_cons"].values])
val_loss_all = np.concatenate([df_lp["val_loss"].values, df_ft["val_loss"].values])

train_f1_all = np.concatenate([df_lp["train_macro_f1"].values, df_ft["train_macro_f1"].values]) * 100
val_f1_all = np.concatenate([df_lp["val_macro_f1"].values, df_ft["val_macro_f1"].values]) * 100

best_lp_idx = int(df_lp["val_macro_f1"].idxmax())
best_lp_epoch = int(df_lp.loc[best_lp_idx, "epoch"])
best_lp_f1 = val_f1_all[best_lp_idx]

best_ft_idx = int(df_ft["val_macro_f1"].idxmax())
best_ft_epoch_local = int(df_ft.loc[best_ft_idx, "epoch"])
best_ft_epoch_cum = n_lp + best_ft_epoch_local
best_ft_f1 = val_f1_all[n_lp + best_ft_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

ax1.plot(epochs_total, train_loss_all, "o-", label=r"Train Total Loss ($\mathcal{L}_{total}$)", color="#1f77b4", linewidth=2, markersize=4)
ax1.plot(epochs_total, train_ce_all, "s--", label=r"Train CE Loss ($\mathcal{L}_{CE}$)", color="#ff7f0e", linewidth=1.5, markersize=4)
ax1.plot(epochs_total, train_cons_all, "^-.", label=r"Train Consistency Loss ($\mathcal{L}_{cons}$)", color="#2ca02c", linewidth=1.5, markersize=4)
ax1.plot(epochs_total, val_loss_all, "d-", label="Validation Loss", color="#d62728", linewidth=2, markersize=4)
ax1.axvline(x=n_lp + 0.5, color="gray", linestyle=":", linewidth=1.5, label=r"Transisi LP $\rightarrow$ FT")
ax1.set_title("Dinamika Fungsi Loss (Stage 1 LP & Stage 2 FT)", fontsize=11, fontweight="bold")
ax1.set_xlabel(f"Epoch Kumulatif (1-{n_lp}: LP, {n_lp+1}-{n_lp+n_ft}: FT)", fontsize=10)
ax1.set_ylabel("Nilai Loss", fontsize=10)
ax1.set_xticks(range(1, n_lp + n_ft + 1, 2))
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=8, loc="upper right")

ax2.plot(epochs_total, train_f1_all, "o--", label="Train Macro F1-Score (%)", color="#1f77b4", linewidth=1.8, markersize=4)
ax2.plot(epochs_total, val_f1_all, "s-", label="Validation Macro F1-Score (%)", color="#d62728", linewidth=2, markersize=5)
ax2.axvline(x=n_lp + 0.5, color="gray", linestyle=":", linewidth=1.5, label=r"Transisi LP $\rightarrow$ FT")
ax2.scatter([best_lp_epoch, best_ft_epoch_cum], [best_lp_f1, best_ft_f1], color="black", zorder=5, s=50,
            label=f"Best Checkpoint (Ep {best_lp_epoch} LP & Ep {best_ft_epoch_local} FT)")
ax2.annotate(f"Best LP\n({best_lp_f1:.2f}%)", xy=(best_lp_epoch, best_lp_f1),
             xytext=(best_lp_epoch - 1.7, best_lp_f1 - 0.8), arrowprops=dict(arrowstyle="->", lw=1), fontsize=8, fontweight="bold")
ax2.annotate(f"Best FT\n({best_ft_f1:.2f}%)", xy=(best_ft_epoch_cum, best_ft_f1),
             xytext=(best_ft_epoch_cum - 2.8, best_ft_f1 - 0.8), arrowprops=dict(arrowstyle="->", lw=1), fontsize=8, fontweight="bold")
ax2.set_title("Progres Macro F1-Score (Stage 1 LP & Stage 2 FT)", fontsize=11, fontweight="bold")
ax2.set_xlabel(f"Epoch Kumulatif (1-{n_lp}: LP, {n_lp+1}-{n_lp+n_ft}: FT)", fontsize=10)
ax2.set_ylabel("Macro F1-Score (%)", fontsize=10)
ax2.set_xticks(range(1, n_lp + n_ft + 1, 2))
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
plt.show()
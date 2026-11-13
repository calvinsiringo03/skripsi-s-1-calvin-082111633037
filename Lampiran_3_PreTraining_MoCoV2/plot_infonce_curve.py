from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Konfigurasi Path Berkas CSV dan Output Gambar
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "raw_checkpoint_audit_summary.csv"
OUTPUT_IMAGE_PATH = BASE_DIR / "Gambar_4_8_Kurva_InfoNCE_Loss.png"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Berkas CSV tidak ditemukan di: {CSV_PATH}")

# 2. Membaca Data Langsung dari Berkas CSV
df = pd.read_csv(CSV_PATH)

# 3. Membuat Plot Garis Konvergensi InfoNCE Loss
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(
    df["epoch"], df["loss"],
    marker="o", linestyle="-", color="#1f77b4",
    linewidth=2.5, markersize=8, label="InfoNCE Loss"
)

plt.title("Kurva InfoNCE Loss terhadap Epoch", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("InfoNCE Loss", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(df["epoch"])

# 4. Anotasi pada Titik Awal (Epoch 5) dan Titik Akhir (Epoch 50)
plt.annotate(
    f"{df['loss'].iloc[0]:.4f}",
    (df["epoch"].iloc[0], df["loss"].iloc[0]),
    textcoords="offset points", xytext=(10, 5), ha="left",
    fontsize=10, fontweight="bold", color="#333333"
)

plt.annotate(
    f"{df['loss'].iloc[-1]:.4f}",
    (df["epoch"].iloc[-1], df["loss"].iloc[-1]),
    textcoords="offset points", xytext=(10, 5), ha="left",
    fontsize=10, fontweight="bold", color="#333333"
)

ax = plt.gca()
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
print(f"Grafik berhasil disimpan di: {OUTPUT_IMAGE_PATH}")
plt.show()

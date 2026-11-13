# Lampiran 2 — Statistik Ruang Warna LAB RandStainNA

Direktori ini memuat kode program ekstraksi parameter statistik warna serta keluaran berkas konfigurasi `.yaml` menggunakan metode **RandStainNA** pada ruang warna **LAB** (Subbab 3.2.4 dan Subbab 4.2.4). Parameter rata-rata ($\mu$) dan standar deviasi ($\sigma$) yang diekstraksi berfungsi sebagai batasan distribusi matematis dalam membangkitkan _virtual stain template_ secara dinamis pada fase _self-supervised pre-training_ dan fase pelatihan klasifikasi.

## Daftar Berkas

1. [`datasets_statistics.py`](./datasets_statistics.py) — Skrip perhitungan nilai rata-rata (_mean_) dan standar deviasi (_std_) intensitas piksel pada kanal ruang warna `L`, `A`, dan `B` untuk seluruh _patch_ citra jaringan.
2. [`TCGA_LAB_randomTrue_n0.yaml`](./TCGA_LAB_randomTrue_n0.yaml) — Berkas konfigurasi parameter statistik warna hasil ekstraksi dari **402.205 _patch_ WSI TCGA** (gabungan TCGA-COAD dan TCGA-READ) untuk augmentasi warna pada tahap _self-supervised pre-training_.
3. [`NCT_LAB_randomTrue_n0.yaml`](./NCT_LAB_randomTrue_n0.yaml) — Berkas konfigurasi parameter statistik warna hasil ekstraksi dari **89.434 _patch_ jaringan NCT-CRC** (mengecualikan 10.566 citra dari kelas latar belakang `BACK`) untuk augmentasi warna pada tahap pelatihan model klasifikasi.

## Perintah Eksekusi Terminal

```bash
# 1. Ekstraksi Statistik Warna Dataset Pre-Training (TCGA: 402.205 Patch)
python preprocess/datasets_statistics.py \
  --data-dir "C:\Calvin_082111633037\local_tcga_patches" \
  --save-dir "G:\My Drive\SIRL3D_Workspace\metadata" \
  --dataset-name TCGA_LAB_randomTrue_n0 \
  --methods Reinhard --color-space LAB --random --n 0

# 2. Ekstraksi Statistik Warna Dataset Klasifikasi (NCT-CRC: 89.434 Patch Tanpa Kelas BACK)
python preprocess/datasets_statistics.py \
  --data-dir "C:\Calvin_082111633037\training\NCT-CRC" \
  --save-dir "G:\My Drive\SIRL3D_Workspace\metadata" \
  --dataset-name NCT_LAB_randomTrue_n0 \
  --methods Reinhard --color-space LAB --random --n 0
```

## Ringkasan Perbandingan Parameter Statistik Warna LAB (Tabel 4.6)

| Kanal | Target Statistik | Parameter | Nilai TCGA | Nilai NCT-CRC |
| :---: | :--------------- | :-------: | :--------: | :-----------: |
| **L** | Pusat (_Mean_)   |   $\mu$   |  147,568   |    157,982    |
| **L** | Pusat (_Mean_)   | $\sigma$  |   33,371   |    48,810     |
| **L** | Variasi (_Std_)  |   $\mu$   |   43,449   |    36,929     |
| **L** | Variasi (_Std_)  | $\sigma$  |   12,688   |    14,381     |
| **A** | Pusat (_Mean_)   |   $\mu$   |  155,691   |    151,189    |
| **A** | Pusat (_Mean_)   | $\sigma$  |   8,982    |    10,952     |
| **A** | Variasi (_Std_)  |   $\mu$   |   10,446   |     8,141     |
| **A** | Variasi (_Std_)  | $\sigma$  |   3,161    |     2,831     |
| **B** | Pusat (_Mean_)   |   $\mu$   |  117,451   |    116,810    |
| **B** | Pusat (_Mean_)   | $\sigma$  |   7,213    |     6,628     |
| **B** | Variasi (_Std_)  |   $\mu$   |   7,923    |     6,128     |
| **B** | Variasi (_Std_)  | $\sigma$  |   2,033    |     2,010     |

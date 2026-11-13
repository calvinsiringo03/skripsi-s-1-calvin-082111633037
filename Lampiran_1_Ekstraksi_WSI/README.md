# Lampiran 1 — Metadata Ekstraksi dan Filtrasi Kualitas WSI TCGA

Direktori ini memuat kode program dan metadata hasil ekstraksi _patch_ serta filtrasi kualitas pada 50 _Whole Slide Image_ (WSI) tanpa label dari repositori _The Cancer Genome Atlas_ (TCGA) melalui infrastruktur _Imaging Data Commons_ (IDC), yang terdiri atas 25 WSI proyek **TCGA-COAD** dan 25 WSI proyek **TCGA-READ** (Subbab 3.2.1 dan Subbab 4.2.1).

## Daftar Berkas

1. [`wsi_extraction_filtration.py`](./wsi_extraction_filtration.py) — Skrip utama untuk mengunduh sediaan WSI (`.dcm`), melakukan pemotongan _grid-based patching_ berukuran `224 x 224` piksel pada magnifikasi `20x` (`0,5 mpp`) menggunakan pustaka `tiatoolbox`, serta menjalankan penyaringan kualitas otomatis (_background & artifact filtration_ dan _foreground ratio thresholding_ minimal 50%).
2. [`tcga_wsi_metadata.csv`](./tcga_wsi_metadata.csv) — Tabel metadata lengkap untuk 50 WSI yang mencakup kode pasien (`PatientID`), proyek dataset (`collection_id`), tanggal pemindaian (`SeriesDate`), produsen pemindai (`Manufacturer`), jumlah _patch_ sebelum filtrasi (`Before Filter`), jumlah _patch_ setelah filtrasi (`After Filter`), dan persentase kelolosan (`Total Passed`).

## Struktur Penyimpanan Lokal

Seluruh _patch_ yang lolos tahapan filtrasi kualitas disimpan secara langsung dalam format `.png` ke dalam 50 folder terpisah berdasarkan identitas WSI pada direktori lokal:

```text
C:\Calvin_082111633037\local_tcga_patches\
├── TCGA-AA-A01Q\
│   ├── TCGA-AA-A01Q_patch_00000.png
│   ├── TCGA-AA-A01Q_patch_00001.png
│   └── ... (6.634 berkas .png)
├── TCGA-T9-A92H\
│   └── ... (194 berkas .png)
...
└── TCGA-DY-A1DE\
    └── ... (9.242 berkas .png)
```

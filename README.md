# Klasifikasi Jaringan Kanker Kolorektal Lintas-Domain Menggunakan ConvNeXt Berbasis Self-Supervised Learning dan Regularisasi Konsistensi Representasi

Repositori ini memuat kode program (sintaks), konfigurasi parameter, dan keluaran (*output*) eksperimen untuk penelitian Skripsi pada Program Studi S-1 Sistem Informasi, Departemen Matematika, Fakultas Sains dan Teknologi, Universitas Airlangga (2026).

---

### Identitas Peneliti
* **Penyusun:** Calvin Immanuel Siringoringo
* **NIM:** 082111633037
* **Program Studi:** S-1 Sistem Informasi, Fakultas Sains dan Teknologi, Universitas Airlangga
* **Dosen Pembimbing I:** Drs. Eto Wuryanto, DEA.
* **Dosen Pembimbing II:** Indah Werdiningsih, S.Si., M.Kom.

---

### Daftar Direktori Lampiran Skripsi
Setiap folder pada repositori ini disusun berurutan sesuai dengan penomoran **Lampiran** pada naskah skripsi:

1. **[Lampiran 1 — Metadata Ekstraksi dan Filtrasi Kualitas WSI TCGA](./Lampiran_1_Ekstraksi_WSI)**
   Memuat sintaks pemotongan *Whole Slide Image* (WSI) menggunakan `tiatoolbox`, algoritma filtrasi kualitas jaringan, serta tabel metadata hasil ekstraksi 50 WSI TCGA (402.205 *patch*).
2. **[Lampiran 2 — Ekstraksi Parameter Statistik Warna RandStainNA](./Lampiran_2_RandStainNA)**
   Memuat sintaks perhitungan distribusi warna ruang LAB dan keluaran berkas konfigurasi `.yaml` pada dataset TCGA dan NCT-CRC.
3. **[Lampiran 3 — Self-Supervised Pre-Training MoCoV2](./Lampiran_3_PreTraining_MoCoV2)**
   Memuat sintaks arsitektur *Momentum Contrast V2* (MoCoV2) dengan *backbone* ConvNeXt V2-Tiny, perhitungan *InfoNCE Loss*, serta log konvergensi *pre-training* 50 *epoch*.
4. **[Lampiran 4 — Pelatihan Model Klasifikasi (Siamese Network & Consistency Loss)](./Lampiran_4_Pelatihan_Klasifikasi)**
   Memuat sintaks transformasi spasial sinkron, fungsi *Normalized Feature Consistency Loss*, serta log pelatihan tahap *linear probing* dan *full fine-tuning*.
5. **[Lampiran 5 — Evaluasi Model Pengujian In-Domain dan Lintas-Domain](./Lampiran_5_Evaluasi_Model)**
   Memuat sintaks pengujian inferensi serta rincian metrik evaluasi per kelas dan *confusion matrix* pada dataset CRC-VAL dan Kather-16.

import os
import glob
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from idc_index import index
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import get_patch_extractor

# 1. Konfigurasi Seed untuk Reprodusibilitas Eksperimen
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 2. Konfigurasi Direktori Penyimpanan Lokal
BASE_DIR = r"C:\Calvin_082111633037"
RAW_WSI_DIR = os.path.join(BASE_DIR, "raw_tcga_wsi")
LOCAL_PATCH_DIR = os.path.join(BASE_DIR, "local_tcga_patches")
METADATA_CSV_PATH = os.path.join(BASE_DIR, "metadata", "tcga_wsi_metadata.csv")


def filter_patch_quality(img_array, min_tissue_ratio=0.50, blur_threshold=40.0, ink_tolerance=0.02, min_contour_area=100):
    """
    Menyaring kualitas patch WSI melalui dua tahapan berurutan (Subbab 3.2.1):
    1. Background and artifact filtration menggunakan algoritma color thresholding (ruang warna HSV),
       operasi morfologi closing, eliminasi coretan tinta (ink), dan deteksi ketajaman (Laplacian).
    2. Foreground ratio thresholding untuk mempertahankan patch dengan tutupan jaringan seluler >= 50%.
    """
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Tahap 1A: Color thresholding untuk memisahkan jaringan dari latar belakang kaca kosong
    tissue_mask = (s > 25) & (v > 50) & (v < 235)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue_mask_closed = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Eliminasi derau/debu mikroskopis berukuran < 100 piksel
    contours, _ = cv2.findContours(tissue_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(tissue_mask_closed)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            cv2.drawContours(clean_mask, [cnt], -1, 1, -1)

    # Tahap 1B: Eliminasi artefak tinta spidol patolog dan bercak buram (blur)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    ink_mask = (((h > 90) & (h < 130)) | ((h > 35) & (h < 90))) & (s > 80) & (v > 127) & (v < 229)
    if (np.sum(ink_mask) / total_pixels) > ink_tolerance:
        return False

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold:
        return False

    # Tahap 2: Foreground ratio thresholding (minimal 50% luas penampang citra)
    foreground_ratio = np.sum(clean_mask) / total_pixels
    return foreground_ratio >= min_tissue_ratio


def run_wsi_extraction_and_filtration():
    """
    Mengunduh 50 WSI TCGA (.dcm) melalui IDC dan mengekstraksi patch berukuran 224x224 piksel
    pada magnifikasi 20x (0,5 mikrometer/piksel) langsung ke dalam 50 folder WSI (.png).
    """
    os.makedirs(RAW_WSI_DIR, exist_ok=True)
    os.makedirs(LOCAL_PATCH_DIR, exist_ok=True)

    client = index.IDCClient()
    df_meta = pd.read_csv(METADATA_CSV_PATH)
    records = []

    for idx, row in df_meta.iterrows():
        patient_id = str(row["PatientID"])
        dataset_name = str(row["collection_id"])
        series_uid = str(row["SeriesInstanceUID"])

        # Direktori tujuan untuk masing-masing WSI (berisi berkas .png)
        wsi_output_folder = os.path.join(LOCAL_PATCH_DIR, patient_id)
        os.makedirs(wsi_output_folder, exist_ok=True)

        # Unduh berkas WSI (.dcm) dari repositori Imaging Data Commons (IDC)
        wsi_temp_dir = os.path.join(RAW_WSI_DIR, patient_id)
        os.makedirs(wsi_temp_dir, exist_ok=True)
        client.download_dicom_series(seriesInstanceUID=[series_uid], downloadDir=wsi_temp_dir)

        dcm_files = glob.glob(os.path.join(wsi_temp_dir, "**", "*.dcm"), recursive=True)
        if not dcm_files:
            continue

        # Grid-based patching berukuran 224x224 piksel pada magnifikasi 20x (0,5 mpp)
        reader = WSIReader.open(os.path.dirname(dcm_files[0]))
        try:
            extractor = get_patch_extractor(
                input_img=reader, method_name="slidingwindow",
                patch_size=(224, 224), stride=(224, 224), resolution=0.5, units="mpp"
            )
        except Exception:
            extractor = get_patch_extractor(
                input_img=reader, method_name="slidingwindow",
                patch_size=(224, 224), stride=(224, 224), resolution=0, units="level"
            )

        before_filter = 0
        after_filter = 0

        for patch_img in extractor:
            if isinstance(patch_img, np.ndarray):
                before_filter += 1
                if filter_patch_quality(patch_img, min_tissue_ratio=0.50):
                    patch_filename = f"{patient_id}_patch_{after_filter:05d}.png"
                    patch_save_path = os.path.join(wsi_output_folder, patch_filename)
                    Image.fromarray(patch_img).save(patch_save_path)
                    after_filter += 1

        pass_rate = (after_filter / before_filter * 100.0) if before_filter > 0 else 0.0
        records.append({
            "PatientID": patient_id,
            "Dataset": dataset_name,
            "SeriesDate": row["SeriesDate"],
            "Manufacturer": row["Manufacturer"],
            "Before Filter": f"{before_filter:,}",
            "After Filter": f"{after_filter:,}",
            "Total Passed": f"{pass_rate:.2f}%"
        })
        print(f"[{idx+1:02d}/50] {patient_id} ({dataset_name}) | Before: {before_filter:,} | After: {after_filter:,} ({pass_rate:.2f}%)")

    df_result = pd.DataFrame(records)
    df_result.to_csv(METADATA_CSV_PATH, index=False)
    print("\nProses ekstraksi dan filtrasi kualitas 50 WSI TCGA selesai.")


if __name__ == "__main__":
    run_wsi_extraction_and_filtration()
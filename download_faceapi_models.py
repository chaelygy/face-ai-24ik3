################################################################################
# download_faceapi_models.py
# ============================================================================
# Script untuk mendownload model face-api.js yang dibutuhkan.
# Jalankan di Colab atau lokal sebelum deploy website.
#
# Model yang dibutuhkan:
# 1. ssd_mobilenetv1 - Deteksi wajah
# 2. face_landmark_68 - Landmark (titik-titik wajah)
# 3. face_recognition - Pengenalan wajah (embedding 128-D)
################################################################################

import os
import sys
import urllib.request

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Base URL untuk model face-api.js
BASE_URL = "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"

# Model yang dibutuhkan
MODELS = {
    "ssd_mobilenetv1": [
        "ssd_mobilenetv1_model-weights_manifest.json",
        "ssd_mobilenetv1_model-shard1",
        "ssd_mobilenetv1_model-shard2",
    ],
    "face_landmark_68": [
        "face_landmark_68_model-weights_manifest.json",
        "face_landmark_68_model-shard1",
    ],
    "face_recognition": [
        "face_recognition_model-weights_manifest.json",
        "face_recognition_model-shard1",
        "face_recognition_model-shard2",
    ],
}

def download_models(output_dir="website/models"):
    """Download semua model face-api.js ke folder lokal."""
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = sum(len(files) for files in MODELS.values())
    downloaded = 0
    
    for model_name, files in MODELS.items():
        print(f"\n[DOWNLOAD] {model_name}...")
        for filename in files:
            url = f"{BASE_URL}/{filename}"
            output_path = os.path.join(output_dir, filename)
            
            if os.path.exists(output_path):
                print(f"  [OK] {filename} (already exists)")
                downloaded += 1
                continue
            
            try:
                print(f"  [GET] {filename}...", end=" ", flush=True)
                urllib.request.urlretrieve(url, output_path)
                size = os.path.getsize(output_path)
                print(f"({size/1024:.0f} KB)")
                downloaded += 1
            except Exception as e:
                print(f"[ERROR] {e}")
    
    print(f"\n[DONE] Downloaded {downloaded}/{total_files} files to {output_dir}/")
    return output_dir

if __name__ == "__main__":
    download_models()

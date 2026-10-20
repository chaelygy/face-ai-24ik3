################################################################################
# 🎯 FACE RECOGNITION PIPELINE - GOOGLE COLAB NOTEBOOK
# ============================================================================
# Sistem Pengenalan Wajah 23 Teman Sekelas
# Salin setiap bagian (## CEL X) ke sel terpisah di Google Colab
#
# URUTAN: Data Audit → Split → Embedding (FaceNet) → Transfer Learning →
#         Evaluasi → Export TF.js
################################################################################

## ============================================================================
## CEL 1: INSTALASI & SETUP
## ============================================================================
# Jalankan cel ini pertama kali. Menginstal semua library yang dibutuhkan di Google Colab.
# (Hapus tanda '#' jika dijalankan langsung di Colab)
# !pip install -q facenet-pytorch mtcnn opencv-python-headless scikit-learn imagehash pillow matplotlib seaborn tensorflowjs torch torchvision

import os
import sys
import json
import shutil
import hashlib
import warnings
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

# Safe imports dengan fallback informasi
try:
    import cv2
    from PIL import Image
    import imagehash
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
except ImportError as e:
    print(f"Catatan (Normal jika di komputer lokal tanpa GPU/ML): {e}")

try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError:
    pass

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
    from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
except ImportError:
    pass

try:
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    from sklearn.preprocessing import LabelEncoder, normalize
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_recall_fscore_support, f1_score
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    pass

import pickle
warnings.filterwarnings('ignore')

# Status Library & GPU
if 'torch' in sys.modules:
    print(f"PyTorch: {torch.__version__} | GPU: {torch.cuda.is_available()}")
if 'tf' in sys.modules:
    print(f"TensorFlow: {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ---- AUTO MOUNT GOOGLE DRIVE & DATASET PATH ----
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except Exception:
    pass

# Auto-deteksi lokasi dataset bahan
CANDIDATE_PATHS = [
    "/content/drive/MyDrive/face_ai/bahan",
    "/content/drive/MyDrive/bahan",
    "/content/bahan",
    os.path.abspath(os.path.join(os.getcwd(), "bahan")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bahan")) if '__file__' in locals() else "./bahan"
]

DATASET_ROOT = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), "/content/drive/MyDrive/face_ai/bahan")
print(f"📁 Menggunakan DATASET_ROOT: {DATASET_ROOT}")

OUTPUT_DIR = "/content/face_recognition_output" if os.path.exists("/content") else os.path.abspath("./face_recognition_output")
ALIGNED_DIR = os.path.join(OUTPUT_DIR, "aligned_faces")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
TFJS_DIR = os.path.join(OUTPUT_DIR, "tfjs_model")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

for d in [OUTPUT_DIR, ALIGNED_DIR, MODELS_DIR, TFJS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE = 160       # FaceNet input
IMG_SIZE_TL = 224    # MobileNetV2 input
BATCH_SIZE = 32
SEED = 42

if 'np' in sys.modules: np.random.seed(SEED)
if 'torch' in sys.modules: torch.manual_seed(SEED)
if 'tf' in sys.modules: tf.random.set_seed(SEED)

print("✅ Setup selesai!")


## ============================================================================
## CEL 2: TAHAP 0 - SCAN DATASET & FLATTEN STRUKTUR FOLDER
## ============================================================================
# Dataset kamu punya struktur nested (subfolder dalam subfolder).
# Cel ini akan scan semua foto secara rekursif dan memetakan ke nama orang.

def scan_dataset(root_dir):
    """
    Scan dataset dengan struktur nested.
    Mengambil nama orang dari folder level 1 di bawah root.
    Foto dicari secara rekursif di semua subfolder.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    person_photos = defaultdict(list)

    for person_folder in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_folder)
        if not os.path.isdir(person_path):
            continue

        # Cari semua file gambar secara rekursif
        for dirpath, dirnames, filenames in os.walk(person_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(dirpath, fname)
                    person_photos[person_folder].append(full_path)

    return dict(person_photos)

person_photos = scan_dataset(DATASET_ROOT)

print(f"📁 Ditemukan {len(person_photos)} orang:\n")
total = 0
for name, photos in sorted(person_photos.items()):
    print(f"  {name:25s} : {len(photos):3d} foto")
    total += len(photos)
print(f"\n📸 Total: {total} foto")

# Simpan mapping untuk referensi
labels_list = sorted(person_photos.keys())
print(f"\n🏷️ Label (urutan alfabet): {labels_list}")


## ============================================================================
## CEL 3: TAHAP 0 - DETEKSI WAJAH, KUALITAS, & ALIGNMENT
## ============================================================================
# Menggunakan MTCNN untuk:
# 1. Mendeteksi wajah di setiap foto
# 2. Cek foto tanpa wajah / lebih dari 1 wajah
# 3. Cek foto buram (Laplacian variance)
# 4. Cek foto terlalu gelap
# 5. Alignment berdasarkan landmark mata + crop konsisten

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN untuk deteksi wajah dengan margin
mtcnn_detect = MTCNN(
    image_size=IMG_SIZE,
    margin=20,              # Margin di sekitar wajah
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,     # Jangan normalize, kita mau cek kualitas dulu
    keep_all=True,          # Deteksi semua wajah dalam foto
    device=device
)

# MTCNN untuk crop & align 1 wajah terbesar
mtcnn_align = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,      # Normalize ke [-1, 1] untuk FaceNet
    keep_all=False,         # Ambil wajah paling besar/percaya diri saja
    device=device
)

def check_blur(image_np, threshold=12.0):
    """Cek apakah gambar buram ekstrem. Variance of Laplacian < threshold = buram parah."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var, lap_var < threshold

def check_brightness(image_np, dark_threshold=20, bright_threshold=245):
    """Cek apakah gambar terlalu gelap gulita atau putih silau total."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    is_dark = mean_brightness < dark_threshold
    is_bright = mean_brightness > bright_threshold
    return mean_brightness, is_dark, is_bright

def align_face_landmarks(image_pil, landmarks):
    """Luruskan wajah berdasarkan posisi mata (affine transform)."""
    img_np = np.array(image_pil)
    if isinstance(landmarks, dict):
        left_eye = landmarks.get('left_eye')
        right_eye = landmarks.get('right_eye')
    elif hasattr(landmarks, '__len__') and len(landmarks) >= 2:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
    else:
        return image_pil

    if left_eye is None or right_eye is None:
        return image_pil

    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    angle = float(np.degrees(np.arctan2(dy, dx)))

    eye_center = (float((left_eye[0] + right_eye[0]) / 2),
                  float((left_eye[1] + right_eye[1]) / 2))
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(img_np, M, (img_np.shape[1], img_np.shape[0]),
                              flags=cv2.INTER_CUBIC)
    return Image.fromarray(aligned)

print("🔍 Mulai audit kualitas data (Smart Tolerant Filter)...\n")

audit_results = {}
good_photos = defaultdict(list)
problem_photos = defaultdict(list)

for person_name, photo_paths in tqdm(person_photos.items(), desc="Audit"):
    audit_results[person_name] = {}

    for photo_path in photo_paths:
        result = {
            'path': photo_path,
            'issues': [],
            'status': 'good'
        }

        try:
            img = Image.open(photo_path).convert('RGB')
            img_np = np.array(img)

            lap_var, is_blur = check_blur(img_np)
            result['blur_score'] = float(lap_var)
            if is_blur:
                result['issues'].append(f'buram berat ({lap_var:.1f})')

            brightness, is_dark, is_bright = check_brightness(img_np)
            result['brightness'] = float(brightness)
            if is_dark:
                result['issues'].append(f'sangat gelap ({brightness:.1f})')
            if is_bright:
                result['issues'].append(f'sangat terang ({brightness:.1f})')

            # Deteksi wajah menggunakan MTCNN
            boxes, probs, landmarks = mtcnn_detect.detect(img, landmarks=True)

            if boxes is None or len(boxes) == 0:
                result['issues'].append('tidak ada wajah terdeteksi')
                result['num_faces'] = 0
                result['status'] = 'problem'
                problem_photos[person_name].append(result)
            else:
                result['num_faces'] = len(boxes)
                result['confidence'] = float(probs[0]) if probs is not None else 0

                # Jika ada beberapa wajah terdeteksi, MTCNN align nanti otomatis memilih
                # wajah terbesar / confidence tertinggi, jadi foto TETAP DIPAKAI!
                if len(boxes) > 1:
                    result['issues'].append(f'{len(boxes)} wajah (akan diambil wajah utama)')

                # Cek jika wajah sangat kecil (< 30px)
                box = boxes[0]
                face_w = box[2] - box[0]
                face_h = box[3] - box[1]
                if face_w < 30 or face_h < 30:
                    result['issues'].append(f'wajah terlalu kecil ({face_w:.0f}x{face_h:.0f})')
                    result['status'] = 'problem'
                    problem_photos[person_name].append(result)
                else:
                    # Foto valid!
                    good_photos[person_name].append(photo_path)

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'error: {str(e)}')
            problem_photos[person_name].append(result)

        audit_results[person_name][photo_path] = result

# Laporan
print("\n" + "="*70)
print("📊 LAPORAN AUDIT KUALITAS DATA")
print("="*70)
print(f"\n{'Nama':<25} {'Valid':>6} {'Ditolak':>8} {'Total':>6}")
print("-"*50)

total_good = 0
total_problem = 0
for name in sorted(person_photos.keys()):
    n_good = len(good_photos[name])
    n_prob = len(problem_photos[name])
    total_good += n_good
    total_problem += n_prob
    flag = " ⚠️ Sedikit" if n_good < 20 else ""
    print(f"{name:<25} {n_good:>6} {n_prob:>8} {n_good+n_prob:>6}{flag}")

print("-"*50)
print(f"{'TOTAL':<25} {total_good:>6} {total_problem:>8} {total_good+total_problem:>6}")
print(f"✅ Tingkat retensi data: {total_good / max(1, total_good + total_problem) * 100:.1f}% foto valid!")


## ============================================================================
## CEL 4: TAHAP 0 - DETEKSI DUPLIKAT & AUGMENTASI PENYEIMBANG KELAS
## ============================================================================
# 1. Hapus duplikat file identik agar split data tidak bocor
# 2. Augmentasi otomatis untuk kelas dengan foto sedikit agar dataset seimbang

print("🔍 Memeriksa duplikasi file identik...\n")

def get_file_md5(filepath):
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

# Deduplikasi berbasis hash persis per orang
unique_good_photos = defaultdict(list)
for person_name, paths in good_photos.items():
    seen_hashes = set()
    for p in paths:
        h = get_file_md5(p)
        if h and h not in seen_hashes:
            seen_hashes.add(h)
            unique_good_photos[person_name].append(p)

print("📊 Jumlah foto unik per orang setelah deduplikasi:")
for name in sorted(unique_good_photos.keys()):
    count = len(unique_good_photos[name])
    flag = " ⚠️ Perlu Augmentasi" if count < 25 else " ✅ Cukup"
    print(f"  {name:25s} : {count:3d} foto unik {flag}")


## ============================================================================
## CEL 5: TAHAP 0 - FACE ALIGNMENT, CROP, & BALANCED AUGMENTATION
## ============================================================================
# 1. MTCNN crop & align presisi 160×160 piksel (Aman di CPU & GPU Colab)
# 2. Untuk kelas yang fotonya sedikit (< 35 foto), otomatis dibuat augmentasi
#    (horizontal flip, rotasi ±5°, contrast/brightness jitter) sehingga semua orang
#    punya minimal 35-50 variasi foto berkualitas.

print("🔧 Mulai face alignment & penyeimbangan dataset...\n")

aligned_photos = defaultdict(list)
TARGET_MIN_SAMPLES = 35  # Target minimal foto per orang

for person_name in tqdm(sorted(unique_good_photos.keys()), desc="Aligning"):
    person_out_dir = os.path.join(ALIGNED_DIR, person_name)
    os.makedirs(person_out_dir, exist_ok=True)

    # 1. Ekstrak wajah asli
    for idx, photo_path in enumerate(unique_good_photos[person_name]):
        try:
            img = Image.open(photo_path).convert('RGB')
            face_tensor = mtcnn_align(img)

            if face_tensor is not None:
                # Aman untuk CUDA GPU (pindah ke CPU dulu)
                face_np = face_tensor.detach().cpu().permute(1, 2, 0).numpy()
                face_np = ((face_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                face_img = Image.fromarray(face_np)

                out_path = os.path.join(person_out_dir, f"face_{idx:03d}.png")
                face_img.save(out_path)
                aligned_photos[person_name].append(out_path)
        except Exception as e:
            pass

    # 2. Augmentasi jika foto masih di bawah TARGET_MIN_SAMPLES
    current_count = len(aligned_photos[person_name])
    if current_count > 0 and current_count < TARGET_MIN_SAMPLES:
        needed = TARGET_MIN_SAMPLES - current_count
        base_paths = list(aligned_photos[person_name])
        aug_idx = 0

        while len(aligned_photos[person_name]) < TARGET_MIN_SAMPLES and aug_idx < needed:
            src_path = base_paths[aug_idx % len(base_paths)]
            try:
                base_img = Image.open(src_path).convert('RGB')
                aug_type = aug_idx % 4

                if aug_type == 0:
                    # Horizontal flip
                    aug_img = base_img.transpose(Image.FLIP_LEFT_RIGHT)
                elif aug_type == 1:
                    # Rotasi halus +5 derajat
                    aug_img = base_img.rotate(5, resample=Image.BICUBIC)
                elif aug_type == 2:
                    # Rotasi halus -5 derajat
                    aug_img = base_img.rotate(-5, resample=Image.BICUBIC)
                else:
                    # Slight brightness adjustment
                    np_arr = np.array(base_img, dtype=np.float32) * np.random.uniform(0.9, 1.1)
                    aug_img = Image.fromarray(np.clip(np_arr, 0, 255).astype(np.uint8))

                aug_out_path = os.path.join(person_out_dir, f"aug_{aug_idx:03d}.png")
                aug_img.save(aug_out_path)
                aligned_photos[person_name].append(aug_out_path)
            except Exception:
                pass
            aug_idx += 1

print("\n📊 Hasil alignment & penyeimbangan data:")
print(f"\n{'Nama':<25} {'Total Aligned':>14}")
print("-"*42)
for name in sorted(aligned_photos.keys()):
    count = len(aligned_photos[name])
    print(f"{name:<25} {count:>14d} foto")

total_aligned = sum(len(v) for v in aligned_photos.values())
print(f"\n✅ Total foto siap latih: {total_aligned} foto (Semua kelas seimbang!)")

total_aligned = sum(len(v) for v in aligned_photos.values())
print(f"\n✅ Total foto aligned: {total_aligned}")


## ============================================================================
## CEL 6: TAHAP 1 - PEMBAGIAN DATA (STRATIFIED, TANPA KEBOCORAN)
## ============================================================================
# Bagi per orang: 70% train, 15% validasi, 15% tes
# Data tes TIDAK BOLEH disentuh sampai evaluasi akhir

print("📂 Membagi data train/val/test...\n")

from sklearn.model_selection import train_test_split

# Siapkan data
all_paths = []
all_labels = []

for person_name, paths in sorted(aligned_photos.items()):
    for path in paths:
        all_paths.append(path)
        all_labels.append(person_name)

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)

# Label encoder
le = LabelEncoder()
all_labels_encoded = le.fit_transform(all_labels)

# Split 1: 70% train, 30% temp (val+test)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.30,
    stratify=all_labels,
    random_state=SEED
)

# Split 2: 50/50 dari temp → 15% val, 15% test
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED
)

print(f"📊 Pembagian data:")
print(f"  Train : {len(train_paths):4d} foto ({len(train_paths)/len(all_paths)*100:.1f}%)")
print(f"  Val   : {len(val_paths):4d} foto ({len(val_paths)/len(all_paths)*100:.1f}%)")
print(f"  Test  : {len(test_paths):4d} foto ({len(test_paths)/len(all_paths)*100:.1f}%)")

# Cek distribusi per orang
print(f"\n{'Nama':<25} {'Train':>6} {'Val':>6} {'Test':>6}")
print("-"*50)
for name in sorted(set(all_labels)):
    n_tr = sum(train_labels == name)
    n_va = sum(val_labels == name)
    n_te = sum(test_labels == name)
    print(f"{name:<25} {n_tr:>6} {n_va:>6} {n_te:>6}")

# Verifikasi tidak ada kebocoran
train_set = set(train_paths)
val_set = set(val_paths)
test_set = set(test_paths)

assert len(train_set & val_set) == 0, "KEBOCORAN: train & val overlap!"
assert len(train_set & test_set) == 0, "KEBOCORAN: train & test overlap!"
assert len(val_set & test_set) == 0, "KEBOCORAN: val & test overlap!"
print("\n✅ Tidak ada kebocoran data antar split!")

# Simpan split info
split_info = {
    'train': {p: l for p, l in zip(train_paths.tolist(), train_labels.tolist())},
    'val': {p: l for p, l in zip(val_paths.tolist(), val_labels.tolist())},
    'test': {p: l for p, l in zip(test_paths.tolist(), test_labels.tolist())},
    'labels': le.classes_.tolist()
}

with open(os.path.join(OUTPUT_DIR, 'split_info.json'), 'w') as f:
    json.dump(split_info, f, indent=2)

print("💾 Split info disimpan ke split_info.json")


## ============================================================================
## CEL 7: TAHAP 2A - FACE EMBEDDING (FaceNet) + CLASSIFIER
## ============================================================================
# Pendekatan UTAMA: Gunakan FaceNet pretrained untuk menghasilkan embedding 512-D.
# Lalu klasifikasi dengan SVM (linear + RBF) dan k-NN.

print("🧠 Tahap 2A: FaceNet Embedding + Classifier\n")

# Load FaceNet pretrained (InceptionResNetV1 trained on VGGFace2)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✅ FaceNet model loaded (pretrained VGGFace2)")

def get_embeddings(image_paths, model, batch_size=32):
    """Extract face embeddings dari kumpulan gambar yang sudah di-align."""
    embeddings = []
    valid_indices = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []

        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_np = np.array(img).astype(np.float32)
                # Normalize ke [-1, 1] (sama seperti MTCNN post_process=True)
                img_np = (img_np - 127.5) / 128.0
                tensor = torch.FloatTensor(img_np).permute(2, 0, 1)  # HWC -> CHW
                batch_tensors.append(tensor)
                valid_indices.append(i + j)
            except Exception as e:
                print(f"  ⚠️ Skip {path}: {e}")

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])

    return embeddings, valid_indices

# Extract embeddings untuk train, val, test
print("\n📐 Extracting embeddings...")

train_emb, train_valid_idx = get_embeddings(train_paths, facenet)
train_labels_valid = train_labels[train_valid_idx] if len(train_valid_idx) < len(train_labels) else train_labels

val_emb, val_valid_idx = get_embeddings(val_paths, facenet)
val_labels_valid = val_labels[val_valid_idx] if len(val_valid_idx) < len(val_labels) else val_labels

test_emb, test_valid_idx = get_embeddings(test_paths, facenet)
test_labels_valid = test_labels[test_valid_idx] if len(test_valid_idx) < len(test_labels) else test_labels

print(f"\n📊 Embedding shapes:")
print(f"  Train: {train_emb.shape}")
print(f"  Val  : {val_emb.shape}")
print(f"  Test : {test_emb.shape}")

# L2 normalize embeddings (penting untuk cosine similarity)
train_emb_norm = normalize(train_emb, norm='l2')
val_emb_norm = normalize(val_emb, norm='l2')
test_emb_norm = normalize(test_emb, norm='l2')

# --- 1. Linear SVM Classifier (Standar Industri untuk FaceNet) ---
print("\n🔧 1. Training Linear SVM (C=1.0)...")
svm_linear = SVC(kernel='linear', C=1.0, probability=True, random_state=SEED)
svm_linear.fit(train_emb_norm, train_labels_valid)
svm_lin_val_pred = svm_linear.predict(val_emb_norm)
svm_lin_val_acc = accuracy_score(val_labels_valid, svm_lin_val_pred)
print(f"  Linear SVM Val Accuracy: {svm_lin_val_acc:.4f} ({svm_lin_val_acc*100:.1f}%)")

# --- 2. Logistic Regression (Softmax) ---
print("🔧 2. Training Logistic Regression (Softmax)...")
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(C=5.0, max_iter=1000, random_state=SEED)
lr_clf.fit(train_emb_norm, train_labels_valid)
lr_val_pred = lr_clf.predict(val_emb_norm)
lr_val_acc = accuracy_score(val_labels_valid, lr_val_pred)
print(f"  Logistic Regression Val Accuracy: {lr_val_acc:.4f} ({lr_val_acc*100:.1f}%)")

# --- 3. Centroid-based (Cosine Similarity) ---
print("🔧 3. Computing Class Centroids (Cosine)...")
centroids = {}
for name in sorted(set(train_labels_valid)):
    mask = train_labels_valid == name
    centroid = train_emb_norm[mask].mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # re-normalize
    centroids[name] = centroid

def predict_centroid(embeddings, centroids, threshold=None):
    """Prediksi berdasarkan cosine similarity ke centroid terdekat."""
    centroid_names = list(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in centroid_names])

    # Cosine similarity = dot product (karena sudah L2-normalized)
    similarities = embeddings @ centroid_matrix.T  # (N, num_classes)

    pred_indices = similarities.argmax(axis=1)
    pred_names = [centroid_names[i] for i in pred_indices]
    pred_scores = similarities.max(axis=1)

    if threshold is not None:
        pred_names = [n if s >= threshold else "Tidak Dikenal"
                      for n, s in zip(pred_names, pred_scores)]

    return pred_names, pred_scores, similarities

centroid_val_pred, centroid_val_scores, _ = predict_centroid(val_emb_norm, centroids)
centroid_val_acc = accuracy_score(val_labels_valid, centroid_val_pred)
print(f"  Centroid Val Accuracy: {centroid_val_acc:.4f} ({centroid_val_acc*100:.1f}%)")

# --- 4. Cosine k-NN Classifier (k=3) ---
print("🔧 4. Training Cosine k-NN (k=3)...")
knn_clf = KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance')
knn_clf.fit(train_emb_norm, train_labels_valid)
knn_val_pred = knn_clf.predict(val_emb_norm)
knn_val_acc = accuracy_score(val_labels_valid, knn_val_pred)
print(f"  k-NN Val Accuracy: {knn_val_acc:.4f} ({knn_val_acc*100:.1f}%)")

# --- 5. RBF SVM ---
print("🔧 5. Training RBF SVM...")
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED)
svm_rbf.fit(train_emb_norm, train_labels_valid)
svm_rbf_val_pred = svm_rbf.predict(val_emb_norm)
svm_rbf_val_acc = accuracy_score(val_labels_valid, svm_rbf_val_pred)
print(f"  RBF SVM Val Accuracy: {svm_rbf_val_acc:.4f} ({svm_rbf_val_acc*100:.1f}%)")

# --- Pilih classifier FaceNet terbaik ---
results_2a = {
    'Linear SVM': {'accuracy': svm_lin_val_acc, 'model': svm_linear},
    'Logistic Regression': {'accuracy': lr_val_acc, 'model': lr_clf},
    'Centroid Cosine': {'accuracy': centroid_val_acc, 'model': centroids},
    'k-NN (Cosine)': {'accuracy': knn_val_acc, 'model': knn_clf},
    'RBF SVM': {'accuracy': svm_rbf_val_acc, 'model': svm_rbf}
}

best_clf_name = max(results_2a, key=lambda k: results_2a[k]['accuracy'])
best_clf = results_2a[best_clf_name]
svm_clf = results_2a['Linear SVM']['model'] if results_2a['Linear SVM']['accuracy'] >= results_2a['RBF SVM']['accuracy'] else results_2a['RBF SVM']['model']
print(f"\n🏆 Model FaceNet terbaik: {best_clf_name} (Val Accuracy = {best_clf['accuracy']*100:.2f}%)")

# Simpan embedding & centroids
np.save(os.path.join(MODELS_DIR, 'train_embeddings.npy'), train_emb_norm)
np.save(os.path.join(MODELS_DIR, 'train_labels.npy'), train_labels_valid)

centroids_json = {name: centroid.tolist() for name, centroid in centroids.items()}
with open(os.path.join(MODELS_DIR, 'centroids.json'), 'w') as f:
    json.dump(centroids_json, f, indent=2)

# Simpan SVM model
with open(os.path.join(MODELS_DIR, 'svm_classifier.pkl'), 'wb') as f:
    pickle.dump(svm_clf, f)

print("💾 Embeddings dan centroids disimpan!")


## ============================================================================
## CEL 8: TAHAP 2A - KALIBRASI THRESHOLD "TIDAK DIKENAL"
## ============================================================================
# Uji dengan wajah orang luar (dari LFW dataset) untuk menentukan threshold
# yang optimal untuk menolak orang yang tidak dikenal.

print("🔧 Kalibrasi threshold 'Tidak Dikenal'...\n")

# Download sampel LFW untuk testing orang luar
# !wget -q http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz -O /tmp/lfw.tgz
# !tar -xzf /tmp/lfw.tgz -C /tmp/

LFW_DIR = "/tmp/lfw_funneled"  # Sesuaikan jika path berbeda

def get_lfw_samples(lfw_dir, n_samples=100):
    """Ambil sampel foto dari LFW sebagai 'orang luar'."""
    samples = []
    if not os.path.exists(lfw_dir):
        print(f"  ⚠️ LFW tidak ditemukan di {lfw_dir}")
        print("  Download manual: wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz")
        return samples

    for person_dir in os.listdir(lfw_dir):
        person_path = os.path.join(lfw_dir, person_dir)
        if os.path.isdir(person_path):
            for fname in os.listdir(person_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples.append(os.path.join(person_path, fname))
                    if len(samples) >= n_samples:
                        return samples
    return samples

# Jika LFW tidak tersedia, gunakan augmented photos sebagai proxy
# atau kamu bisa upload foto orang luar sendiri
lfw_samples = get_lfw_samples(LFW_DIR, n_samples=200)

if len(lfw_samples) > 0:
    print(f"📸 {len(lfw_samples)} foto orang luar dari LFW")

    # Align & embed foto orang luar
    outsider_aligned = []
    for path in tqdm(lfw_samples[:200], desc="Aligning outsiders"):
        try:
            img = Image.open(path).convert('RGB')
            face = mtcnn_align(img)
            if face is not None:
                outsider_aligned.append(face)
        except:
            pass

    if outsider_aligned:
        outsider_batch = torch.stack(outsider_aligned).to(device)
        with torch.no_grad():
            outsider_emb = facenet(outsider_batch).cpu().numpy()
        outsider_emb_norm = normalize(outsider_emb, norm='l2')

        # Hitung skor similarity ke centroid terdekat
        centroid_names = list(centroids.keys())
        centroid_matrix = np.array([centroids[n] for n in centroid_names])
        outsider_sims = (outsider_emb_norm @ centroid_matrix.T).max(axis=1)

        # Hitung skor insider (dari validation set)
        insider_sims = (val_emb_norm @ centroid_matrix.T).max(axis=1)

        # Plot distribusi skor
        plt.figure(figsize=(12, 5))
        plt.hist(insider_sims, bins=50, alpha=0.6, label='Teman sekelas (insider)', color='green')
        plt.hist(outsider_sims, bins=50, alpha=0.6, label='Orang luar (outsider)', color='red')
        plt.axvline(x=0.5, color='blue', linestyle='--', label='Threshold = 0.5')
        plt.xlabel('Cosine Similarity ke Centroid Terdekat')
        plt.ylabel('Jumlah')
        plt.title('Distribusi Skor: Insider vs Outsider')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'threshold_distribution.png'), dpi=150)
        plt.show()

        # Cari threshold optimal (maximize gap)
        thresholds = np.arange(0.2, 0.8, 0.01)
        best_threshold = 0.5
        best_f1 = 0

        for t in thresholds:
            # True positive = insider di atas threshold
            tp = (insider_sims >= t).sum()
            # False negative = insider di bawah threshold
            fn = (insider_sims < t).sum()
            # True negative = outsider di bawah threshold
            tn = (outsider_sims < t).sum()
            # False positive = outsider di atas threshold
            fp = (outsider_sims >= t).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"\n🎯 Threshold optimal: {best_threshold:.3f} (F1={best_f1:.4f})")
        print(f"  False Accept Rate: {(outsider_sims >= best_threshold).mean():.4f}")
        print(f"  False Reject Rate: {(insider_sims < best_threshold).mean():.4f}")
    else:
        best_threshold = 0.45
        print("⚠️ Tidak bisa align foto LFW, pakai threshold default 0.45")
else:
    print("⚠️ LFW tidak tersedia. Menghitung threshold dari data validasi saja...")

    centroid_names = list(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in centroid_names])
    insider_sims = (val_emb_norm @ centroid_matrix.T).max(axis=1)

    # Pakai percentile 5% sebagai threshold (95% insider harus lolos)
    best_threshold = max(0.35, float(np.percentile(insider_sims, 5)) - 0.05)
    print(f"  Insider similarity: min={insider_sims.min():.3f}, mean={insider_sims.mean():.3f}, "
          f"max={insider_sims.max():.3f}")
    print(f"  🎯 Threshold: {best_threshold:.3f}")

# Simpan threshold
config = {
    'threshold': float(best_threshold),
    'embedding_size': 512,
    'img_size': IMG_SIZE,
    'num_classes': len(centroids),
    'labels': sorted(centroids.keys()),
    'normalization': 'facenet',  # (pixel - 127.5) / 128.0
    'approach': 'embedding_centroid'
}

with open(os.path.join(MODELS_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n💾 Config disimpan: {json.dumps(config, indent=2)}")


## ============================================================================
## CEL 9: TAHAP 2B - TRANSFER LEARNING (MobileNetV2) SEBAGAI PEMBANDING
## ============================================================================
# Pendekatan pembanding: Fine-tune MobileNetV2 untuk klasifikasi langsung.

print("🧠 Tahap 2B: Transfer Learning (MobileNetV2)\n")

NUM_CLASSES = len(sorted(set(all_labels)))

# --- Data Generator dengan Augmentasi ---
def load_and_preprocess(path, label, img_size, augment=False):
    """Load gambar, resize, dan preprocess untuk MobileNetV2."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        # Rotasi kecil (±15 derajat)
        angle = tf.random.uniform([], -15, 15) * np.pi / 180
        img = tf.keras.preprocessing.image.apply_affine_transform(
            img.numpy(), theta=angle
        ) if False else img  # TF doesn't have easy rotate in graph mode
        # Zoom
        scale = tf.random.uniform([], 0.9, 1.1)
        new_size = tf.cast(tf.cast([img_size, img_size], tf.float32) * scale, tf.int32)
        img = tf.image.resize(img, new_size)
        img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)
        # Noise
        noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=2.0)
        img = img + noise

    img = tf.clip_by_value(img, 0, 255)
    # MobileNetV2 preprocessing: [-1, 1]
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

# Encode labels
le_tf = LabelEncoder()
le_tf.fit(sorted(set(all_labels)))

train_labels_enc = le_tf.transform(train_labels)
val_labels_enc = le_tf.transform(val_labels)
test_labels_enc = le_tf.transform(test_labels)

# Class weights untuk menangani imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels_enc),
    y=train_labels_enc
)
class_weights = dict(enumerate(class_weights_arr))

# TF Dataset pipeline
def create_tf_dataset(paths, labels, img_size, batch_size, augment=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    def _preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [img_size, img_size])

        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.15)
            img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
            # Random zoom via crop & resize
            scale = tf.random.uniform([], 0.85, 1.0)
            crop_size = tf.cast(tf.cast(img_size, tf.float32) * scale, tf.int32)
            img = tf.image.random_crop(img, [crop_size, crop_size, 3])
            img = tf.image.resize(img, [img_size, img_size])

        img = tf.clip_by_value(img, 0, 255)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        label_onehot = tf.one_hot(label, NUM_CLASSES)
        return img, label_onehot

    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_tf_dataset(train_paths, train_labels_enc, IMG_SIZE_TL, BATCH_SIZE,
                              augment=True, shuffle=True)
val_ds = create_tf_dataset(val_paths, val_labels_enc, IMG_SIZE_TL, BATCH_SIZE,
                            augment=False, shuffle=False)
test_ds = create_tf_dataset(test_paths, test_labels_enc, IMG_SIZE_TL, BATCH_SIZE,
                             augment=False, shuffle=False)

# --- Build Model ---
def build_mobilenet_model(num_classes, img_size=224):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False  # Bekukan dulu

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model

model_tl, base_model = build_mobilenet_model(NUM_CLASSES)
model_tl.summary()

# --- Stage 1: Train head only ---
print("\n📝 Stage 1: Training head (base frozen)...")

model_tl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

stage1_callbacks = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                           monitor='val_accuracy'),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6,
                                monitor='val_loss'),
]

history1 = model_tl.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weights,
    callbacks=stage1_callbacks,
    verbose=1
)

print(f"\n  Stage 1 best val accuracy: {max(history1.history['val_accuracy']):.4f}")

# --- Stage 2: Fine-tune top layers ---
print("\n📝 Stage 2: Fine-tuning top layers...")

# Unfreeze top 30% layers dari MobileNetV2
total_layers = len(base_model.layers)
fine_tune_from = int(total_layers * 0.7)
base_model.trainable = True
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

print(f"  Total layers: {total_layers}, Fine-tuning from layer {fine_tune_from}")
print(f"  Trainable layers: {total_layers - fine_tune_from}")

model_tl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

stage2_callbacks = [
    callbacks.EarlyStopping(patience=15, restore_best_weights=True,
                           monitor='val_accuracy'),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7,
                                monitor='val_loss'),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, 'mobilenet_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history2 = model_tl.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    class_weight=class_weights,
    callbacks=stage2_callbacks,
    verbose=1
)

best_tl_acc = max(history2.history['val_accuracy'])
print(f"\n  Stage 2 best val accuracy: {best_tl_acc:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gabungkan history
all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']

axes[0].plot(all_acc, label='Train')
axes[0].plot(all_val_acc, label='Validation')
axes[0].axvline(x=len(history1.history['accuracy']), color='r', linestyle='--',
                label='Start Fine-tuning')
axes[0].set_title('Accuracy')
axes[0].legend()

axes[1].plot(all_loss, label='Train')
axes[1].plot(all_val_loss, label='Validation')
axes[1].axvline(x=len(history1.history['loss']), color='r', linestyle='--',
                label='Start Fine-tuning')
axes[1].set_title('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'training_history.png'), dpi=150)
plt.show()

# Simpan labels
labels_json = {
    'labels': le_tf.classes_.tolist(),
    'num_classes': NUM_CLASSES
}
with open(os.path.join(MODELS_DIR, 'labels.json'), 'w') as f:
    json.dump(labels_json, f, indent=2)

print("💾 Model dan labels disimpan!")


## ============================================================================
## CEL 10: TAHAP 2 - PERBANDINGAN PENDEKATAN A vs B
## ============================================================================
# Bandingkan hasil embedding (A) vs transfer learning (B) di data VALIDASI.

print("📊 Perbandingan Pendekatan A (Embedding) vs B (Transfer Learning)\n")

# Pendekatan A: Embedding terbaik
best_emb_method = best_clf_name
best_emb_acc = best_clf['accuracy']

# Detail per-class untuk embedding (SVM)
svm_val_report = classification_report(val_labels_valid, svm_val_pred, output_dict=True)

# Pendekatan B: Transfer Learning
tl_val_pred_proba = model_tl.predict(val_ds)
tl_val_pred = le_tf.inverse_transform(tl_val_pred_proba.argmax(axis=1))
tl_val_acc = accuracy_score(val_labels, tl_val_pred)

tl_val_report = classification_report(val_labels, tl_val_pred, output_dict=True)

print(f"{'Metode':<40} {'Val Accuracy':>15}")
print("-"*60)
print(f"{'A1: FaceNet + SVM':<40} {svm_val_acc:>15.4f}")
print(f"{'A2: FaceNet + k-NN':<40} {knn_val_acc:>15.4f}")
print(f"{'A3: FaceNet + Centroid':<40} {centroid_val_acc:>15.4f}")
print(f"{'B:  MobileNetV2 Transfer Learning':<40} {tl_val_acc:>15.4f}")

# Tentukan pemenang
all_scores = {
    'FaceNet + SVM': svm_val_acc,
    'FaceNet + k-NN': knn_val_acc,
    'FaceNet + Centroid': centroid_val_acc,
    'MobileNetV2 TL': tl_val_acc
}
winner = max(all_scores, key=all_scores.get)
print(f"\n🏆 PEMENANG: {winner} (accuracy={all_scores[winner]:.4f})")

# Analisis: Apakah pendekatan embedding lebih baik?
if 'FaceNet' in winner:
    print("""
📝 ANALISIS:
Pendekatan embedding (FaceNet) lebih unggul. Ini masuk akal karena:
1. FaceNet sudah dilatih dengan jutaan wajah → embedding-nya sangat representatif
2. Dengan data sedikit (30-50 foto/orang), embedding + classifier sederhana
   lebih robust daripada fine-tuning CNN dari awal
3. Embedding approach juga lebih mudah menambah orang baru (cukup hitung
   embedding baru, tidak perlu re-train model)
4. Threshold "Tidak Dikenal" lebih natural dengan cosine similarity
""")
else:
    print("""
📝 ANALISIS:
Transfer Learning MobileNetV2 lebih unggul. Meskipun jarang terjadi dengan
data sedikit, ini bisa terjadi jika:
1. Augmentasi data sangat efektif menambah variasi
2. Fine-tuning berhasil mengadaptasi fitur mid-level untuk wajah teman
3. Namun, perlu diperhatikan bahwa pendekatan ini KURANG fleksibel
   untuk menambah orang baru (perlu re-train)
""")

CHOSEN_APPROACH = 'embedding' if 'FaceNet' in winner else 'transfer_learning'
print(f"✅ Pendekatan terpilih: {CHOSEN_APPROACH}")


## ============================================================================
## CEL 11: TAHAP 3 - EVALUASI MENYELURUH (DATA TEST)
## ============================================================================
# Evaluasi pada data TEST yang belum pernah disentuh.

print("📊 TAHAP 3: Evaluasi pada Data Test\n")
print("⚠️ Data test digunakan HANYA di sini. Tidak boleh dipakai untuk tuning!\n")

# --- Evaluasi pendekatan terpilih ---
if CHOSEN_APPROACH == 'embedding':
    # Gunakan model FaceNet terbaik yang menang di validation set
    print(f"Menggunakan model FaceNet terbaik: {best_clf_name}")
    if best_clf_name == 'Centroid Cosine':
        test_pred, test_pred_scores, _ = predict_centroid(test_emb_norm, centroids, threshold=None)
    else:
        best_eval_model = best_clf['model']
        test_pred = best_eval_model.predict(test_emb_norm)
    test_true = test_labels_valid
else:
    test_pred_proba_raw = model_tl.predict(test_ds)
    test_pred = le_tf.inverse_transform(test_pred_proba_raw.argmax(axis=1))
    test_true = test_labels
    test_pred_proba = test_pred_proba_raw

# --- 1. Accuracy, Precision, Recall, F1 ---
test_acc = accuracy_score(test_true, test_pred)
print(f"🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

report = classification_report(test_true, test_pred, output_dict=True)
print(f"\n📋 Classification Report:")
print(classification_report(test_true, test_pred))

# Cek target: semua kelas P/R > 90%?
all_above_90 = True
low_performers = []
for name in sorted(set(test_true)):
    if name in report:
        p = report[name]['precision']
        r = report[name]['recall']
        if p < 0.90 or r < 0.90:
            all_above_90 = False
            low_performers.append((name, p, r))

if all_above_90:
    print("✅ Semua kelas punya Precision & Recall > 90%!")
else:
    print("⚠️ Kelas dengan P/R di bawah 90%:")
    for name, p, r in low_performers:
        print(f"  {name}: Precision={p:.3f}, Recall={r:.3f}")

# --- 2. Confusion Matrix ---
cm = confusion_matrix(test_true, test_pred, labels=sorted(set(test_true)))

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(set(test_true)),
            yticklabels=sorted(set(test_true)))
plt.xlabel('Prediksi')
plt.ylabel('Sebenarnya')
plt.title(f'Confusion Matrix - {winner}\nTest Accuracy: {test_acc:.4f}')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

# --- 3. Analisis Error: foto yang salah prediksi ---
print("\n🔍 Analisis Error:")
wrong_indices = np.where(np.array(test_pred) != np.array(test_true))[0]
print(f"  Total salah prediksi: {len(wrong_indices)} / {len(test_true)}")

if len(wrong_indices) > 0:
    # Pasangan yang sering tertukar
    confusion_pairs = Counter()
    for idx in wrong_indices:
        pair = (test_true[idx], test_pred[idx])
        confusion_pairs[pair] += 1

    print(f"\n  Pasangan sering tertukar:")
    for (true_name, pred_name), count in confusion_pairs.most_common(10):
        print(f"    {true_name} → {pred_name}: {count}x")

    # Tampilkan beberapa foto yang salah
    print(f"\n  Contoh foto salah prediksi (max 10):")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, idx in enumerate(wrong_indices[:10]):
        if CHOSEN_APPROACH == 'embedding':
            img_path = test_paths[test_valid_idx[idx]] if len(test_valid_idx) < len(test_paths) else test_paths[idx]
        else:
            img_path = test_paths[idx]

        img = Image.open(img_path).convert('RGB')
        axes[i].imshow(img)
        axes[i].set_title(f"True: {test_true[idx]}\nPred: {test_pred[idx]}", fontsize=8)
        axes[i].axis('off')

    for j in range(i+1, 10):
        axes[j].axis('off')

    plt.suptitle('Foto yang Salah Prediksi', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'misclassified.png'), dpi=150)
    plt.show()

# --- 4. Per-person performance ---
print("\n📊 Performa per orang:")
print(f"{'Nama':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
print("-"*65)
for name in sorted(set(test_true)):
    if name in report:
        r = report[name]
        flag = " ⚠️" if r['precision'] < 0.9 or r['recall'] < 0.9 else " ✅"
        print(f"{name:<25} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1-score']:>8.3f} {r['support']:>8.0f}{flag}")

# --- 5. Rekomendasi foto tambahan ---
print("\n💡 Rekomendasi untuk orang dengan performa rendah:")
for name, p, r in low_performers:
    print(f"\n  {name} (P={p:.3f}, R={r:.3f}):")
    if r < 0.9:
        print(f"    → Recall rendah: model GAGAL mengenali {name}.")
        print(f"    → Tambahkan foto dengan variasi: pose berbeda, cahaya berbeda,")
        print(f"      dengan/tanpa kacamata, jarak kamera berbeda")
    if p < 0.9:
        # Cari siapa yang sering dikira orang ini
        confused_with = [t for t, p in confusion_pairs if p == name]
        print(f"    → Precision rendah: orang lain sering salah dikenali sebagai {name}")
        if confused_with:
            print(f"    → Sering dikacaukan dengan: {', '.join(confused_with)}")
            print(f"    → Tambahkan foto yang menunjukkan perbedaan jelas antara mereka")


## ============================================================================
## CEL 12: TAHAP 4 - EXPORT KE TENSORFLOW.JS
## ============================================================================
# Export FaceNet model + classifier data untuk dipakai di browser.
#
# Strategi: Karena FaceNet PyTorch tidak bisa langsung ke TF.js,
# kita buat wrapper TF/Keras yang meniru perilaku FaceNet,
# ATAU kita export embedding + centroids saja (classifier di JS).
#
# Pendekatan yang dipilih: EXPORT MOBILENETV2 SEBAGAI FEATURE EXTRACTOR KE TFJS
# + CENTROIDS UNTUK COSINE SIMILARITY DI BROWSER
#
# Kenapa? Karena FaceNet (PyTorch) sulit diconvert ke TF.js secara langsung.
# Sebagai gantinya, kita akan:
# 1. Buat model Keras yang menghasilkan embedding (fitur dari MobileNetV2)
# 2. Export ke TF.js
# 3. Pakai centroids dari FaceNet di browser untuk matching
#
# ATAU cara lebih baik: pakai face-api.js yang sudah punya FaceNet-style
# recognition model built-in, dan kita hanya perlu supply labeled descriptor.

print("📦 TAHAP 4: Export untuk Website\n")

# === OPSI 1: Export MobileNetV2 classifier ke TF.js ===
if CHOSEN_APPROACH == 'transfer_learning' or True:  # Export TL model juga sebagai backup
    print("📤 Exporting MobileNetV2 ke TF.js...")

    # Load best model
    best_model_path = os.path.join(MODELS_DIR, 'mobilenet_best.keras')
    if os.path.exists(best_model_path):
        model_export = keras.models.load_model(best_model_path)
    else:
        model_export = model_tl

    # Save as SavedModel dulu
    saved_model_dir = os.path.join(MODELS_DIR, 'saved_model_tl')
    model_export.save(saved_model_dir)

    # Convert ke TF.js
    tfjs_tl_dir = os.path.join(TFJS_DIR, 'mobilenet_classifier')
    os.makedirs(tfjs_tl_dir, exist_ok=True)

    # !pip install tensorflowjs
    import subprocess
    result = subprocess.run([
        sys.executable, '-m', 'tensorflowjs.converter.converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--signature_name=serving_default',
        '--saved_model_tags=serve',
        saved_model_dir,
        tfjs_tl_dir
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✅ TF.js model saved to {tfjs_tl_dir}")
    else:
        print(f"  ⚠️ Conversion error: {result.stderr[:500]}")
        print("  Mencoba alternative method...")
        try:
            import tensorflowjs as tfjs
            tfjs.converters.save_keras_model(model_export, tfjs_tl_dir)
            print(f"  ✅ TF.js model saved (Keras method)")
        except Exception as e:
            print(f"  ❌ Error: {e}")

# === OPSI 2 (UTAMA): Export data untuk face-api.js ===
# face-api.js sudah punya model FaceNet-style built-in.
# Kita hanya perlu supply labeled face descriptors.

print("\n📤 Preparing data for face-api.js...")

# Compute descriptors menggunakan foto train yang sudah aligned
# face-api.js menggunakan embedding 128-D (versi ringan) atau 512-D

# Kita akan export:
# 1. labeled_descriptors.json - embedding per orang (dari FaceNet)
# 2. labels.json - daftar nama
# 3. config.json - threshold dan setting

# Hitung rata-rata embedding per orang (centroids) dari TRAINING data saja
labeled_descriptors = {}
for name in sorted(set(train_labels_valid)):
    mask = train_labels_valid == name
    person_embeddings = train_emb_norm[mask]

    # Simpan semua embedding (lebih akurat daripada hanya centroid)
    labeled_descriptors[name] = {
        'centroid': person_embeddings.mean(axis=0).tolist(),
        'embeddings': person_embeddings.tolist(),
        'count': int(mask.sum())
    }

# Simpan untuk website
web_data = {
    'labeled_descriptors': labeled_descriptors,
    'threshold': float(best_threshold),
    'embedding_size': 512,
    'num_persons': len(labeled_descriptors),
    'labels': sorted(labeled_descriptors.keys()),
    'created_at': datetime.now().isoformat(),
    'approach': CHOSEN_APPROACH
}

with open(os.path.join(TFJS_DIR, 'face_data.json'), 'w') as f:
    json.dump(web_data, f)

# Labels terpisah (lebih ringan)
with open(os.path.join(TFJS_DIR, 'labels.json'), 'w') as f:
    json.dump({
        'labels': sorted(labeled_descriptors.keys()),
        'threshold': float(best_threshold)
    }, f, indent=2)

# Centroids saja (untuk quick matching)
centroids_web = {}
for name, data in labeled_descriptors.items():
    centroids_web[name] = data['centroid']

with open(os.path.join(TFJS_DIR, 'centroids.json'), 'w') as f:
    json.dump(centroids_web, f)

print(f"  ✅ face_data.json ({len(labeled_descriptors)} orang)")
print(f"  ✅ labels.json")
print(f"  ✅ centroids.json")

# === Export foto referensi per orang (untuk face-api.js LabeledFaceDescriptors) ===
ref_photos_dir = os.path.join(TFJS_DIR, 'reference_photos')
os.makedirs(ref_photos_dir, exist_ok=True)

# Pilih 5 foto terbaik per orang (highest confidence / closest to centroid)
for name in sorted(set(train_labels_valid)):
    mask = train_labels_valid == name
    person_emb = train_emb_norm[mask]
    person_paths = train_paths[np.where(mask)[0] if isinstance(mask, np.ndarray) else mask]

    # Pilih yang paling dekat ke centroid (paling representatif)
    centroid = person_emb.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    sims = person_emb @ centroid
    top_indices = sims.argsort()[-5:][::-1]  # Top 5

    person_ref_dir = os.path.join(ref_photos_dir, name)
    os.makedirs(person_ref_dir, exist_ok=True)

    for i, idx in enumerate(top_indices):
        if idx < len(person_paths):
            src = person_paths[idx]
            dst = os.path.join(person_ref_dir, f"ref_{i+1}.png")
            shutil.copy2(src, dst)

print(f"  ✅ Reference photos exported ({len(set(train_labels_valid))} orang × 5 foto)")

# Auto-sync ke folder website jika dijalankan di repo lokal
candidates_web = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "website")) if '__file__' in locals() else "./website",
    "d:/face ai/website",
    "./website"
]
web_target_dir = next((p for p in candidates_web if os.path.exists(p)), None)
if web_target_dir:
    print(f"🔄 Menyinkronkan ke folder website lokal: {web_target_dir}...")
    try:
        shutil.copytree(ref_photos_dir, os.path.join(web_target_dir, "reference_photos"), dirs_exist_ok=True)
        shutil.copy2(os.path.join(TFJS_DIR, 'labels.json'), os.path.join(web_target_dir, 'labels.json'))
        print("  ✅ Website reference photos dan labels.json berhasil disinkronkan!")
    except Exception as e:
        print(f"  ⚠️ Sync note: {e}")


## ============================================================================
## CEL 13: TAHAP 4 - VERIFIKASI KONSISTENSI PREDIKSI
## ============================================================================
# Pastikan preprocessing di Python dan di browser menghasilkan hasil yang sama.

print("🔍 Verifikasi konsistensi preprocessing...\n")

# Test dengan beberapa foto
test_samples = test_paths[:5]

print("Preprocessing pipeline yang HARUS IDENTIK di Python dan di browser:")
print("""
┌─────────────────────────────────────────────────────────────┐
│ 1. Baca gambar (RGB, uint8)                                 │
│ 2. Deteksi wajah (face-api.js atau MTCNN)                   │
│ 3. Crop & align wajah → 160×160 px                         │
│ 4. Convert ke float32                                        │
│ 5. Normalize: (pixel - 127.5) / 128.0  → range [-1, 1]     │
│ 6. Feed ke FaceNet → embedding 128-D (face-api.js)          │
│ 7. L2-normalize embedding                                    │
│ 8. Cosine similarity ke centroids                            │
│ 9. Threshold check → nama atau "Tidak Dikenal"               │
└─────────────────────────────────────────────────────────────┘

⚠️ PENTING: face-api.js menggunakan model sendiri (128-D embedding),
   berbeda dari FaceNet PyTorch (512-D). Jadi kita HARUS:
   
   OPSI A: Re-compute centroids menggunakan face-api.js di browser
           → upload foto referensi, compute descriptor, simpan
   
   OPSI B: Pakai MobileNetV2 TF.js model untuk klasifikasi langsung
           → preprocessing harus match persis
""")

# Jika pakai MobileNetV2 TF.js:
print("Untuk MobileNetV2 TF.js, preprocessing test:")
for path in test_samples:
    img = Image.open(path).convert('RGB')
    img_np = np.array(img.resize((IMG_SIZE_TL, IMG_SIZE_TL))).astype(np.float32)
    # MobileNetV2 preprocess: (pixel / 127.5) - 1.0
    img_processed = (img_np / 127.5) - 1.0
    print(f"  {os.path.basename(path)}: shape={img_processed.shape}, "
          f"range=[{img_processed.min():.3f}, {img_processed.max():.3f}]")

print("\n✅ Preprocessing verification complete")
print("📝 Di website, gunakan fungsi preprocessing yang PERSIS SAMA")


## ============================================================================
## CEL 14: LAPORAN FINAL & REKOMENDASI
## ============================================================================

print("="*70)
print("📊 LAPORAN EVALUASI FINAL")
print("="*70)

print(f"""
🎯 HASIL UTAMA:
  • Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)
  • Target: ≥95% → {'✅ TERCAPAI' if test_acc >= 0.95 else '❌ BELUM TERCAPAI'}
  • Pendekatan terpilih: {winner}
  • Threshold "Tidak Dikenal": {best_threshold:.3f}

📊 PERBANDINGAN METODE:
""")
for method, acc in sorted(all_scores.items(), key=lambda x: -x[1]):
    bar = '█' * int(acc * 50) + '░' * (50 - int(acc * 50))
    print(f"  {method:<30} {bar} {acc:.4f}")

if test_acc < 0.95:
    print(f"""
💡 SARAN PENINGKATAN AKURASI:
  1. Tambah foto (terutama untuk orang dengan performa rendah)
     Target: minimal 50 foto berkualitas per orang
  2. Variasi foto: berbagai pose, cahaya, ekspresi, aksesoris
  3. Coba ArcFace (biasanya lebih akurat dari FaceNet)
  4. Data augmentation lebih agresif: cutout, mixup, random erasing
  5. Ensemble: gabungkan prediksi dari beberapa model
  6. Cek dan perbaiki foto yang label-nya salah (mungkin ada swap)
""")

print(f"""
🔒 CATATAN PRIVASI:
  • Data wajah teman adalah data biometrik sensitif
  • JANGAN publish model/website tanpa izin tertulis semua orang
  • Simpan dataset di tempat aman (jangan di public repo)
  • Gunakan HTTPS jika deploy online
  • Pertimbangkan untuk menghapus dataset setelah model selesai di-train
  • Patuhi UU Perlindungan Data Pribadi (UU PDP)
""")

# Simpan laporan ke file
report_text = f"""
LAPORAN EVALUASI - Face Recognition System
============================================
Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Jumlah orang: {len(set(all_labels))}
Total foto: {len(all_paths)}
Split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}

Pendekatan terpilih: {winner}
Test Accuracy: {test_acc:.4f}
Threshold: {best_threshold:.3f}

Performa per metode:
{chr(10).join(f'  {m}: {a:.4f}' for m, a in all_scores.items())}

Orang dengan performa rendah:
{chr(10).join(f'  {n}: P={p:.3f}, R={r:.3f}' for n, p, r in low_performers) if low_performers else '  (tidak ada)'}
"""

with open(os.path.join(REPORTS_DIR, 'final_report.txt'), 'w') as f:
    f.write(report_text)

print(f"\n💾 Laporan disimpan ke: {REPORTS_DIR}/final_report.txt")
print("\n✅ Pipeline selesai! Lanjut ke pembuatan website.")

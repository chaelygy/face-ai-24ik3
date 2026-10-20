# 🎯 FaceID Kelas - Sistem Pengenalan Wajah Teman Sekelas

## Deskripsi
Sistem pengenalan wajah 23 teman sekelas menggunakan deep learning.
- **Training**: Google Colab (FaceNet embedding + SVM/Centroid classifier)
- **Website**: HTML + CSS + JavaScript + face-api.js (real-time di browser)

## ⚠️ Catatan Privasi
> Data wajah teman adalah data biometrik sensitif yang dilindungi UU PDP.
> - JANGAN publish model/website tanpa izin tertulis SEMUA orang yang fotonya digunakan
> - JANGAN upload dataset ke repository publik
> - Gunakan HTTPS jika deploy online
> - Pertimbangkan hapus dataset setelah training selesai

---

## 📁 Struktur Folder

```
face ai/
├── bahan/                          # Dataset foto (JANGAN COMMIT KE GIT!)
│   ├── citra sinaga/
│   │   └── citra yohana sinaga 5/  # 50 foto
│   ├── Devi Manurung/
│   │   └── Devi Manurung/          # 54 foto
│   ├── ... (23 orang total)
│   └── zhesryn/
│       └── zhesryn/                # 50 foto
│
├── colab_notebook/
│   └── face_recognition_pipeline.py  # Pipeline training (copy ke Colab)
│
├── website/
│   ├── index.html                  # Halaman utama
│   ├── style.css                   # Styling (dark theme premium)
│   ├── app.js                      # Logika pengenalan wajah
│   ├── models/                     # Model face-api.js (auto-download)
│   │   ├── ssd_mobilenetv1_model-*
│   │   ├── face_landmark_68_model-*
│   │   └── face_recognition_model-*
│   ├── reference_photos/           # Foto referensi per orang (dari Colab)
│   │   ├── citra sinaga/ref_1.png ... ref_5.png
│   │   └── .../
│   ├── labels.json                 # Daftar nama + threshold
│   └── face_data.json              # Embedding data (opsional)
│
├── download_faceapi_models.py      # Script download model face-api.js
└── README.md                       # File ini
```

---

## 🚀 Cara Menjalankan

### Langkah 1: Training di Google Colab

1. **Upload dataset** ke Google Drive:
   ```
   Google Drive/face_ai/bahan/nama_orang/subfolder/foto.jpg
   ```

2. **Buka Google Colab** (https://colab.research.google.com)

3. **Buat notebook baru**, lalu copy-paste isi `colab_notebook/face_recognition_pipeline.py`
   per bagian (setiap `## CEL X`) ke sel terpisah di Colab.

4. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. **Sesuaikan path** di CEL 1:
   ```python
   DATASET_ROOT = "/content/drive/MyDrive/face_ai/bahan"
   ```

6. **Jalankan semua sel secara berurutan** (CEL 1 → CEL 14)

7. **Download hasil** dari folder `face_recognition_output/`:
   - `tfjs_model/reference_photos/` → copy ke `website/reference_photos/`
   - `tfjs_model/labels.json` → copy ke `website/labels.json`
   - `tfjs_model/face_data.json` → copy ke `website/face_data.json`
   - `reports/` → lihat laporan evaluasi

### Langkah 2: Setup Website Lokal

1. **Download model face-api.js** (cukup sekali):
   ```bash
   python download_faceapi_models.py
   ```

2. **Copy hasil training** dari Colab ke folder website:
   ```
   website/reference_photos/  ← foto referensi
   website/labels.json        ← daftar nama
   ```

3. **Jalankan web server lokal**:
   
   Opsi A - Python:
   ```bash
   cd website
   python -m http.server 8080
   ```
   
   Opsi B - Node.js:
   ```bash
   npx serve website
   ```
   
   Opsi C - VS Code Live Server Extension

4. **Buka browser**: http://localhost:8080

### Langkah 3: Deploy ke Vercel (Gratis)

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Buat file `.gitignore`**:
   ```
   bahan/
   *.pkl
   *.npy
   __pycache__/
   ```

3. **Deploy**:
   ```bash
   cd website
   vercel
   ```

4. **PENTING**: Pastikan semua teman sudah setuju sebelum deploy online!

---

## 🧠 Arsitektur Sistem

### Pipeline Training (Colab)

```
Foto mentah → MTCNN (deteksi wajah) → Alignment → Crop 160×160
    ↓
FaceNet (embedding 512-D) → L2 normalize
    ↓
├── SVM Classifier      → accuracy
├── k-NN Classifier      → accuracy
├── Centroid + Cosine    → accuracy
└── MobileNetV2 (TL)    → accuracy
    ↓
Pilih yang terbaik → Export ke website
```

### Pipeline Website (Browser)

```
Kamera/Upload → face-api.js SSD (deteksi wajah)
    ↓
Landmark 68 titik → Alignment otomatis
    ↓
FaceRecognitionNet (embedding 128-D)
    ↓
Euclidean distance ke labeled descriptors
    ↓
├── Distance < threshold → Nama + confidence
└── Distance ≥ threshold → "Tidak Dikenal"
    ↓
Frame smoothing (voting N frame) → Hasil stabil
```

### Catatan Penting tentang Embedding

| Aspek | Python (Colab) | Browser (face-api.js) |
|-------|----------------|----------------------|
| Model | FaceNet InceptionResNetV1 | FaceRecognitionNet (face-api.js) |
| Embedding size | 512-D | 128-D |
| Normalization | (pixel - 127.5) / 128 | pixel / 255 |
| Metric | Cosine similarity | Euclidean distance |
| Threshold | ~0.45-0.55 (cosine) | ~0.5-0.6 (euclidean) |

**Karena model berbeda, embedding dari Python TIDAK bisa dipakai langsung di browser.**
Website menghitung ulang descriptor dari foto referensi menggunakan model face-api.js sendiri.

---

## ⚙️ Pengaturan Website

| Setting | Default | Penjelasan |
|---------|---------|-----------|
| Threshold Keyakinan | 0.60 | Euclidean distance max untuk pengenalan. Semakin kecil = semakin ketat |
| Frame Stabilisasi | 5 | Jumlah frame untuk voting. Semakin banyak = semakin stabil tapi lambat |
| Ukuran Deteksi Min | 100px | Abaikan wajah yang terlalu kecil |

---

## 📊 Target Performa

- [x] Akurasi validasi/tes ≥ 95%
- [x] Precision & Recall per orang > 90%
- [x] Orang luar ditolak sebagai "Tidak Dikenal"
- [x] False Accept Rate rendah
- [x] Real-time stabil (tidak berkedip)

---

## 💡 Tips Meningkatkan Akurasi

1. **Tambah foto berkualitas**: Target 50+ foto/orang dengan variasi
2. **Variasi kondisi**: Cahaya berbeda, pose berbeda, ekspresi berbeda
3. **Aksesoris**: Sertakan foto dengan/tanpa kacamata, masker, topi
4. **Jarak kamera**: Dekat, sedang, jauh
5. **Resolusi**: Pastikan wajah cukup besar (minimal 100×100 px)
6. **Hindari duplikat**: Foto yang hampir sama mengurangi variasi training

---

## 📝 Lisensi & Kredit

- **face-api.js**: MIT License (Vincent Mühler)
- **FaceNet**: Apache 2.0 (Google)
- **MobileNetV2**: Apache 2.0 (Google)
- **Dataset**: Pribadi, hanya untuk penggunaan internal kelas

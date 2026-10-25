# Prenava AI Ecosystem 🤰👶

Repositori terintegrasi untuk layanan Machine Learning dan Deep Learning pada ekosistem kesehatan maternal & anak **Prenava**.

---

## 📁 Struktur Repositori

```text
prenava-new/
├── ml-service-stunting/          # [PROD] Layanan Skrining Risiko Stunting Maternal (Flask, Port 5000)
│   ├── api/                      # REST API Flask & model artifacts (.pkl)
│   ├── retrain_optimized.py      # Pipeline pelatihan Random Forest + SMOTETomek + Threshold Medis
│   ├── Dockerfile & compose      # Containerization deployment
│   └── PANDUAN_OPTIMASI_ML_DL... # Dokumentasi pedoman optimasi klinis Kemenkes
│
├── ml-service-sport-new/         # [PROD] Layanan Rekomendasi Olahraga Kehamilan (FastAPI, Port 8000)
│   ├── app/                      # Arsitektur modular FastAPI (API, Services, Rules, Schemas)
│   ├── app/services/             # Hard Medical Safety Filter (Zero False-Safety) & Trimester Rules
│   ├── ml/                       # Script pelatihan & kalibrasi model
│   ├── artifacts/                # Model artifacts (.pkl & .joblib)
│   └── tests/                    # Unit test keamanan medis komprehensif
│
├── ml-service-food/              # [PROD] Layanan Food Scanner & Nutrisi 'Isi Piringku' (FastAPI, Port 9002)
│   ├── app/                      # Arsitektur modular FastAPI (API, Services, Schemas)
│   ├── app/services/             # YOLOv8 Detector & FoodAnalyzer (Kemenkes & Heme-Iron Check)
│   ├── ml/                       # Pipeline training YOLOv8-Nano (train.py)
│   └── tests/                    # Unit & API integration tests
│
├── ml-anemia/                    # [PROD] Layanan Deteksi Anemia Maternal Konjungtiva Mata (Port 9001)
│   ├── main.py                   # API Server deteksi anemia MobileNetV2
│   ├── best_model_final.keras    # Model deep learning anemia
│   └── Dockerfile & compose      # Containerization deployment
│
├── stunting/                     # [R&D / Arsip] Eksplorasi data mentah IFLS-5 & notebook riset
├── ml-service-sports/            # [Arsip] Versi monolitik v1 layanan olahraga
└── ml-service-stunting-main/     # [Arsip] Referensi ekstraksi model stunting
```

---

## 🚀 Layanan Machine Learning Utama

### 1. Skrining Risiko Stunting Maternal (`ml-service-stunting`)
- **Framework:** Flask (Port 5000)
- **Model:** Random Forest Classifier (250 estimators) + SMOTETomek
- **Clinical Alignment:** 18 Fitur klinis Standar Kemenkes RI (KEK/LILA, anemia, kunjungan ANC, kepatuhan TTD, dsb.)
- **Performa Medis:** Recall / Sensitivitas **80.70%** (ambang batas optimal 0.3254) untuk meminimalisasi *False Negatives*.

### 2. Rekomendasi Olahraga Kehamilan (`ml-service-sport-new`)
- **Framework:** FastAPI (Port 8000)
- **Model:** Random Forest Classifier (Akurasi: 99.16%, AUC: 0.9994)
- **Sistem Keamanan Berlapis:**
  - **Layer 1 (Hard Medical Safety Filter):** Kontraindikasi mutlak (Preeklampsia, perdarahan, kontraksi dini, ketuban pecah) langsung membatalkan rekomendasi dan mengarahkan ke faskes (*Zero False-Safety*).
  - **Layer 2 (Trimester Safety Rules):** Filter biomekanik trimester (larangan posisi terlentang/supine pasca minggu ke-16, latihan mobilitas panggul di Trimester 3).
  - **Layer 3 (Live Backend Sync):** Sinkronisasi otomatis aturan olahraga dengan backend Prenava.

### 3. Food Scanner & Verifikasi Nutrisi "Isi Piringku" (`ml-service-food`)
- **Framework:** FastAPI (Port 9002)
- **Model:** YOLOv8-Nano Object Detection (5 Kelas Target Makanan Nusantara)
- **Sistem Keamanan & Evaluasi Gizi:**
  - **Estimasi Proporsi Permukaan:** Kalkulasi luas relatif bounding box terhadap standar piring Kemenkes RI (1/3 Karbo, 1/3 Sayur, 1/6 Protein, 1/6 Buah).
  - **Heme-Iron Deficiency Check:** Peringatan otomatis risiko anemia jika piring tidak mengandung protein hewani kaya zat besi.

### 4. Deteksi Anemia Kehamilan (`ml-anemia`)
- **Framework:** FastAPI / TensorFlow Keras (Port 9001)
- **Model:** Deep Learning CNN MobileNetV2 untuk klasifikasi anemia berdasarkan citra konjungtiva mata.

---

## 🛠️ Panduan Menjalankan Layanan Secara Lokal

### Menjalankan `ml-service-stunting`
```bash
cd ml-service-stunting
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python api/app.py
# Berjalan di http://127.0.0.1:5000
```

### Menjalankan `ml-service-sport-new`
```bash
cd ml-service-sport-new
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Berjalan di http://127.0.0.1:8000 (Dokumentasi Swagger: http://127.0.0.1:8000/docs)
```

---

## 🔒 Catatan Keamanan & Git
Seluruh lingkungan virtual (`venv/`), file cache Python (`__pycache__/`), dan file arsip terkompresi besar (>100MB) telah diabaikan melalui `.gitignore` untuk menjamin integritas repositori dan mematuhi batasan GitHub.

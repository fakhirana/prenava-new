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
├── ml-anemia/                    # [PROD] Layanan Deteksi Anemia Maternal Konjungtiva Mata
│   ├── main.py                   # API Server deteksi anemia
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

### 3. Deteksi Anemia Kehamilan (`ml-anemia`)
- **Framework:** FastAPI / TensorFlow Keras (Port 8000)
- **Model:** Deep Learning CNN untuk klasifikasi anemia berdasarkan citra konjungtiva mata.

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

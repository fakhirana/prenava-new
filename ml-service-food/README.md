# Prenava Food Scanner (`ml-service-food`) 🥗🍛

Microservice Computer Vision berbasis **FastAPI** dan **YOLOv8-Nano** untuk mendeteksi objek makanan multi-kelas, mengestimasi proporsi piring gizi seimbang standar **Kemenkes RI ("Isi Piringku Ibu Hamil")**, dan memverifikasi kecukupan zat besi (*Heme-Iron*) untuk pencegahan anemia maternal dan stunting pada janin.

---

## 📌 Identitas Layanan
- **Framework:** FastAPI
- **Port:** `9002`
- **Model Object Detection:** YOLOv8-Nano (`best.pt` / `yolov8n.pt`)
- **Tanggung Jawab:** Individu 3 (Deep Learning & Computer Vision Specialist)

---

## 🎯 5 Kelas Target Makanan Nusantara
1. `karbohidrat` (nasi putih, nasi merah, jagung, kentang, ubi)
2. `protein_hewani` (ayam, ikan, telur, daging sapi, hati ayam — *sumber utama zat besi heme*)
3. `protein_nabati` (tahu, tempe, kacang-kacangan)
4. `sayuran` (bayam, kangkung, daun singkong, brokoli, wortel)
5. `buah` (pepaya, pisang, jeruk, apel, semangka)

---

## ⚖️ Standar Proporsi "Isi Piringku Ibu Hamil" Kemenkes RI
- **1/3 Piring (33.3%)** = Makanan Pokok (Karbohidrat)
- **1/3 Piring (33.3%)** = Sayuran Hijau
- **1/6 Piring (16.7%)** = Lauk Pauk (Prioritas Protein Hewani kaya Zat Besi)
- **1/6 Piring (16.7%)** = Buah-buahan Segar

### 🛡️ Hard Medical Safety Filter (Verifikasi Zat Besi):
Jika tidak ditemukan `protein_hewani` dalam piring:
- `has_iron_protein: false`
- Penalti skor kepatuhan (maksimal 68).
- Muncul peringatan otomatis:
  > ⚠️ *"Peringatan Gizi: Piring Anda kekurangan Protein Hewani (Zat Besi Heme). Risiko Anemia tinggi!"*

---

## 🚀 Panduan Menjalankan Layanan

### 1. Menjalankan secara Lokal
```bash
cd ml-service-food
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 9002 --reload
```
Akses Swagger UI di: **http://127.0.0.1:9002/docs**

### 2. Menjalankan dengan Docker
```bash
docker-compose up --build -d
```

### 3. Menjalankan Unit Tests
```bash
pytest
```

---

## 📡 API Contract Specification

### Request
- **Method:** `POST`
- **URL:** `http://127.0.0.1:9002/predict-food`
- **Content-Type:** `multipart/form-data`
- **Body:** `file` (File gambar JPG/PNG/WEBP)

### Contoh Response Payload
```json
{
  "status": "success",
  "data": {
    "detected_items": [
      {
        "class": "karbohidrat",
        "name": "Nasi Putih",
        "confidence": 0.94,
        "box": [112, 85, 340, 310],
        "relative_area_percentage": 34.5
      },
      {
        "class": "protein_hewani",
        "name": "Ikan Kembung Goreng",
        "confidence": 0.89,
        "box": [350, 100, 480, 260],
        "relative_area_percentage": 18.2
      },
      {
        "class": "sayuran",
        "name": "Sayur Bayam",
        "confidence": 0.88,
        "box": [120, 320, 310, 490],
        "relative_area_percentage": 28.1
      },
      {
        "class": "buah",
        "name": "Potongan Pepaya",
        "confidence": 0.91,
        "box": [330, 290, 460, 480],
        "relative_area_percentage": 19.2
      }
    ],
    "isi_piringku_analysis": {
      "score": 92,
      "status": "SEIMBANG",
      "has_iron_protein": true,
      "breakdown": {
        "karbohidrat": "34.5% (Standar: 33%) - Sesuai",
        "sayuran": "28.1% (Standar: 33%) - Cukup",
        "lauk_protein": "18.2% (Standar: 17%) - Baik (Kaya Zat Besi)",
        "buah": "19.2% (Standar: 17%) - Baik"
      },
      "clinical_feedback": "Menu makanan sangat baik! Terdeteksi Ikan Kembung Goreng sebagai sumber protein hewani kaya zat besi..."
    },
    "inference_time_ms": 38.4
  }
}
```

---

## 🏋️ Pelatihan Model YOLOv8-Nano
Setelah dataset 5 kelas di-download dari Roboflow:
```bash
python ml/train.py --data dataset_isi_piringku/data.yaml --epochs 75 --batch 16 --imgsz 640
```
Bobot terbaik otomatis diekspor ke folder `models/best.pt`.

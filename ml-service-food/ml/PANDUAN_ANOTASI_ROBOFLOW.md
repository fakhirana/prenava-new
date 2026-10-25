# Panduan Teknis Akuisisi & Anotasi Dataset "Isi Piringku" (Roboflow) 🥗📸

Panduan operasional riset Tugas Akhir untuk **Individu 3 (Deep Learning & Computer Vision Specialist)** Program Studi S1 Sistem Informasi Telkom University.

---

## 🎯 1. Standar 5 Kelas Target Makanan Nusantara

Sesuai pedoman **Kemenkes RI ("Isi Piringku Ibu Hamil")**, seluruh objek makanan diklasifikasikan ke dalam 5 kelas eksklusif:

| Class ID | Nama Kelas | Contoh Makanan Nusantara | Relevansi Klinis Maternal |
| :-: | :--- | :--- | :--- |
| **0** | `karbohidrat` | Nasi putih, nasi merah, jagung, kentang rebus, ubi, singkong | Sumber energi utama ibu hamil (target porsi: 33.3% / sepertiga piring). |
| **1** | `protein_hewani` | Ikan kembung, telur rebus/dadar, ayam goreng, daging sapi, hati ayam | **KRUSIAL: Sumber Zat Besi Heme (*Heme-Iron*) & Asam Folat untuk pencegahan anemia maternal & stunting.** |
| **2** | `protein_nabati` | Tempe bacem/goreng, tahu putih/goreng, kacang merah, oncom | Sumber protein nabati dan serat pendukung. |
| **3** | `sayuran` | Sayur bayam, kangkung, daun singkong, daun katuk, brokoli, wortel | Sumber vitamin, mineral, dan asam folat alami (target porsi: 33.3% / sepertiga piring). |
| **4** | `buah` | Potongan pepaya, pisang, jeruk, apel, semangka, melon | Sumber Vitamin C untuk memaksimalkan penyerapan zat besi (target porsi: 16.7% / seperenam piring). |

---

## 📸 2. Protokol Pengambilan & Pengumpulan Citra

1. **Sudut Pengambilan (Camera Angle):**
   - Foto diambil dari **tampak atas (Top-Down / 90 derajat)** agar rasio luas permukaan 2D merepresentasikan proporsi piring sebenarnya.
2. **Kondisi Pencahayaan:**
   - Pencahayaan alami atau lampu ruangan yang terang tanpa bayangan gelap ekstrem.
3. **Piring Makan:**
   - Gunakan piring bundar standar atau piring sekat.
4. **Target Volume Data:**
   - **Minimal 600 s/d 1000 citra** piring makanan nusantara.
   - Komposisi split: **70% Training**, **20% Validasi**, **10% Testing**.

---

## 🏷️ 3. Langkah Demi Langkah Setup & Anotasi di Roboflow

### Langkah 1: Buat Proyek di Roboflow
1. Buka [https://roboflow.com/](https://roboflow.com/) dan login/daftar.
2. Klik **"Create New Project"**.
3. Atur nama proyek: `prenava-isi-piringku`.
4. Pilih **Project Type**: `Object Detection` (Bukan Classification!).
5. Masukkan 5 kelas target: `karbohidrat`, `protein_hewani`, `protein_nabati`, `sayuran`, `buah`.

### Langkah 2: Upload Gambar
- Drag & drop kumpulan foto makanan yang telah dikumpulkan.
- Atur Train/Valid/Test Split otomatis menjadi **70% / 20% / 10%**.

### Langkah 3: Aturan Anotasi Bounding Box
- Buat kotak bounding box membungkus rapat setiap objek makanan.
- **Kasus Makanan Bertumpuk (*Occlusion*):** Jika lauk (misal ayam/tempe) berada di atas nasi putih:
  - Kotak 1: Meliputi seluruh area nasi putih (`karbohidrat`).
  - Kotak 2: Meliputi objek lauk di atasnya (`protein_hewani` atau `protein_nabati`).
- Pastikan tidak ada objek makanan utama yang terlewat tanpa label.

---

## ⚙️ 4. Konfigurasi Preprocessing & Augmentasi di Roboflow

Sebelum men-generate versi dataset, terapkan aturan berikut pada tab **Generate**:

### A. Preprocessing (Wajib):
1. **Auto-Orient:** ON.
2. **Resize:** Stretch to **640x640** pixels (sesuai resolusi standar input YOLOv8-Nano).

### B. Augmentation (Sesuai Panduan Riset):
1. **Horizontal Flip:** ON (membantu variasi tata letak piring kiri/kanan).
2. **Random Brightness:** Between **-20%** and **+20%** (mengatasi variasi lampu ruangan ibu hamil).
3. **Shear:** $\pm 10^\circ$ Horizontal & Vertical (simulasi sedikit kemiringan sudut kamera smartphone).

---

## 💾 5. Ekspor Dataset & Pemasangan ke Workspace

1. Klik tombol **"Export Dataset"** di Roboflow.
2. Pilih format: **YOLOv8**.
3. Pilih opsi **"download zip to computer"**.
4. Pasang dataset ke repositori Prenava dengan salah satu cara:
   - **Cara A (Menggunakan Skrip Otomatis):**
     ```bash
     cd d:\prenava-new\ml-service-food
     .\venv\Scripts\python.exe ml/download_roboflow.py --zip-file "C:\Users\Ida\Downloads\prenava-isi-piringku.zip"
     ```
   - **Cara B (Ekstrak Manual):**
     Ekstrak isi zip ke folder: `d:\prenava-new\ml-service-food\dataset_isi_piringku\`.

---

## ✅ 6. Verifikasi Integritas Dataset

Jalankan script verifikasi untuk memastikan tidak ada file label yang rusak dan mendapatkan tabel statistik untuk Bab 3 & 4 Laporan Tugas Akhir:

```bash
cd d:\prenava-new\ml-service-food
.\venv\Scripts\python.exe ml/verify_dataset.py
```

Jika output menampilkan `[STATUS: VALID]`, dataset siap digunakan untuk pelatihan model dengan perintah:
```bash
.\venv\Scripts\python.exe ml/train.py --data dataset_isi_piringku/data.yaml --epochs 75 --batch 16 --imgsz 640
```

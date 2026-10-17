# Anemia Detection - Backend API

Backend FastAPI untuk deteksi anemia berbasis gambar konjungtiva mata dengan keamanan production-ready.

---

## 🔒 Fitur Keamanan

✅ **CORS Validation** - Hanya domain spesifik yang diizinkan  
✅ **API Key Authentication** - Mandatory di production mode  
✅ **Rate Limiting** - Proteksi dari abuse/DDoS  
✅ **Error Handling** - Model loading dengan fallback  
✅ **Environment Variables** - Konfigurasi aman tanpa hardcode  

---

## Struktur Folder

```
anemia_detection/
├── main.py                  # FastAPI app (production-safe)
├── requirements.txt         # Dependencies
├── .env.example             # Template environment variables
├── .gitignore               # Git ignore rules
├── best_model_final.keras   # ⚠️ Model file (download dari Colab)
└── README.md
```

---

## ⚙️ Setup & Menjalankan

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/your-username/anemia_detection.git
cd anemia_detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
  pip install -r requirements.txt
```

### 3. Download Model dari Google Colab

Di Colab, jalankan:
```python
from google.colab import files
files.download('best_model_final.keras')
```

Pindahkan file `best_model_final.keras` ke folder project.

### 4. Setup Environment Variables

Copy `.env.example` ke `.env` dan konfigurasi:

```bash
cp .env.example .env
```

Edit `.env` dengan nilai yang sesuai:

```env
# Production Setup
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
RELOAD=False

# Model
MODEL_PATH=best_model_final.keras
IMG_SIZE=224
THRESHOLD=0.4968

# Security
API_KEY=your-super-secret-key-12345

# CORS - domain yang diizinkan
ALLOWED_ORIGINS=https://yourdomain.com,http://localhost:3000

# Rate Limiting
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60
```

### 5. Jalankan Server

**Development Mode (dengan reload):**
```bash
# Di .env set: ENVIRONMENT=development
python main.py
```

**Production Mode (tanpa reload):**
```bash
# Di .env set: ENVIRONMENT=production, API_KEY=your-key
python main.py
```

Server akan berjalan di: `http://localhost:8000`  
Dokumentasi API: `http://localhost:8000/docs`

---

## 📡 API Endpoints

| Method | Endpoint   | Deskripsi | Auth |
|--------|------------|-----------|-----|
| GET    | `/`        | Status API | ❌ |
| GET    | `/health`  | Health check | ❌ |
| POST   | `/predict` | Prediksi anemia | ✅ (prod only) |

---

## 📝 Contoh Request

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediksi (Development Mode - Tanpa API Key)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@foto_mata.jpg"
```

### Prediksi (Production Mode - Dengan API Key)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-super-secret-key-12345" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@foto_mata.jpg"
```

### Response Example
```json
{
  "label": "anemia",
  "confidence": 0.8731,
  "probability_anemia": 0.8731,
  "probability_non_anemia": 0.1269,
  "threshold_used": 0.4968
}
```

---

## 🚀 Deploy ke VPS/Cloud

### Option 1: Railway / Render
1. Push ke GitHub
2. Connect repository di dashboard mereka
3. Set environment variables di dashboard
4. Deploy!

### Option 2: Manual VPS (Ubuntu/Debian)

```bash
# SSH ke VPS
ssh root@your-vps-ip

# Setup
git clone https://github.com/your-username/anemia_detection.git
cd anemia_detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup .env untuk production
nano .env
# ENVIRONMENT=production
# API_KEY=your-very-secure-key
# ALLOWED_ORIGINS=https://yourdomain.com

# Jalankan dengan Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

Atau gunakan **Systemd Service** untuk auto-restart:

```bash
sudo nano /etc/systemd/system/anemia-api.service
```

```ini
[Unit]
Description=Anemia Detection API
After=network.target

[Service]
User=root
WorkingDirectory=/root/anemia_detection
Environment="PATH=/root/anemia_detection/venv/bin"
ExecStart=/root/anemia_detection/venv/bin/gunicorn -w 4 -b 0.0.0.0:8000 main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable anemia-api
sudo systemctl start anemia-api
sudo systemctl status anemia-api
```

---

## 🔐 Security Best Practices

✅ **Selalu gunakan HTTPS** di production  
✅ **Jangan hardcode API key** - gunakan environment variables  
✅ **Ubah API_KEY yang strong** - minimal 32 karakter  
✅ **Limit CORS origins** ke domain yang trusted  
✅ **Monitor rate limiting** untuk deteksi abuse  
✅ **Keep dependencies updated** - `pip list --outdated`  
✅ **Jangan push `.env`ke GitHub** - hanya `.env.example`  

---

## 🐛 Troubleshooting

### Error: "Model file tidak ditemukan"
- Pastikan `best_model_final.keras` ada di folder project
- Check `MODEL_PATH` di file `.env`

### Error: "Invalid or missing API Key"
- Pastikan header `X-API-Key` dikirim dengan benar
- Cek value API key di `.env` dan request harus match

### Error: "Too many requests"
- Rate limiting aktif (default 100 request per 60 detik)
- Adjust `RATE_LIMIT_CALLS` dan `RATE_LIMIT_PERIOD` di `.env`

### Port sudah terpakai
- Ganti `PORT` di `.env`
- Atau close aplikasi lain yang pakai port 8000

---

## 📚 Dokumentasi Interaktif

Setelah server jalan, akses:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📄 License

MIT License

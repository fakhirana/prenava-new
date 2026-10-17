from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import uvicorn

# ─────────────────────────────────────────
#  Konfigurasi
# ─────────────────────────────────────────
MODEL_PATH = "model_anemia_kfold_final.keras"   # Letakkan file model di folder yang sama
IMG_SIZE   = (224, 224)
THRESHOLD  = 0.4968   # Ganti dengan Youden's J threshold optimal dari notebook jika ada

# ─────────────────────────────────────────
#  Load Model saat startup
# ─────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ─────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────
app = FastAPI(
    title="Anemia Detection API",
    description="API untuk deteksi anemia berbasis gambar konjungtiva mata menggunakan MobileNetV2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Ganti dengan domain spesifik saat production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
#  Response Schema
# ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    label: str           # "anemia" atau "non-anemia"
    confidence: float    # Probabilitas 0.0 - 1.0
    probability_anemia: float
    probability_non_anemia: float
    threshold_used: float

# ─────────────────────────────────────────
#  Helper: Preprocess Gambar
# ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Konversi bytes gambar ke tensor siap prediksi."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses gambar: {str(e)}")

# ─────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Anemia Detection API is running!", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload gambar konjungtiva mata, API akan mengembalikan prediksi anemia.
    
    - **file**: File gambar (JPG, PNG, JPEG)
    """
    # Validasi tipe file
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe file tidak didukung: {file.content_type}. Gunakan JPG atau PNG."
        )

    # Baca dan preprocess gambar
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    # Prediksi
    prob = float(model.predict(img_array, verbose=0)[0][0])

    # Tentukan label
    is_anemia = prob <= THRESHOLD   # prob rendah = anemia (class index 0)
    label = "anemia" if is_anemia else "non-anemia"

    return PredictionResponse(
        label=label,
        confidence=round(1 - prob if is_anemia else prob, 4),
        probability_anemia=round(1 - prob, 4),
        probability_non_anemia=round(prob, 4),
        threshold_used=THRESHOLD
    )

# ─────────────────────────────────────────
#  Run (untuk development)
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

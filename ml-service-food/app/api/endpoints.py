import io
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from PIL import Image

from app.schemas import FoodPredictionResponse, FoodPredictionData, HealthCheckResponse
from app.services.detector import detector_service
from app.config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Endpoint untuk mengecek status kesehatan microservice."""
    return HealthCheckResponse(
        service=settings.APP_NAME,
        status="active",
        port=settings.PORT,
        model_loaded=(detector_service.model is not None),
        model_type=detector_service.model_type
    )

@router.post(
    "/predict-food",
    response_model=FoodPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Deteksi Makanan & Verifikasi Proporsi Nutrisi 'Isi Piringku'",
    description="Menerima foto piring makanan, melakukan deteksi objek berbasis YOLOv8, menghitung proporsi luas, dan mengevaluasi standar gizi Kemenkes RI.",
    tags=["Food Detection"]
)
async def predict_food(
    file: UploadFile = File(..., description="File foto piring makanan (format JPG, PNG, atau WEBP)")
):
    # 1. Validasi tipe file
    content_type = file.content_type or ""
    if content_type.lower() not in ALLOWED_EXTENSIONS and not (file.filename and file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Format file '{content_type}' tidak didukung. Harap unggah gambar JPG, PNG, atau WEBP."
        )

    # 2. Baca file gambar
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Gagal membaca file gambar: {str(e)}"
        )

    # 3. Jalankan inferensi & evaluasi proporsi
    detected_items, analysis_result, latency_ms = detector_service.predict(image)

    return FoodPredictionResponse(
        status="success",
        data=FoodPredictionData(
            detected_items=detected_items,
            isi_piringku_analysis=analysis_result,
            inference_time_ms=latency_ms
        )
    )

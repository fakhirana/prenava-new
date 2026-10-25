from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class DetectedItem(BaseModel):
    class_name: str = Field(..., alias="class", description="Kategori kelas makanan (karbohidrat, protein_hewani, etc.)")
    name: str = Field(..., description="Nama spesifik makanan jika teridentifikasi")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Tingkat kepercayaan deteksi")
    box: List[int] = Field(..., description="Koordinat bounding box [x1, y1, x2, y2]")
    relative_area_percentage: float = Field(..., ge=0.0, le=100.0, description="Persentase luas relatif terhadap total area makanan")

    model_config = {
        "populate_by_name": True
    }

class IsiPiringkuBreakdown(BaseModel):
    karbohidrat: str
    sayuran: str
    lauk_protein: str
    buah: str

class IsiPiringkuAnalysis(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Skor kepatuhan gizi piringku (0 - 100)")
    status: str = Field(..., description="Status keseimbangan piring (SEIMBANG / KURANG SEIMBANG / PERLU PERBAIKAN)")
    has_iron_protein: bool = Field(..., description="Apakah piring mengandung protein hewani kaya zat besi")
    breakdown: IsiPiringkuBreakdown
    clinical_feedback: str = Field(..., description="Edukasi dan rekomendasi gizi maternal Kemenkes")

class FoodPredictionData(BaseModel):
    detected_items: List[DetectedItem]
    isi_piringku_analysis: IsiPiringkuAnalysis
    inference_time_ms: float

class FoodPredictionResponse(BaseModel):
    status: str = "success"
    data: FoodPredictionData

class HealthCheckResponse(BaseModel):
    service: str
    status: str
    port: int
    model_loaded: bool
    model_type: str

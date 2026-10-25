import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    APP_NAME: str = "Prenava Food Nutrition Scanner (Isi Piringku)"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 9002
    
    # Model configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.pt"))
    BASE_YOLO_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    
    # 5 Target Classes based on Kemenkes RI
    TARGET_CLASSES: list[str] = [
        "karbohidrat",
        "protein_hewani",
        "protein_nabati",
        "sayuran",
        "buah"
    ]
    
    # Standard Kemenkes "Isi Piringku Ibu Hamil" Proportions
    STANDARD_PROPORTIONS: dict[str, float] = {
        "karbohidrat": 33.33,
        "sayuran": 33.33,
        "lauk_protein": 16.67,  # Combined protein_hewani + protein_nabati
        "buah": 16.67
    }

settings = Settings()

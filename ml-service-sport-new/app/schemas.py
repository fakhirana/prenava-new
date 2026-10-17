from pydantic import BaseModel, Field, field_validator
from typing import List


class ModelInput(BaseModel):
    age: int = Field(..., ge=12, le=60)
    systolic_bp: float = Field(..., ge=60, le=250)
    diastolic_bp: float = Field(..., ge=40, le=150)
    blood_sugar: float = Field(..., ge=20, le=500)
    body_temp: float = Field(..., ge=34.0, le=42.0)
    bmi: float = Field(..., ge=10.0, le=60.0)
    previous_complications: bool
    preexisting_diabetes: bool
    gestational_diabetes: bool
    mental_health_issue: bool
    heart_rate: float = Field(..., ge=30, le=220)


class RecommendationContext(BaseModel):
    gestational_age_weeks: int = Field(..., ge=1, le=42)
    placenta_position_restriction: bool = False
    low_impact_preference: bool = False
    water_access: bool = False
    back_pain: bool = False
    # Indikasi Bahaya Medis Kehamilan (ACOG & Standar Kemenkes RI)
    vaginal_bleeding: bool = False
    premature_contractions: bool = False
    amniotic_fluid_leak: bool = False
    severe_dizziness: bool = False
    chest_pain_or_dyspnea: bool = False


class PredictionRequest(BaseModel):
    model_input: ModelInput
    recommendation_context: RecommendationContext


class ActivityItem(BaseModel):
    code: str
    name: str
    category: str
    recommendation_level: str


class PredictionOutput(BaseModel):
    risk_label: str
    risk_score: float
    probabilities: dict


class RecommendationOutput(BaseModel):
    highly_recommended: List[ActivityItem]
    allowed_with_caution: List[ActivityItem]
    avoid: List[ActivityItem]
    all: List[ActivityItem]
    is_contraindicated: bool = False
    safety_status: str = "safe"  # safe | caution | contraindicated
    safety_alert: str | None = None
    trimester: int = 1


class PredictionResponse(BaseModel):
    prediction: PredictionOutput
    recommendations: RecommendationOutput
    meta: dict
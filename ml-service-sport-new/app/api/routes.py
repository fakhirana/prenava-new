from fastapi import APIRouter, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from app.services.prediction_service import prediction_service
from app.services.recommendation_service import recommendation_service
from app.utils.validation import collect_warnings
from app.utils.response_builder import build_response
from app.exceptions import PredictionError, RuleEngineError
from app.services.artifact_loader import artifact_loader

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "prenava-ml-service"}


@router.get("/model-info")
def model_info() -> dict:
    return artifact_loader.metadata

@router.post("/sync-rules")
def sync_rules() -> dict:
    success = recommendation_service.sync_now()
    if not success:
        raise HTTPException(status_code=502, detail="Gagal sync rules dari backend.")
    return {"status": "ok", "message": "Rulebook berhasil di-sync ulang."}

@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        model_input = payload.model_input.model_dump()
        context = payload.recommendation_context.model_dump()

        prediction = prediction_service.predict(model_input)
        recommendations = recommendation_service.recommend(
            risk_label=prediction["risk_label"],
            model_input=model_input,
            context=context,
        )
        warnings = collect_warnings(model_input)

        return build_response(prediction, recommendations, warnings)
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuleEngineError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada service.")
import pandas as pd

from app.exceptions import PredictionError
from app.services.artifact_loader import artifact_loader
from ml.feature_spec import MODEL_FEATURES


class PredictionService:
    def predict(self, model_input: dict) -> dict:
        try:
            row = {feature: model_input[feature] for feature in MODEL_FEATURES}
            df = pd.DataFrame([row])

            for col in [
                "previous_complications",
                "preexisting_diabetes",
                "gestational_diabetes",
                "mental_health_issue",
            ]:
                df[col] = df[col].astype(int)

            probs = artifact_loader.model.predict_proba(df)[0]
            pred = int(artifact_loader.model.predict(df)[0])

            risk_label = "high" if pred == 1 else "low"
            risk_score = float(probs[1])

            return {
                "risk_label": risk_label,
                "risk_score": round(risk_score, 6),
                "probabilities": {
                    "low": round(float(probs[0]), 6),
                    "high": round(float(probs[1]), 6),
                },
            }
        except Exception as e:
            raise PredictionError(f"Gagal melakukan prediksi: {str(e)}") from e


prediction_service = PredictionService()
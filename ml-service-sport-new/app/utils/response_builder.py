from app.services.artifact_loader import artifact_loader


def build_response(prediction: dict, recommendations: dict, warnings: list[str]) -> dict:
    return {
        "prediction": prediction,
        "recommendations": recommendations,
        "meta": {
            "model_version": artifact_loader.metadata.get("model_version", "unknown"),
            "rules_version": "rules_v1",
            "warnings": warnings,
        },
    }
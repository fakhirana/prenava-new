import json
import joblib

from app.config import settings


class ArtifactLoader:
    def __init__(self) -> None:
        self.model = joblib.load(settings.model_path)
        with open(settings.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)


artifact_loader = ArtifactLoader()
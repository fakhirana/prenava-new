from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "prenava-ml-service"
    app_version: str = "1.0.0"
    model_path: str = "artifacts/model.pkl"
    metadata_path: str = "artifacts/metadata.json"
    rules_path: str = "configs/rules.yml"
    backend_sync_url: str = "https://prenavabe.cloud/api/ml-service/sport-rules"
    backend_sync_timeout_seconds: float = 5.0

    class Config:
        env_file = ".env"


settings = Settings()
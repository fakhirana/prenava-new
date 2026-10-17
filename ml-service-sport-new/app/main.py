from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.logging_config import setup_logging

setup_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(router)
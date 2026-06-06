from fastapi import APIRouter

from backend.routers import detect, health, models

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(models.router, prefix="/v1/models", tags=["models"])
api_router.include_router(detect.router, prefix="/v1", tags=["detect"])

__all__ = ["api_router"]

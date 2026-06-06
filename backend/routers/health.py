from fastapi import APIRouter, Request, status

router = APIRouter(tags=["health"])


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request) -> dict[str, object]:
    registry = getattr(request.app.state, "registry", None)
    active = registry.active_name if registry is not None else None
    if active is None:
        return {"status": "not_ready", "active_model": None}
    return {"status": "ready", "active_model": active}

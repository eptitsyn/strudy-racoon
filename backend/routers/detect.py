from fastapi import APIRouter, HTTPException, Request

from backend.dependencies import RegistryDep, ServiceDep
from backend.services import DetectorService
from detector import ModelRegistry
from shared.contracts import DetectRequest, DetectResponse

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
async def detect(
    body: DetectRequest,
    request: Request,
    service: DetectorService = ServiceDep,
    registry: ModelRegistry = RegistryDep,
) -> DetectResponse:
    if body.model is not None and body.model != registry.active_name:
        raise HTTPException(
            status_code=409,
            detail=(
                f"requested model {body.model!r} is not active "
                f"(active: {registry.active_name!r}); switch first via POST /v1/models/switch"
            ),
        )
    return await service.detect(text=body.text, request_id=request.state.request_id)

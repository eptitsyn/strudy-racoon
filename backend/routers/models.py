from fastapi import APIRouter

from backend.dependencies import RegistryDep
from detector import ModelRegistry
from shared.contracts import (
    ModelsListResponse,
    SwitchRequest,
    SwitchResponse,
)

router = APIRouter()


@router.get("", response_model=ModelsListResponse)
async def list_models(registry: ModelRegistry = RegistryDep) -> ModelsListResponse:
    active = registry.active_name
    assert active is not None, "registry started but active is None"
    return ModelsListResponse(active=active, available=registry.list())


@router.post("/switch", response_model=SwitchResponse)
async def switch_model(
    body: SwitchRequest,
    registry: ModelRegistry = RegistryDep,
) -> SwitchResponse:
    info = await registry.switch(body.name)
    return SwitchResponse(active=info)

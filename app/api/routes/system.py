from fastapi import APIRouter

from app.system.hardware import SystemCapabilities, get_system_capabilities

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/capabilities", response_model=SystemCapabilities)
def capabilities() -> SystemCapabilities:
    return get_system_capabilities()

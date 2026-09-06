from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.api.dependencies import get_cleanup_manager
from app.ingestion.cache import CleanupManager, CleanupResult
from app.system.hardware import SystemCapabilities, get_system_capabilities

router = APIRouter(prefix="/api/system", tags=["system"])


class CleanupRequest(BaseModel):
    dry_run: bool = True


@router.get("/capabilities", response_model=SystemCapabilities)
def capabilities(request: Request) -> SystemCapabilities:
    return get_system_capabilities(request.app.state.settings)


@router.post("/cleanup", response_model=CleanupResult)
def cleanup(
    payload: CleanupRequest,
    manager: CleanupManager = Depends(get_cleanup_manager),
) -> CleanupResult:
    return manager.cleanup(dry_run=payload.dry_run)

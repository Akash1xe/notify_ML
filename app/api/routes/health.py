from fastapi import APIRouter, Request

from app.observability.environment import EnvironmentValidator
from app.observability.models import SystemReadinessReport
from app.version import APP_VERSION

router = APIRouter(tags=["system"])


@router.get("/")
def root(request: Request) -> dict[str, str]:
    return {
        "name": request.app.state.settings.app_name,
        "service": "notify",
        "version": APP_VERSION,
        "phase": "9-release-hardening",
        "current_phase": "9-release-hardening",
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "notify", "version": APP_VERSION}


@router.get("/ready", response_model=SystemReadinessReport)
def readiness(request: Request) -> SystemReadinessReport:
    return EnvironmentValidator(request.app.state.settings).validate(deep=False)

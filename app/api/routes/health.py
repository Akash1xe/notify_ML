from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/")
def root(request: Request) -> dict[str, str]:
    return {
        "name": request.app.state.settings.app_name,
        "service": "notify",
        # Kept for backward compatibility with Phase-1 clients.
        "phase": "1-foundation",
        "current_phase": "3-frame-analysis",
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "notify"}

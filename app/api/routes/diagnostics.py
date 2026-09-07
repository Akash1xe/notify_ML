from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.observability.environment import SensitiveValueRedactor
from app.version import APP_VERSION

router = APIRouter(prefix="/api/jobs", tags=["diagnostics"])


@router.get("/{job_id}/diagnostics")
def job_diagnostics(job_id: str, request: Request) -> dict:
    service = request.app.state.job_service
    job = service.get_job(job_id)
    workspace = request.app.state.workspace_manager.workspace(job_id)
    summary_path = workspace / "diagnostics" / "job_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            return SensitiveValueRedactor.redact(payload)
        except (OSError, json.JSONDecodeError):
            pass
    return SensitiveValueRedactor.redact(
        {
            "version": "1",
            "job_id": job_id,
            "job_status": getattr(job.status, "value", str(job.status)),
            "last_checkpoint": getattr(getattr(job, "stage", None), "value", str(getattr(job, "stage", ""))),
            "pipeline_version": APP_VERSION,
            "warnings": ["DIAGNOSTIC_SUMMARY_NOT_YET_MATERIALIZED"],
        }
    )

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError

from app.api.dependencies import get_job_runner, get_job_service, get_workspace
from app.ingestion.models import IngestionResult
from app.ingestion.youtube.models import YouTubeMetadata
from app.jobs.models import Job, JobCreateRequest, JobListResponse
from app.jobs.runner import JobRunner
from app.jobs.service import JobService
from app.storage.workspace import WorkspaceManager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", response_model=Job, status_code=status.HTTP_201_CREATED)
def create_job(
    payload: JobCreateRequest,
    service: JobService = Depends(get_job_service),
    runner: JobRunner = Depends(get_job_runner),
) -> Job:
    job = service.create_job(str(payload.source_url))
    runner.enqueue(job.id)
    return job


@router.get("", response_model=JobListResponse)
def list_jobs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    service: JobService = Depends(get_job_service),
) -> JobListResponse:
    jobs = service.list_jobs()
    return JobListResponse(items=jobs[offset : offset + limit], total=len(jobs))


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str, service: JobService = Depends(get_job_service)) -> Job:
    return service.get_job(job_id)


@router.post("/{job_id}/cancel", response_model=Job)
def cancel_job(job_id: str, service: JobService = Depends(get_job_service)) -> Job:
    return service.cancel_job(job_id)


@router.post("/{job_id}/retry", response_model=Job)
def retry_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
    runner: JobRunner = Depends(get_job_runner),
) -> Job:
    job = service.retry_job(job_id)
    runner.enqueue(job.id)
    return job


@router.get("/{job_id}/metadata", response_model=YouTubeMetadata)
def get_metadata(
    job_id: str,
    service: JobService = Depends(get_job_service),
    workspace: WorkspaceManager = Depends(get_workspace),
) -> YouTubeMetadata:
    service.get_job(job_id)
    path = workspace.youtube_metadata_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Metadata is not available yet")
    try:
        return YouTubeMetadata.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail="Stored metadata is invalid") from exc


@router.get("/{job_id}/ingestion", response_model=IngestionResult)
def get_ingestion(
    job_id: str,
    service: JobService = Depends(get_job_service),
    workspace: WorkspaceManager = Depends(get_workspace),
) -> IngestionResult:
    service.get_job(job_id)
    path = workspace.ingestion_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Ingestion result is not available yet")
    try:
        return IngestionResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail="Stored ingestion result is invalid") from exc

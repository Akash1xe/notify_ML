from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import get_job_runner, get_job_service
from app.jobs.models import Job, JobCreateRequest, JobListResponse
from app.jobs.runner import JobRunner
from app.jobs.service import JobService

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

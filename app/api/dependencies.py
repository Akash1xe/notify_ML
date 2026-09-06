from __future__ import annotations

from fastapi import Request

from app.jobs.runner import JobRunner
from app.jobs.service import JobService


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


def get_job_runner(request: Request) -> JobRunner:
    return request.app.state.job_runner

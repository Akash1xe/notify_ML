from pathlib import Path

import pytest

from app.core.exceptions import InvalidJobTransitionError
from app.jobs.models import JobStage, JobStatus
from app.jobs.repository import JobRepository
from app.jobs.service import JobService
from app.storage.workspace import WorkspaceManager


def build_service(tmp_path: Path) -> JobService:
    workspace = WorkspaceManager(tmp_path / "jobs")
    workspace.ensure_root()
    return JobService(JobRepository(workspace), workspace)


def test_job_creation_and_persistence(tmp_path: Path):
    service = build_service(tmp_path)
    job = service.create_job("https://www.youtube.com/watch?v=abcdef12345")
    loaded = service.get_job(job.id)
    assert loaded.id == job.id
    assert loaded.status is JobStatus.QUEUED
    assert loaded.progress == 0
    restarted = build_service(tmp_path)
    assert restarted.get_job(job.id).source_url == job.source_url


def test_valid_job_lifecycle(tmp_path: Path):
    service = build_service(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=abcdef12345")
    running = service.mark_running(job.id)
    assert running.status is JobStatus.RUNNING
    service.update_stage(job.id, JobStage.SAMPLING_FRAMES, "Sampling")
    service.update_progress(job.id, 55)
    completed = service.mark_completed(job.id)
    assert completed.status is JobStatus.COMPLETED
    assert completed.progress == 100


def test_progress_is_monotonic(tmp_path: Path):
    service = build_service(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=abcdef12345")
    service.mark_running(job.id)
    service.update_progress(job.id, 55)
    service.update_progress(job.id, 30)
    assert service.get_job(job.id).progress == 55


def test_invalid_transition_is_rejected(tmp_path: Path):
    service = build_service(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=abcdef12345")
    service.mark_running(job.id)
    service.mark_completed(job.id)
    with pytest.raises(InvalidJobTransitionError):
        service.mark_running(job.id)


def test_progress_validation(tmp_path: Path):
    service = build_service(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=abcdef12345")
    service.mark_running(job.id)
    with pytest.raises(ValueError):
        service.update_progress(job.id, 101)


def test_cancel_queued_job(tmp_path: Path):
    service = build_service(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=abcdef12345")
    cancelled = service.cancel_job(job.id)
    assert cancelled.status is JobStatus.CANCELLED


def test_recover_interrupted_job(tmp_path: Path):
    service = build_service(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=abcdef12345")
    service.mark_running(job.id)
    recovered = service.recover_interrupted_jobs()
    assert len(recovered) == 1
    assert recovered[0].status is JobStatus.QUEUED
    assert recovered[0].stage is JobStage.QUEUED


def test_failed_or_cancelled_job_can_be_retried(tmp_path: Path):
    service = build_service(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=abcdef12345")
    service.cancel_job(job.id)
    retried = service.retry_job(job.id)
    assert retried.status is JobStatus.QUEUED
    assert retried.stage is JobStage.QUEUED
    assert retried.error is None

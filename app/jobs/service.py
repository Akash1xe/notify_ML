from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.core.exceptions import InvalidJobTransitionError
from app.jobs.models import Job, JobErrorInfo, JobStage, JobStatus
from app.jobs.repository import JobRepository
from app.jobs.state import validate_transition
from app.storage.workspace import WorkspaceManager


class JobService:
    def __init__(self, repository: JobRepository, workspace_manager: WorkspaceManager) -> None:
        self._repository = repository
        self._workspace_manager = workspace_manager

    def create_job(self, source_url: str) -> Job:
        job = Job(id=str(uuid4()), source_url=source_url)
        self._workspace_manager.create_workspace(job.id)
        return self._repository.save(job)

    def get_job(self, job_id: str) -> Job:
        return self._repository.get(job_id)

    def list_jobs(self) -> list[Job]:
        return self._repository.list()

    def _save(self, job: Job) -> Job:
        job.updated_at = datetime.now(UTC)
        return self._repository.save(job)

    def mark_running(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        validate_transition(job.status, JobStatus.RUNNING)
        job.status = JobStatus.RUNNING
        job.stage = JobStage.PREPARING
        job.started_at = datetime.now(UTC)
        job.message = "Processing started"
        job.error = None
        return self._save(job)

    def update_stage(self, job_id: str, stage: JobStage, message: str) -> Job:
        job = self.get_job(job_id)
        if job.status is not JobStatus.RUNNING:
            raise InvalidJobTransitionError("Only running jobs can change processing stage")
        if stage in {JobStage.COMPLETED, JobStage.FAILED, JobStage.CANCELLED, JobStage.QUEUED}:
            raise InvalidJobTransitionError("Terminal/queue stages require a status transition")
        job.stage = stage
        job.message = message
        return self._save(job)

    def update_progress(self, job_id: str, progress: int, message: str | None = None) -> Job:
        if not 0 <= progress <= 100:
            raise ValueError("Progress must be between 0 and 100")
        job = self.get_job(job_id)
        if job.status is not JobStatus.RUNNING:
            raise InvalidJobTransitionError("Only running jobs can update progress")
        job.progress = progress
        if message is not None:
            job.message = message
        return self._save(job)

    def mark_completed(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        validate_transition(job.status, JobStatus.COMPLETED)
        job.status = JobStatus.COMPLETED
        job.stage = JobStage.COMPLETED
        job.progress = 100
        job.message = "Processing completed"
        job.completed_at = datetime.now(UTC)
        return self._save(job)

    def mark_failed(self, job_id: str, *, code: str, message: str) -> Job:
        job = self.get_job(job_id)
        validate_transition(job.status, JobStatus.FAILED)
        job.status = JobStatus.FAILED
        job.stage = JobStage.FAILED
        job.error = JobErrorInfo(code=code, message=message)
        job.message = "Processing failed"
        job.completed_at = datetime.now(UTC)
        return self._save(job)

    def cancel_job(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        validate_transition(job.status, JobStatus.CANCELLED)
        job.status = JobStatus.CANCELLED
        job.stage = JobStage.CANCELLED
        job.message = "Job cancelled"
        job.completed_at = datetime.now(UTC)
        return self._save(job)

    def recover_interrupted_jobs(self) -> list[Job]:
        recovered: list[Job] = []
        for job in self.list_jobs():
            if job.status is JobStatus.RUNNING:
                # Recovery is intentionally a special transition. A normal API
                # caller can never move RUNNING back to QUEUED.
                job.status = JobStatus.QUEUED
                job.stage = JobStage.QUEUED
                job.message = "Recovered after application restart; queued for retry"
                recovered.append(self._save(job))
        return recovered

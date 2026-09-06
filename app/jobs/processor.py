from __future__ import annotations

import asyncio
from typing import Protocol

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, ProcessingError
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService


class JobProcessor(Protocol):
    async def process(self, job_id: str) -> None: ...


class FakeProcessor:
    """Phase-1 compatibility processor used only for deterministic tests."""

    STEPS = (
        (JobStage.PREPARING, 10, "Preparing lecture processing job"),
        (JobStage.SAMPLING_FRAMES, 40, "Simulating frame-analysis stage"),
        (JobStage.AI_ANALYSIS, 75, "Simulating AI-analysis stage"),
        (JobStage.GENERATING_PDF, 95, "Simulating finalization stage"),
    )

    def __init__(self, service: JobService, checkpoints: CheckpointStore, settings: AppSettings) -> None:
        self._service = service
        self._checkpoints = checkpoints
        self._settings = settings

    async def process(self, job_id: str) -> None:
        for stage, progress, message in self.STEPS:
            current = self._service.get_job(job_id)
            if current.status is JobStatus.CANCELLED:
                raise JobCancelledError(f"Job {job_id} was cancelled")
            self._service.update_stage(job_id, stage, message)
            self._service.update_progress(job_id, progress, message)
            fail_at = self._settings.simulated_failure_at_progress
            if fail_at is not None and progress >= fail_at:
                raise ProcessingError(f"Injected Phase-1 processor failure at {progress}%")
            if self._settings.fake_processor_step_delay:
                await asyncio.sleep(self._settings.fake_processor_step_delay)
            current = self._service.get_job(job_id)
            if current.status is JobStatus.CANCELLED:
                raise JobCancelledError(f"Job {job_id} was cancelled")
            self._checkpoints.mark_completed(job_id, stage)
        self._service.mark_completed(job_id)

from __future__ import annotations

import asyncio
import logging

from app.core.exceptions import InvalidJobTransitionError, JobCancelledError
from app.core.logging import JobEventLogger, log_job
from app.jobs.models import JobStatus
from app.jobs.processor import FakeProcessor
from app.jobs.service import JobService


logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(
        self,
        service: JobService,
        processor: FakeProcessor,
        event_logger: JobEventLogger,
        *,
        max_concurrent_jobs: int,
    ) -> None:
        self._service = service
        self._processor = processor
        self._event_logger = event_logger
        self._max_concurrent_jobs = max_concurrent_jobs
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._queued_ids: set[str] = set()
        self._active_count = 0
        self.max_observed_concurrency = 0

    async def start(self) -> None:
        if self._workers:
            return
        self._service.recover_interrupted_jobs()
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"notify-worker-{index}")
            for index in range(self._max_concurrent_jobs)
        ]
        for job in reversed(self._service.list_jobs()):
            if job.status is JobStatus.QUEUED:
                self.enqueue(job.id)

    async def stop(self) -> None:
        if not self._workers:
            return
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._queued_ids.clear()

    def enqueue(self, job_id: str) -> None:
        if job_id in self._queued_ids:
            return
        self._queued_ids.add(job_id)
        self._queue.put_nowait(job_id)

    async def _worker(self, worker_index: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                if job_id is None:
                    return
                self._queued_ids.discard(job_id)
                job = self._service.get_job(job_id)
                if job.status is not JobStatus.QUEUED:
                    continue

                try:
                    self._service.mark_running(job_id)
                except InvalidJobTransitionError:
                    # A cancellation may win the race between reading QUEUED
                    # and starting it. Skip safely instead of killing a worker.
                    continue

                self._active_count += 1
                self.max_observed_concurrency = max(
                    self.max_observed_concurrency, self._active_count
                )
                self._event_logger.write(
                    job_id, level="INFO", stage="PREPARING", message="Job started"
                )
                log_job(
                    logger,
                    logging.INFO,
                    f"Worker {worker_index} started job",
                    job_id=job_id,
                    stage="PREPARING",
                )

                try:
                    await self._processor.process(job_id)
                    self._event_logger.write(
                        job_id,
                        level="INFO",
                        stage="COMPLETED",
                        message="Job completed",
                    )
                except JobCancelledError:
                    self._event_logger.write(
                        job_id,
                        level="INFO",
                        stage="CANCELLED",
                        message="Job cancelled during processing",
                    )
                except Exception as exc:  # isolate a worker failure to one job
                    current = self._service.get_job(job_id)
                    if current.status is JobStatus.RUNNING:
                        self._service.mark_failed(
                            job_id,
                            code="processing_error",
                            message=str(exc),
                        )
                    self._event_logger.write(
                        job_id,
                        level="ERROR",
                        stage=current.stage.value,
                        message=str(exc),
                    )
                    log_job(
                        logger,
                        logging.ERROR,
                        "Job processor failed",
                        job_id=job_id,
                        stage=current.stage.value,
                        exc_info=True,
                    )
                finally:
                    self._active_count -= 1
            finally:
                self._queue.task_done()

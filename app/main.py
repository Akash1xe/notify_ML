from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health, jobs, system
from app.core.config import AppSettings, get_settings
from app.core.exceptions import (
    InvalidJobTransitionError,
    JobNotFoundError,
    NotifyError,
    StorageError,
)
from app.core.logging import JobEventLogger, configure_logging
from app.jobs.checkpoints import CheckpointStore
from app.jobs.processor import FakeProcessor
from app.jobs.repository import JobRepository
from app.jobs.runner import JobRunner
from app.jobs.service import JobService
from app.storage.workspace import WorkspaceManager


def create_app(settings: AppSettings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(resolved_settings.log_level)
        workspace = WorkspaceManager(resolved_settings.storage_root)
        workspace.ensure_root()
        repository = JobRepository(workspace)
        service = JobService(repository, workspace)
        checkpoints = CheckpointStore(workspace)
        event_logger = JobEventLogger(workspace)
        processor = FakeProcessor(service, checkpoints, resolved_settings)
        runner = JobRunner(
            service,
            processor,
            event_logger,
            max_concurrent_jobs=resolved_settings.max_concurrent_jobs,
        )

        app.state.settings = resolved_settings
        app.state.workspace_manager = workspace
        app.state.job_repository = repository
        app.state.job_service = service
        app.state.checkpoint_store = checkpoints
        app.state.job_runner = runner

        protected = {
            job.id
            for job in service.list_jobs()
            if job.status.value in {"RUNNING", "QUEUED"}
        }
        workspace.cleanup_expired(
            resolved_settings.job_retention_hours,
            protected_job_ids=protected,
        )
        await runner.start()
        try:
            yield
        finally:
            await runner.stop()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(system.router)

    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(_: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": exc.code, "message": str(exc)})

    @app.exception_handler(InvalidJobTransitionError)
    async def invalid_transition_handler(
        _: Request, exc: InvalidJobTransitionError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": exc.code, "message": str(exc)})

    @app.exception_handler(StorageError)
    async def storage_error_handler(_: Request, exc: StorageError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": exc.code, "message": str(exc)})

    @app.exception_handler(NotifyError)
    async def notify_error_handler(_: Request, exc: NotifyError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": exc.code, "message": str(exc)})

    return app


app = create_app()

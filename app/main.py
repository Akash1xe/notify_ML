from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health, jobs, system
from app.core.config import AppSettings, get_settings
from app.core.exceptions import InvalidJobTransitionError, JobNotFoundError, NotifyError, StorageError
from app.core.logging import JobEventLogger, configure_logging
from app.ingestion.cache import CacheManager, CleanupManager
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.youtube.downloader import YouTubeDownloadManager
from app.ingestion.youtube.metadata import YouTubeMetadataExtractor
from app.ingestion.youtube.service import YouTubeService
from app.jobs.checkpoints import CheckpointStore
from app.jobs.processor import FakeProcessor
from app.jobs.repository import JobRepository
from app.jobs.runner import JobRunner
from app.jobs.service import JobService
from app.media.audio import AudioExtractor
from app.media.probe import MediaInspector
from app.media.tools import MediaToolsService
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

        media_tools = MediaToolsService(resolved_settings)
        media_inspector = MediaInspector(resolved_settings, media_tools)
        audio_extractor = AudioExtractor(resolved_settings, media_tools, media_inspector)
        youtube = YouTubeService(
            workspace,
            YouTubeMetadataExtractor(),
            YouTubeDownloadManager(resolved_settings),
        )
        cache = CacheManager(workspace, checkpoints)
        ingestion = IngestionPipeline(
            settings=resolved_settings,
            jobs=service,
            workspace=workspace,
            checkpoints=checkpoints,
            events=event_logger,
            youtube=youtube,
            media_tools=media_tools,
            media_inspector=media_inspector,
            audio_extractor=audio_extractor,
            cache=cache,
        )
        processor = (
            FakeProcessor(service, checkpoints, resolved_settings)
            if resolved_settings.processor_mode == "fake"
            else ingestion
        )
        runner = JobRunner(
            service,
            processor,
            event_logger,
            max_concurrent_jobs=resolved_settings.max_concurrent_jobs,
        )
        cleanup = CleanupManager(workspace, service, resolved_settings.job_retention_hours)

        app.state.settings = resolved_settings
        app.state.workspace_manager = workspace
        app.state.job_repository = repository
        app.state.job_service = service
        app.state.checkpoint_store = checkpoints
        app.state.job_runner = runner
        app.state.youtube_service = youtube
        app.state.media_tools = media_tools
        app.state.media_inspector = media_inspector
        app.state.audio_extractor = audio_extractor
        app.state.cache_manager = cache
        app.state.ingestion_pipeline = ingestion
        app.state.cleanup_manager = cleanup

        await runner.start()  # recovery happens before maintenance cleanup
        cleanup.cleanup(dry_run=False)
        try:
            yield
        finally:
            await runner.stop()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(system.router)

    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(_: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": exc.code, "message": str(exc)})

    @app.exception_handler(InvalidJobTransitionError)
    async def invalid_transition_handler(_: Request, exc: InvalidJobTransitionError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": exc.code, "message": str(exc)})

    @app.exception_handler(StorageError)
    async def storage_error_handler(_: Request, exc: StorageError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": exc.code, "message": str(exc)})

    @app.exception_handler(NotifyError)
    async def notify_error_handler(_: Request, exc: NotifyError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": exc.code, "message": str(exc)})

    return app


app = create_app()

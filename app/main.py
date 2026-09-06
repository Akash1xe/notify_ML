from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health, jobs, system
from app.candidate_analysis.boundaries import StableBoundaryDetector
from app.candidate_analysis.cache import CandidateAnalysisCacheManager
from app.candidate_analysis.evaluation import Phase4Evaluator
from app.candidate_analysis.generation import CandidateGenerator
from app.candidate_analysis.heuristics import CandidateHeuristicAnalyzer
from app.candidate_analysis.pipeline import CandidateAnalysisPipeline
from app.candidate_analysis.ranking import CandidateRankingService
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.candidate_analysis.stability import StabilityWindowDetector
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
from app.video_analysis.cache import FrameAnalysisCacheManager
from app.video_analysis.changes import MajorChangeDetector
from app.video_analysis.differences import VisualDifferenceService
from app.video_analysis.pipeline import FrameAnalysisPipeline, NotifyPipeline
from app.video_analysis.preprocessing import FramePreprocessor
from app.video_analysis.repository import FrameAnalysisRepository
from app.video_analysis.sampling import FrameSampler
from app.video_analysis.timeline import TemporalTimelineService


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
        ingestion_cache = CacheManager(workspace, checkpoints)
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
            cache=ingestion_cache,
        )

        analysis_repository = FrameAnalysisRepository(workspace)
        analysis_cache = FrameAnalysisCacheManager(resolved_settings, workspace, checkpoints)
        frame_analysis = FrameAnalysisPipeline(
            settings=resolved_settings,
            jobs=service,
            workspace=workspace,
            checkpoints=checkpoints,
            events=event_logger,
            sampler=FrameSampler(resolved_settings, workspace, media_tools),
            preprocessor=FramePreprocessor(resolved_settings, workspace),
            differences=VisualDifferenceService(resolved_settings, workspace),
            major_changes=MajorChangeDetector(resolved_settings, workspace),
            timeline=TemporalTimelineService(resolved_settings, workspace),
            cache=analysis_cache,
            repository=analysis_repository,
        )
        candidate_repository = CandidateAnalysisRepository(workspace)
        candidate_cache = CandidateAnalysisCacheManager(resolved_settings, workspace, checkpoints)
        candidate_analysis = CandidateAnalysisPipeline(
            settings=resolved_settings,
            jobs=service,
            workspace=workspace,
            checkpoints=checkpoints,
            events=event_logger,
            frame_cache=analysis_cache,
            frame_repository=analysis_repository,
            stability=StabilityWindowDetector(resolved_settings, workspace),
            boundaries=StableBoundaryDetector(resolved_settings, workspace),
            generator=CandidateGenerator(resolved_settings, workspace),
            heuristics=CandidateHeuristicAnalyzer(resolved_settings, workspace),
            ranking=CandidateRankingService(resolved_settings, workspace),
            cache=candidate_cache,
            repository=candidate_repository,
        )
        phase4_evaluator = Phase4Evaluator(
            resolved_settings, workspace, candidate_repository, analysis_repository
        )
        phase3_pipeline = NotifyPipeline(
            ingestion=ingestion,
            frame_analysis=frame_analysis,
            ingestion_cache=ingestion_cache,
            jobs=service,
        )
        full_pipeline = NotifyPipeline(
            ingestion=ingestion,
            frame_analysis=frame_analysis,
            ingestion_cache=ingestion_cache,
            jobs=service,
            candidate_analysis=candidate_analysis,
        )

        if resolved_settings.processor_mode == "fake":
            processor = FakeProcessor(service, checkpoints, resolved_settings)
        elif resolved_settings.processor_mode == "ingestion":
            processor = ingestion
        elif resolved_settings.processor_mode == "analysis":
            processor = phase3_pipeline
        else:
            processor = full_pipeline

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
        app.state.cache_manager = ingestion_cache
        app.state.ingestion_pipeline = ingestion
        app.state.frame_analysis_cache = analysis_cache
        app.state.frame_analysis_repository = analysis_repository
        app.state.frame_analysis_pipeline = frame_analysis
        app.state.candidate_analysis_cache = candidate_cache
        app.state.candidate_analysis_repository = candidate_repository
        app.state.candidate_analysis_pipeline = candidate_analysis
        app.state.phase4_evaluator = phase4_evaluator
        app.state.notify_pipeline = full_pipeline
        app.state.cleanup_manager = cleanup

        await runner.start()
        cleanup.cleanup(dry_run=False)
        try:
            yield
        finally:
            await runner.stop()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.4.0",
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

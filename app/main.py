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
from app.transcription.alignment import CandidateTranscriptAlignmentService
from app.transcription.cache import TranscriptCacheCoordinator
from app.transcription.context import CandidateTranscriptContextService
from app.transcription.engine import TranscriptionEngine
from app.transcription.evaluation import Phase5Evaluator
from app.transcription.faster_whisper_adapter import FasterWhisperAdapter
from app.transcription.normalization import TranscriptNormalizationService
from app.transcription.pipeline import TranscriptionPipeline
from app.transcription.preparation import AudioPreparationService
from app.transcription.repository import TranscriptionRepository
from app.semantic_analysis.analysis import SemanticCandidateAnalyzer
from app.semantic_analysis.cache import SemanticCacheCoordinator
from app.semantic_analysis.decision import SemanticDecisionEngine
from app.semantic_analysis.evaluation import Phase6Evaluator
from app.semantic_analysis.pipeline import SemanticPipeline
from app.semantic_analysis.preparation import SemanticInputPreparationService
from app.semantic_analysis.prompts import SemanticPromptBuilder
from app.semantic_analysis.repository import SemanticRepository
from app.semantic_analysis.runtime import QwenRuntimeManager
from app.semantic_analysis.temporal_context import TemporalVisualContextService


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

        transcription_repository = TranscriptionRepository(workspace)
        audio_preparation = AudioPreparationService(
            resolved_settings, workspace, checkpoints, transcription_repository, candidate_repository
        )
        whisper_adapter = FasterWhisperAdapter(resolved_settings)
        transcription_engine = TranscriptionEngine(
            resolved_settings, workspace, transcription_repository, whisper_adapter
        )
        transcript_normalization = TranscriptNormalizationService(
            resolved_settings, transcription_repository
        )
        transcript_alignment = CandidateTranscriptAlignmentService(
            resolved_settings, transcription_repository, candidate_repository
        )
        transcript_context = CandidateTranscriptContextService(
            resolved_settings, transcription_repository, candidate_repository
        )
        transcript_cache = TranscriptCacheCoordinator(
            workspace,
            checkpoints,
            transcription_repository,
            candidate_repository,
            audio_preparation,
            transcription_engine,
            transcript_normalization,
            transcript_alignment,
            transcript_context,
        )
        transcription_pipeline = TranscriptionPipeline(
            jobs=service,
            checkpoints=checkpoints,
            events=event_logger,
            preparation=audio_preparation,
            engine=transcription_engine,
            normalization=transcript_normalization,
            alignment=transcript_alignment,
            context=transcript_context,
            cache=transcript_cache,
            repository=transcription_repository,
        )
        phase5_evaluator = Phase5Evaluator(
            resolved_settings, workspace, transcription_repository
        )

        semantic_repository = SemanticRepository(workspace)
        semantic_preparation = SemanticInputPreparationService(
            resolved_settings,
            workspace,
            checkpoints,
            semantic_repository,
            candidate_repository,
            transcription_repository,
            analysis_repository,
        )
        vlm_runtime = QwenRuntimeManager(resolved_settings)
        semantic_prompt_builder = SemanticPromptBuilder(resolved_settings)
        temporal_context = TemporalVisualContextService(
            resolved_settings,
            workspace,
            checkpoints,
            semantic_repository,
            analysis_repository,
            candidate_repository,
        )
        semantic_analyzer = SemanticCandidateAnalyzer(
            resolved_settings,
            workspace,
            checkpoints,
            semantic_repository,
            vlm_runtime,
            semantic_prompt_builder,
        )
        semantic_decision = SemanticDecisionEngine(
            resolved_settings,
            checkpoints,
            semantic_repository,
            candidate_repository,
        )
        semantic_cache = SemanticCacheCoordinator(
            workspace,
            checkpoints,
            semantic_repository,
            candidate_repository,
            transcription_repository,
            analysis_repository,
            semantic_preparation,
            temporal_context,
            semantic_analyzer,
            semantic_decision,
        )
        semantic_pipeline = SemanticPipeline(
            jobs=service,
            checkpoints=checkpoints,
            events=event_logger,
            preparation=semantic_preparation,
            temporal_context=temporal_context,
            analyzer=semantic_analyzer,
            decision=semantic_decision,
            cache=semantic_cache,
            repository=semantic_repository,
        )
        phase6_evaluator = Phase6Evaluator(resolved_settings, semantic_repository)
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
        transcription_full_pipeline = NotifyPipeline(
            ingestion=ingestion,
            frame_analysis=frame_analysis,
            ingestion_cache=ingestion_cache,
            jobs=service,
            candidate_analysis=candidate_analysis,
            transcription=transcription_pipeline,
        )
        semantic_full_pipeline = NotifyPipeline(
            ingestion=ingestion,
            frame_analysis=frame_analysis,
            ingestion_cache=ingestion_cache,
            jobs=service,
            candidate_analysis=candidate_analysis,
            transcription=transcription_pipeline,
            semantic=semantic_pipeline,
        )

        if resolved_settings.processor_mode == "fake":
            processor = FakeProcessor(service, checkpoints, resolved_settings)
        elif resolved_settings.processor_mode == "ingestion":
            processor = ingestion
        elif resolved_settings.processor_mode == "analysis":
            processor = phase3_pipeline
        elif resolved_settings.processor_mode == "candidates":
            processor = full_pipeline
        elif resolved_settings.processor_mode == "transcription":
            processor = transcription_full_pipeline
        else:
            processor = semantic_full_pipeline

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
        app.state.transcription_repository = transcription_repository
        app.state.audio_preparation_service = audio_preparation
        app.state.transcription_engine = transcription_engine
        app.state.transcript_normalization_service = transcript_normalization
        app.state.transcript_alignment_service = transcript_alignment
        app.state.transcript_context_service = transcript_context
        app.state.transcript_cache = transcript_cache
        app.state.transcription_pipeline = transcription_pipeline
        app.state.phase5_evaluator = phase5_evaluator
        app.state.semantic_repository = semantic_repository
        app.state.semantic_input_preparation = semantic_preparation
        app.state.vlm_runtime = vlm_runtime
        app.state.temporal_visual_context_service = temporal_context
        app.state.semantic_analyzer = semantic_analyzer
        app.state.semantic_decision_engine = semantic_decision
        app.state.semantic_cache = semantic_cache
        app.state.semantic_pipeline = semantic_pipeline
        app.state.phase6_evaluator = phase6_evaluator
        if resolved_settings.processor_mode == "semantic":
            app.state.notify_pipeline = semantic_full_pipeline
        elif resolved_settings.processor_mode == "transcription":
            app.state.notify_pipeline = transcription_full_pipeline
        else:
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
        version="0.6.0",
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

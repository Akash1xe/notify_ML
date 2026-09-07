from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError

from app.api.dependencies import (
    get_candidate_analysis_cache,
    get_candidate_analysis_repository,
    get_frame_analysis_cache,
    get_frame_analysis_repository,
    get_job_runner,
    get_job_service,
    get_workspace,
    get_transcription_repository,
    get_transcript_cache,
    get_semantic_repository,
    get_semantic_cache,
)
from app.ingestion.models import IngestionResult
from app.candidate_analysis.cache import CandidateAnalysisCacheManager
from app.candidate_analysis.models import CandidateAnalysisSummary, CandidateCacheSnapshot, SelectionsManifest
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.ingestion.youtube.models import YouTubeMetadata
from app.jobs.models import Job, JobCreateRequest, JobListResponse
from app.jobs.runner import JobRunner
from app.jobs.service import JobService
from app.storage.workspace import WorkspaceManager
from app.video_analysis.cache import FrameAnalysisCacheManager
from app.video_analysis.models import AnalysisCacheSnapshot, FrameAnalysisSummary
from app.video_analysis.repository import FrameAnalysisRepository
from app.transcription.cache import TranscriptCacheCoordinator
from app.transcription.models import (
    CandidateAlignmentManifest,
    CandidateContextsManifest,
    CandidateTranscriptContext,
    ContextStats,
    TranscriptCacheSnapshot,
    TranscriptSummary,
    TranscriptionPreparationManifest,
)
from app.transcription.repository import TranscriptionRepository
from app.semantic_analysis.cache import SemanticCacheCoordinator
from app.semantic_analysis.models import (
    SemanticCacheSnapshot,
    SemanticInputRecord,
    SemanticInputStats,
    SemanticSelectionStats,
    SemanticSummary,
    TemporalContextStats,
    TemporalVisualContext,
)
from app.semantic_analysis.repository import SemanticRepository

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


@router.post("/{job_id}/retry", response_model=Job)
def retry_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
    runner: JobRunner = Depends(get_job_runner),
) -> Job:
    job = service.retry_job(job_id)
    runner.enqueue(job.id)
    return job


@router.get("/{job_id}/metadata", response_model=YouTubeMetadata)
def get_metadata(
    job_id: str,
    service: JobService = Depends(get_job_service),
    workspace: WorkspaceManager = Depends(get_workspace),
) -> YouTubeMetadata:
    service.get_job(job_id)
    path = workspace.youtube_metadata_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Metadata is not available yet")
    try:
        return YouTubeMetadata.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail="Stored metadata is invalid") from exc


@router.get("/{job_id}/ingestion", response_model=IngestionResult)
def get_ingestion(
    job_id: str,
    service: JobService = Depends(get_job_service),
    workspace: WorkspaceManager = Depends(get_workspace),
) -> IngestionResult:
    service.get_job(job_id)
    path = workspace.ingestion_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Ingestion result is not available yet")
    try:
        return IngestionResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail="Stored ingestion result is invalid") from exc


@router.get("/{job_id}/analysis", response_model=FrameAnalysisSummary)
def get_analysis_summary(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: FrameAnalysisRepository = Depends(get_frame_analysis_repository),
) -> FrameAnalysisSummary:
    service.get_job(job_id)
    try:
        return repository.load_summary(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Frame analysis is not available yet") from exc


@router.get("/{job_id}/analysis/cache", response_model=AnalysisCacheSnapshot)
def get_analysis_cache_state(
    job_id: str,
    service: JobService = Depends(get_job_service),
    workspace: WorkspaceManager = Depends(get_workspace),
    cache: FrameAnalysisCacheManager = Depends(get_frame_analysis_cache),
) -> AnalysisCacheSnapshot:
    service.get_job(job_id)
    ingestion_path = workspace.ingestion_path(job_id)
    if not ingestion_path.exists():
        raise HTTPException(status_code=404, detail="Phase-2 ingestion is not available yet")
    try:
        ingestion = IngestionResult.model_validate(json.loads(ingestion_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail="Stored ingestion result is invalid") from exc
    root = workspace.workspace(job_id)
    source = (root / ingestion.source_video).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Stored source path is invalid") from exc
    return cache.inspect(job_id, source)


@router.get("/{job_id}/candidates", response_model=CandidateAnalysisSummary)
def get_candidate_summary(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: CandidateAnalysisRepository = Depends(get_candidate_analysis_repository),
) -> CandidateAnalysisSummary:
    service.get_job(job_id)
    try:
        return repository.load_summary(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Visual candidates are not available yet") from exc


@router.get("/{job_id}/candidates/selections", response_model=SelectionsManifest)
def get_candidate_selections(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: CandidateAnalysisRepository = Depends(get_candidate_analysis_repository),
) -> SelectionsManifest:
    service.get_job(job_id)
    try:
        return repository.load_selections(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Candidate selections are not available yet") from exc


@router.get("/{job_id}/candidates/cache", response_model=CandidateCacheSnapshot)
def get_candidate_cache_state(
    job_id: str,
    service: JobService = Depends(get_job_service),
    frame_repository: FrameAnalysisRepository = Depends(get_frame_analysis_repository),
    cache: CandidateAnalysisCacheManager = Depends(get_candidate_analysis_cache),
) -> CandidateCacheSnapshot:
    service.get_job(job_id)
    try:
        return cache.inspect(
            job_id,
            sampling=frame_repository.load_sampling(job_id),
            preprocessing=frame_repository.load_preprocessing(job_id),
            differences=frame_repository.load_differences(job_id),
            major_changes=frame_repository.load_major_changes(job_id),
            timeline=frame_repository.load_timeline(job_id),
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Candidate cache state is not available yet") from exc


@router.get("/{job_id}/transcript/preparation", response_model=TranscriptionPreparationManifest)
def get_transcription_preparation(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> TranscriptionPreparationManifest:
    service.get_job(job_id)
    try:
        return repository.load_preparation(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Transcription preparation is not available yet") from exc


@router.get("/{job_id}/transcript/summary", response_model=TranscriptSummary)
def get_transcript_summary(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> TranscriptSummary:
    service.get_job(job_id)
    try:
        return repository.load_summary(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Transcript context is not available yet") from exc


@router.get("/{job_id}/transcript/alignment", response_model=CandidateAlignmentManifest)
def get_transcript_alignment(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> CandidateAlignmentManifest:
    service.get_job(job_id)
    try:
        return repository.load_alignment(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Candidate transcript alignment is not available yet") from exc


@router.get("/{job_id}/transcript/contexts", response_model=ContextStats)
def get_transcript_contexts(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> ContextStats:
    service.get_job(job_id)
    try:
        return repository.load_contexts(job_id).stats
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Candidate transcript contexts are not available yet") from exc


@router.get("/{job_id}/transcript/contexts/{candidate_id}", response_model=CandidateTranscriptContext)
def get_candidate_transcript_context(
    job_id: str,
    candidate_id: int,
    service: JobService = Depends(get_job_service),
    repository: TranscriptionRepository = Depends(get_transcription_repository),
) -> CandidateTranscriptContext:
    service.get_job(job_id)
    try:
        return repository.get_candidate_context(job_id, candidate_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Candidate transcript context is not available") from exc


@router.get("/{job_id}/transcript/cache", response_model=TranscriptCacheSnapshot)
def get_transcript_cache_state(
    job_id: str,
    service: JobService = Depends(get_job_service),
    cache: TranscriptCacheCoordinator = Depends(get_transcript_cache),
) -> TranscriptCacheSnapshot:
    service.get_job(job_id)
    return cache.inspect(job_id)


@router.get("/{job_id}/semantic/input", response_model=SemanticInputStats)
def get_semantic_input_summary(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: SemanticRepository = Depends(get_semantic_repository),
) -> SemanticInputStats:
    service.get_job(job_id)
    try:
        return repository.load_input_manifest(job_id).stats
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Semantic input is not available yet") from exc


@router.get("/{job_id}/semantic/input/{candidate_id}", response_model=SemanticInputRecord)
def get_semantic_input_detail(
    job_id: str,
    candidate_id: int,
    service: JobService = Depends(get_job_service),
    repository: SemanticRepository = Depends(get_semantic_repository),
) -> SemanticInputRecord:
    service.get_job(job_id)
    try:
        return repository.get_semantic_input(job_id, candidate_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Semantic candidate input is not available") from exc


@router.get("/{job_id}/semantic/context", response_model=TemporalContextStats)
def get_semantic_context_summary(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: SemanticRepository = Depends(get_semantic_repository),
) -> TemporalContextStats:
    service.get_job(job_id)
    try:
        return repository.load_temporal_contexts(job_id).stats
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Temporal visual context is not available yet") from exc


@router.get("/{job_id}/semantic/context/{candidate_id}", response_model=TemporalVisualContext)
def get_semantic_context_detail(
    job_id: str,
    candidate_id: int,
    service: JobService = Depends(get_job_service),
    repository: SemanticRepository = Depends(get_semantic_repository),
) -> TemporalVisualContext:
    service.get_job(job_id)
    try:
        return repository.get_temporal_context(job_id, candidate_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Temporal visual context is not available") from exc


@router.get("/{job_id}/semantic/selections", response_model=SemanticSelectionStats)
def get_semantic_selections_summary(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: SemanticRepository = Depends(get_semantic_repository),
) -> SemanticSelectionStats:
    service.get_job(job_id)
    try:
        return repository.load_selections(job_id).stats
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Semantic selections are not available yet") from exc


@router.get("/{job_id}/semantic/summary", response_model=SemanticSummary)
def get_semantic_summary(
    job_id: str,
    service: JobService = Depends(get_job_service),
    repository: SemanticRepository = Depends(get_semantic_repository),
) -> SemanticSummary:
    service.get_job(job_id)
    try:
        return repository.load_summary(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Semantic summary is not available yet") from exc


@router.get("/{job_id}/semantic/cache", response_model=SemanticCacheSnapshot)
def get_semantic_cache_state(
    job_id: str,
    service: JobService = Depends(get_job_service),
    cache: SemanticCacheCoordinator = Depends(get_semantic_cache),
) -> SemanticCacheSnapshot:
    service.get_job(job_id)
    return cache.inspect(job_id)

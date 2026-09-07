from __future__ import annotations

from fastapi import Request

from app.ingestion.cache import CleanupManager
from app.candidate_analysis.cache import CandidateAnalysisCacheManager
from app.candidate_analysis.pipeline import CandidateAnalysisPipeline
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.ingestion.pipeline import IngestionPipeline
from app.jobs.runner import JobRunner
from app.jobs.service import JobService
from app.storage.workspace import WorkspaceManager
from app.video_analysis.cache import FrameAnalysisCacheManager
from app.video_analysis.pipeline import FrameAnalysisPipeline
from app.video_analysis.repository import FrameAnalysisRepository


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


def get_job_runner(request: Request) -> JobRunner:
    return request.app.state.job_runner


def get_workspace(request: Request) -> WorkspaceManager:
    return request.app.state.workspace_manager


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.ingestion_pipeline


def get_cleanup_manager(request: Request) -> CleanupManager:
    return request.app.state.cleanup_manager


def get_frame_analysis_pipeline(request: Request) -> FrameAnalysisPipeline:
    return request.app.state.frame_analysis_pipeline


def get_frame_analysis_cache(request: Request) -> FrameAnalysisCacheManager:
    return request.app.state.frame_analysis_cache


def get_frame_analysis_repository(request: Request) -> FrameAnalysisRepository:
    return request.app.state.frame_analysis_repository


def get_candidate_analysis_pipeline(request: Request) -> CandidateAnalysisPipeline:
    return request.app.state.candidate_analysis_pipeline


def get_candidate_analysis_cache(request: Request) -> CandidateAnalysisCacheManager:
    return request.app.state.candidate_analysis_cache


def get_candidate_analysis_repository(request: Request) -> CandidateAnalysisRepository:
    return request.app.state.candidate_analysis_repository


def get_transcription_pipeline(request: Request):
    return request.app.state.transcription_pipeline


def get_transcription_repository(request: Request):
    return request.app.state.transcription_repository


def get_transcript_cache(request: Request):
    return request.app.state.transcript_cache


def get_phase5_evaluator(request: Request):
    return request.app.state.phase5_evaluator


def get_semantic_repository(request: Request):
    return request.app.state.semantic_repository


def get_semantic_cache(request: Request):
    return request.app.state.semantic_cache


def get_semantic_pipeline(request: Request):
    return request.app.state.semantic_pipeline


def get_vlm_runtime(request: Request):
    return request.app.state.vlm_runtime


def get_phase6_evaluator(request: Request):
    return request.app.state.phase6_evaluator


def get_screenshot_repository(request: Request):
    return request.app.state.screenshot_repository


def get_phase7_cache(request: Request):
    return request.app.state.phase7_cache


def get_screenshot_pipeline(request: Request):
    return request.app.state.screenshot_pipeline


def get_phase7_evaluator(request: Request):
    return request.app.state.phase7_evaluator

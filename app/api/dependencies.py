from __future__ import annotations

from fastapi import Request

from app.ingestion.cache import CleanupManager
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

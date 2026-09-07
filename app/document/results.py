from __future__ import annotations

from pathlib import Path

from app.core.config import AppSettings
from app.core.exceptions import DocumentCorruptError, DocumentNotReadyError, DocumentStaleError
from app.document.cache import CP_FINAL_DOCUMENT_READY, DocumentCacheCoordinator
from app.document.models import (
    DocumentApiStatus,
    DocumentDownloadDescriptor,
    DocumentScreenshotListResponse,
    DocumentScreenshotResponse,
    DocumentStatusResponse,
    DocumentSummaryResponse,
    DocumentCacheState,
)
from app.document.repository import DocumentRepository
from app.document.utils import resolve_document_artifact_path, safe_download_filename
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStatus
from app.jobs.service import JobService
from app.screenshots.repository import ScreenshotRepository
from app.storage.workspace import WorkspaceManager


class DocumentResultService:
    def __init__(
        self,
        settings: AppSettings,
        jobs: JobService,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        repository: DocumentRepository,
        screenshots: ScreenshotRepository,
        cache: DocumentCacheCoordinator,
    ) -> None:
        self._settings = settings
        self._jobs = jobs
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = repository
        self._screenshots = screenshots
        self._cache = cache

    def _light_snapshot(self, job_id: str):
        return self._cache.inspect(job_id, deep_pdf=False, deep_images=False)

    def status(self, job_id: str) -> DocumentStatusResponse:
        job = self._jobs.get_job(job_id)
        input_path = self._workspace.document_input_manifest_path(job_id)
        title = "Lecture Notes"
        page_count = file_size = count = 0
        if job.status is JobStatus.FAILED and job.error and job.error.code == "pdf_empty_document":
            return DocumentStatusResponse(job_id=job_id, status=DocumentApiStatus.EMPTY, error_code=job.error.code)
        if not input_path.exists():
            status = DocumentApiStatus.PROCESSING if job.status in {JobStatus.QUEUED, JobStatus.RUNNING} else DocumentApiStatus.NOT_STARTED
            if job.status is JobStatus.FAILED:
                status = DocumentApiStatus.FAILED
            return DocumentStatusResponse(job_id=job_id, status=status, error_code=job.error.code if job.error else None)
        try:
            input_manifest = self._repository.load_input_manifest(job_id)
            title = input_manifest.source.title or title
            count = input_manifest.stats.item_count
            snapshot = self._light_snapshot(job_id)
            if snapshot.document_input.state is DocumentCacheState.CORRUPT or snapshot.layout.state is DocumentCacheState.CORRUPT or snapshot.render_plan.state is DocumentCacheState.CORRUPT or snapshot.pdf.state is DocumentCacheState.CORRUPT:
                status = DocumentApiStatus.CORRUPT
            elif snapshot.document_input.state is DocumentCacheState.STALE or snapshot.layout.state is DocumentCacheState.STALE or snapshot.render_plan.state is DocumentCacheState.STALE or snapshot.pdf.state is DocumentCacheState.STALE:
                status = DocumentApiStatus.STALE
            elif snapshot.pdf.valid and self._checkpoints.is_completed(job_id, CP_FINAL_DOCUMENT_READY):
                status = DocumentApiStatus.READY
                pdf = self._repository.load_pdf_manifest(job_id)
                page_count, file_size = pdf.page_count, pdf.file_size_bytes
            elif job.status is JobStatus.FAILED:
                status = DocumentApiStatus.FAILED
            elif job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                status = DocumentApiStatus.PROCESSING
            else:
                status = DocumentApiStatus.STALE
            return DocumentStatusResponse(
                job_id=job_id,
                status=status,
                checkpoint=CP_FINAL_DOCUMENT_READY if self._checkpoints.is_completed(job_id, CP_FINAL_DOCUMENT_READY) else None,
                document_title=title,
                page_count=page_count,
                file_size_bytes=file_size,
                final_screenshot_count=count,
                download_available=status is DocumentApiStatus.READY,
                error_code=job.error.code if job.error else None,
            )
        except Exception:
            return DocumentStatusResponse(job_id=job_id, status=DocumentApiStatus.CORRUPT, document_title=title, error_code="document_corrupt")

    def summary(self, job_id: str) -> DocumentSummaryResponse:
        status = self.status(job_id)
        if status.status is not DocumentApiStatus.READY:
            raise DocumentNotReadyError("The PDF is not ready yet.")
        summary = self._repository.load_summary(job_id)
        return DocumentSummaryResponse(
            document_title=summary.document_title,
            page_count=summary.page_count,
            final_screenshot_count=summary.final_screenshot_count,
            suppressed_duplicate_count=summary.suppressed_duplicate_count,
            file_size_bytes=summary.file_size_bytes,
            first_timestamp_seconds=summary.first_timestamp_seconds,
            last_timestamp_seconds=summary.last_timestamp_seconds,
            content_type_counts=summary.content_type_counts,
        )

    def screenshots(self, job_id: str, *, offset: int, limit: int) -> DocumentScreenshotListResponse:
        self._jobs.get_job(job_id)
        manifest = self._repository.load_input_manifest(job_id)
        items = manifest.items[offset : offset + limit]
        return DocumentScreenshotListResponse(
            items=[
                DocumentScreenshotResponse(
                    order=x.order,
                    candidate_id=x.candidate_id,
                    timestamp_seconds=x.timestamp_seconds,
                    timestamp_display=x.timestamp_display,
                    content_type=x.content_type,
                    width=x.width,
                    height=x.height,
                    quality_score=x.quality_score,
                    semantic_score=x.semantic_decision_score,
                    preview_available=self._settings.document_preview_enabled,
                )
                for x in items
            ],
            total=len(manifest.items),
            offset=offset,
            limit=limit,
        )

    def cache_snapshot(self, job_id: str):
        self._jobs.get_job(job_id)
        return self._light_snapshot(job_id)

    def preview_path(self, job_id: str, candidate_id: int) -> tuple[Path, str]:
        self._jobs.get_job(job_id)
        if not self._settings.document_preview_enabled:
            raise DocumentNotReadyError("Document previews are disabled.")
        item = self._repository.get_document_item(job_id, candidate_id)
        path = resolve_document_artifact_path(self._workspace, job_id, item.image_relative_path)
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "JPG": "image/jpeg", "WEBP": "image/webp"}.get(item.image_format, "application/octet-stream")
        return path, mime

    def download(self, job_id: str) -> DocumentDownloadDescriptor:
        status = self.status(job_id)
        if status.status is DocumentApiStatus.STALE:
            raise DocumentStaleError("The generated PDF is stale and must be regenerated.")
        if status.status is DocumentApiStatus.CORRUPT:
            raise DocumentCorruptError("The generated PDF could not be validated.")
        if status.status is not DocumentApiStatus.READY:
            raise DocumentNotReadyError("The PDF is not ready yet.")
        input_manifest = self._repository.load_input_manifest(job_id)
        render = self._repository.load_render_manifest(job_id)
        pdf = self._repository.load_pdf_manifest(job_id)
        pdf_check, _ = self._cache.validate_pdf(job_id, input_manifest, render, deep=True)
        if not pdf_check.valid:
            raise DocumentCorruptError("The generated PDF could not be validated.")
        path = self._repository.get_pdf_path(job_id)
        return DocumentDownloadDescriptor(
            path=str(path),
            filename=safe_download_filename(input_manifest.source.title, max_length=self._settings.document_download_filename_max_length),
            file_size_bytes=pdf.file_size_bytes,
            sha256=pdf.file_sha256,
            etag=f'"sha256-{pdf.file_sha256}"',
        )

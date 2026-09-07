from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from app.api.dependencies import get_document_result_service, get_job_service
from app.core.exceptions import DocumentCorruptError, DocumentNotReadyError, DocumentStaleError
from app.document.models import (
    DocumentScreenshotListResponse,
    DocumentStatusResponse,
    DocumentSummaryResponse,
    Phase8CacheSnapshot,
)
from app.document.results import DocumentResultService
from app.jobs.service import JobService

router = APIRouter(prefix="/api/jobs/{job_id}/document", tags=["Document"])


def _conflict(exc: Exception) -> HTTPException:
    code = getattr(exc, "code", "document_not_ready")
    return HTTPException(status_code=409, detail={"code": code, "message": str(exc)})


@router.get("", response_model=DocumentStatusResponse, summary="Get generated document status")
def document_status(job_id: str, service: DocumentResultService = Depends(get_document_result_service)) -> DocumentStatusResponse:
    return service.status(job_id)


@router.get("/summary", response_model=DocumentSummaryResponse, summary="Get generated document summary")
def document_summary(job_id: str, service: DocumentResultService = Depends(get_document_result_service)) -> DocumentSummaryResponse:
    try:
        return service.summary(job_id)
    except (DocumentNotReadyError, DocumentStaleError, DocumentCorruptError) as exc:
        raise _conflict(exc) from exc


@router.get("/screenshots", response_model=DocumentScreenshotListResponse, summary="List final document screenshots")
def document_screenshots(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    service: DocumentResultService = Depends(get_document_result_service),
) -> DocumentScreenshotListResponse:
    try:
        return service.screenshots(job_id, offset=offset, limit=limit)
    except Exception as exc:
        if getattr(exc, "code", "") == "job_not_found":
            raise
        raise HTTPException(status_code=404, detail="Document screenshots are not available yet") from exc


@router.get("/cache", response_model=Phase8CacheSnapshot, summary="Inspect document cache state")
def document_cache(job_id: str, service: DocumentResultService = Depends(get_document_result_service)) -> Phase8CacheSnapshot:
    return service.cache_snapshot(job_id)


@router.get("/screenshots/{candidate_id}/preview", summary="Preview a final document screenshot")
def screenshot_preview(job_id: str, candidate_id: int, service: DocumentResultService = Depends(get_document_result_service)):
    try:
        path, mime = service.preview_path(job_id, candidate_id)
    except Exception as exc:
        if getattr(exc, "code", "") == "job_not_found":
            raise
        raise HTTPException(status_code=404, detail="Screenshot preview is not available") from exc
    return FileResponse(path=path, media_type=mime, headers={"Cache-Control": "private, max-age=0, must-revalidate"})


@router.get("/download", summary="Download generated PDF")
def document_download(job_id: str, request: Request, service: DocumentResultService = Depends(get_document_result_service)):
    try:
        descriptor = service.download(job_id)
    except (DocumentNotReadyError, DocumentStaleError, DocumentCorruptError) as exc:
        raise _conflict(exc) from exc
    if request.headers.get("if-none-match") == descriptor.etag:
        return Response(status_code=304, headers={"ETag": descriptor.etag, "Cache-Control": "private, max-age=0, must-revalidate"})
    return FileResponse(
        path=Path(descriptor.path),
        media_type="application/pdf",
        filename=descriptor.filename,
        headers={
            "ETag": descriptor.etag,
            "Cache-Control": "private, max-age=0, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
    )

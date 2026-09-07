from __future__ import annotations

import asyncio
from collections import defaultdict

from app.core.exceptions import DocumentPipelineError, JobCancelledError
from app.core.logging import JobEventLogger
from app.document.cache import CP_FINAL_DOCUMENT_READY, DocumentCacheCoordinator
from app.document.input import DocumentInputBuilder
from app.document.layout import DocumentLayoutEngine
from app.document.models import DocumentResumeStage, FinalDocumentSummary
from app.document.pdf import DocumentPdfGenerator
from app.document.render import DocumentRenderPlanner
from app.document.repository import DocumentRepository
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService
from app.screenshots.repository import ScreenshotRepository


class DocumentPipeline:
    """Phase 8 orchestrator: deterministic screenshots -> validated final PDF."""

    def __init__(
        self,
        *,
        jobs: JobService,
        checkpoints: CheckpointStore,
        events: JobEventLogger,
        screenshots: ScreenshotRepository,
        repository: DocumentRepository,
        cache: DocumentCacheCoordinator,
        input_builder: DocumentInputBuilder,
        layout: DocumentLayoutEngine,
        render: DocumentRenderPlanner,
        pdf: DocumentPdfGenerator,
    ) -> None:
        self._jobs = jobs
        self._checkpoints = checkpoints
        self._events = events
        self._screenshots = screenshots
        self._repository = repository
        self._cache = cache
        self._input = input_builder
        self._layout = layout
        self._render = render
        self._pdf = pdf
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _cancelled(self, job_id: str) -> bool:
        return self._jobs.get_job(job_id).status is JobStatus.CANCELLED

    def _guard(self, job_id: str) -> None:
        if self._cancelled(job_id):
            raise JobCancelledError(f"Job {job_id} was cancelled")

    def _stage(self, job_id: str, stage: JobStage, message: str) -> None:
        self._guard(job_id)
        self._jobs.update_stage(job_id, stage, message)
        self._events.write(job_id, level="INFO", stage=stage.value, message=message)

    def _progress(self, job_id: str, value: int, message: str) -> None:
        if not self._cancelled(job_id):
            self._jobs.update_progress(job_id, value, message)

    async def process(self, job_id: str, *, finalize_job: bool = True) -> FinalDocumentSummary:
        async with self._locks[job_id]:
            return await self._process_locked(job_id, finalize_job=finalize_job)

    async def _process_locked(self, job_id: str, *, finalize_job: bool) -> FinalDocumentSummary:
        if not self._checkpoints.is_completed(job_id, "FINAL_SCREENSHOTS_READY"):
            raise DocumentPipelineError("FINAL_SCREENSHOTS_READY is required before document generation.")

        removed = self._cache.cleanup_temp_artifacts(job_id)
        self._stage(job_id, JobStage.VALIDATING_DOCUMENT_CACHE, "Checking reusable document work")
        snapshot = await asyncio.to_thread(self._cache.inspect, job_id, deep_pdf=False)
        self._cache.repair_checkpoints(job_id, snapshot)
        self._events.write(
            job_id,
            level="INFO",
            stage="DOCUMENT_CACHE",
            message=f"Phase-8 resume stage selected: {snapshot.resume_plan.resume_stage.value}; removed {len(removed)} temporary artifacts",
        )

        input_manifest = None
        layout_manifest = None
        render_manifest = None
        pdf_manifest = None

        if snapshot.resume_plan.rebuild_document_input:
            self._stage(job_id, JobStage.PREPARING_DOCUMENT_INPUT, "Preparing final screenshots for document generation")
            input_manifest = await asyncio.to_thread(self._input.build, job_id)
            self._progress(job_id, 97, "Document input ready")
        else:
            input_manifest = self._repository.load_input_manifest(job_id)
            self._progress(job_id, 97, "Document input cache reused")
        self._guard(job_id)

        if snapshot.resume_plan.rebuild_layout:
            self._stage(job_id, JobStage.PLANNING_DOCUMENT_LAYOUT, "Planning document page layout")
            layout_manifest = await asyncio.to_thread(self._layout.build, job_id, input_manifest)
            self._progress(job_id, 98, "Document layout ready")
        else:
            layout_manifest = self._repository.load_layout_manifest(job_id)
            self._progress(job_id, 98, "Document layout cache reused")
        self._guard(job_id)

        if snapshot.resume_plan.rebuild_render_plan:
            self._stage(job_id, JobStage.PREPARING_DOCUMENT_RENDER_PLAN, "Preparing screenshots and captions for PDF rendering")
            render_manifest = await asyncio.to_thread(self._render.build, job_id, input_manifest, layout_manifest)
            self._progress(job_id, 99, "Document render plan ready")
        else:
            render_manifest = self._repository.load_render_manifest(job_id)
            self._progress(job_id, 99, "Document render-plan cache reused")
        self._guard(job_id)

        if snapshot.resume_plan.rebuild_pdf:
            self._stage(job_id, JobStage.GENERATING_PDF, "Generating final PDF document")
            pdf_manifest = await asyncio.to_thread(
                self._pdf.generate,
                job_id,
                input_manifest,
                render_manifest,
                lambda: self._cancelled(job_id),
            )
        else:
            pdf_manifest = self._repository.load_pdf_manifest(job_id)
        self._guard(job_id)

        final_snapshot = await asyncio.to_thread(self._cache.inspect, job_id, deep_pdf=True)
        if final_snapshot.resume_plan.resume_stage is not DocumentResumeStage.DOCUMENT_READY:
            raise DocumentPipelineError("Phase-8 artifacts failed final consistency validation.")
        self._cache.repair_checkpoints(job_id, final_snapshot)

        phase7_summary = self._screenshots.load_summary(job_id)
        summary = FinalDocumentSummary(
            document_ready=True,
            document_title=input_manifest.source.title or "Lecture Notes",
            page_count=pdf_manifest.page_count,
            final_screenshot_count=input_manifest.stats.item_count,
            suppressed_duplicate_count=phase7_summary.suppressed_duplicate_count,
            file_size_bytes=pdf_manifest.file_size_bytes,
            first_timestamp_seconds=input_manifest.stats.first_timestamp_seconds,
            last_timestamp_seconds=input_manifest.stats.last_timestamp_seconds,
            content_type_counts=input_manifest.stats.content_type_counts,
            cache_hits={
                "document_input": not snapshot.resume_plan.rebuild_document_input,
                "layout": not snapshot.resume_plan.rebuild_layout,
                "render_plan": not snapshot.resume_plan.rebuild_render_plan,
                "pdf": not snapshot.resume_plan.rebuild_pdf,
            },
        )
        self._repository.save_summary(job_id, summary)
        self._checkpoints.mark_completed(job_id, CP_FINAL_DOCUMENT_READY)
        self._progress(job_id, 100, "Your visual notes are ready")
        self._events.write(job_id, level="INFO", stage=CP_FINAL_DOCUMENT_READY, message="Final document ready")
        if finalize_job:
            self._jobs.mark_completed(job_id, message="Your visual notes are ready")
        return summary

from __future__ import annotations

import os

from app.document.input import CP_DOCUMENT_INPUT_READY, DocumentInputBuilder
from app.document.layout import CP_DOCUMENT_LAYOUT_READY, DocumentLayoutEngine
from app.document.models import (
    DOCUMENT_INPUT_ALGORITHM_VERSION, DOCUMENT_INPUT_MANIFEST_VERSION,
    DOCUMENT_LAYOUT_ALGORITHM_VERSION, DOCUMENT_LAYOUT_MANIFEST_VERSION,
    DOCUMENT_RENDER_PLAN_ALGORITHM_VERSION, DOCUMENT_RENDER_MANIFEST_VERSION,
    PDF_MANIFEST_VERSION, PDF_RENDER_ALGORITHM_VERSION,
    DocumentCacheState,
    DocumentResumeStage,
    DocumentStageCheck,
    Phase8CacheSnapshot,
    Phase8ResumePlan,
)
from app.document.pdf import CP_PDF_READY, DocumentPdfGenerator
from app.document.render import CP_DOCUMENT_RENDER_PLAN_READY, DocumentRenderPlanner
from app.document.repository import DocumentRepository
from app.document.utils import file_sha256, resolve_document_artifact_path
from app.jobs.checkpoints import CheckpointStore
from app.screenshots.repository import ScreenshotRepository
from app.storage.workspace import WorkspaceManager

CP_FINAL_DOCUMENT_READY = "FINAL_DOCUMENT_READY"


class DocumentCacheCoordinator:
    def __init__(self, workspace: WorkspaceManager, checkpoints: CheckpointStore, screenshots: ScreenshotRepository, repository: DocumentRepository, input_builder: DocumentInputBuilder, layout: DocumentLayoutEngine, render: DocumentRenderPlanner, pdf: DocumentPdfGenerator) -> None:
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._screenshots = screenshots
        self._repository = repository
        self._input = input_builder
        self._layout = layout
        self._render = render
        self._pdf = pdf

    def cleanup_temp_artifacts(self, job_id: str) -> list[str]:
        final_pdf = self._workspace.document_pdf_path(job_id)
        backup_pdf = final_pdf.with_suffix(".previous.pdf")
        removed: list[str] = []
        if backup_pdf.exists():
            if not final_pdf.exists():
                os.replace(backup_pdf, final_pdf)
            else:
                backup_pdf.unlink(missing_ok=True)
                removed.append(backup_pdf.name)
        paths = [
            self._workspace.document_input_manifest_path(job_id).with_suffix(".json.tmp"),
            self._workspace.document_layout_path(job_id).with_suffix(".json.tmp"),
            self._workspace.document_render_plan_path(job_id).with_suffix(".json.tmp"),
            self._workspace.document_pdf_manifest_path(job_id).with_suffix(".json.tmp"),
            self._workspace.document_temp_pdf_path(job_id),
        ]
        for path in paths:
            if path.exists():
                path.unlink(missing_ok=True)
                removed.append(path.name)
        return removed

    @staticmethod
    def _missing(reason: str) -> DocumentStageCheck:
        return DocumentStageCheck(state=DocumentCacheState.MISSING, reasons=[reason])

    @staticmethod
    def _stale(reason: str) -> DocumentStageCheck:
        return DocumentStageCheck(state=DocumentCacheState.STALE, reasons=[reason])

    @staticmethod
    def _corrupt(reason: str) -> DocumentStageCheck:
        return DocumentStageCheck(state=DocumentCacheState.CORRUPT, reasons=[reason])

    @staticmethod
    def _valid() -> DocumentStageCheck:
        return DocumentStageCheck(state=DocumentCacheState.VALID)

    def validate_input(self, job_id: str, *, verify_image_sha: bool = True):
        if not self._workspace.document_input_manifest_path(job_id).exists():
            return self._missing("ARTIFACT_MISSING"), None
        try:
            manifest = self._repository.load_input_manifest(job_id)
            if manifest.version != DOCUMENT_INPUT_MANIFEST_VERSION or manifest.algorithm_version != DOCUMENT_INPUT_ALGORITHM_VERSION:
                return self._stale("ALGORITHM_VERSION_MISMATCH"), None
            config, expected = self._input.expected_fingerprint(job_id)
            if manifest.config_fingerprint != config or manifest.document_input_fingerprint != expected:
                return self._stale("DEPENDENCY_MISMATCH"), None
            selections = self._screenshots.load_final_selections(job_id)
            if manifest.source_final_selection_fingerprint != selections.artifact_fingerprint:
                return self._stale("DEPENDENCY_MISMATCH"), None
            if [x.candidate_id for x in manifest.items] != [x.candidate_id for x in selections.final_screenshots]:
                return self._stale("DEPENDENCY_MISMATCH"), None
            for item in manifest.items:
                path = resolve_document_artifact_path(self._workspace, job_id, item.image_relative_path)
                if path.stat().st_size != item.file_size_bytes:
                    return self._corrupt("IMAGE_REFERENCE_BROKEN"), None
                if verify_image_sha and file_sha256(path) != item.file_sha256:
                    return self._corrupt("IMAGE_REFERENCE_BROKEN"), None
            return self._valid(), manifest
        except Exception:
            return self._corrupt("INVALID_INPUT_MANIFEST"), None

    def validate_layout(self, job_id: str, input_manifest):
        if input_manifest is None:
            return self._stale("UPSTREAM_INVALID"), None
        if not self._workspace.document_layout_path(job_id).exists():
            return self._missing("ARTIFACT_MISSING"), None
        try:
            layout = self._repository.load_layout_manifest(job_id)
            if layout.version != DOCUMENT_LAYOUT_MANIFEST_VERSION or layout.algorithm_version != DOCUMENT_LAYOUT_ALGORITHM_VERSION:
                return self._stale("ALGORITHM_VERSION_MISMATCH"), None
            config, expected = self._layout.expected_fingerprint(input_manifest)
            if layout.document_input_fingerprint != input_manifest.document_input_fingerprint or layout.layout_config_fingerprint != config or layout.layout_fingerprint != expected:
                return self._stale("DEPENDENCY_MISMATCH"), None
            self._layout.validate(input_manifest, layout)
            return self._valid(), layout
        except Exception:
            return self._corrupt("INVALID_LAYOUT_MANIFEST"), None

    def validate_render(self, job_id: str, input_manifest, layout):
        if input_manifest is None or layout is None:
            return self._stale("UPSTREAM_INVALID"), None
        if not self._workspace.document_render_plan_path(job_id).exists():
            return self._missing("ARTIFACT_MISSING"), None
        try:
            render = self._repository.load_render_manifest(job_id)
            if render.version != DOCUMENT_RENDER_MANIFEST_VERSION or render.algorithm_version != DOCUMENT_RENDER_PLAN_ALGORITHM_VERSION:
                return self._stale("ALGORITHM_VERSION_MISMATCH"), None
            config, expected = self._render.expected_fingerprint(input_manifest, layout)
            if render.layout_fingerprint != layout.layout_fingerprint or render.render_config_fingerprint != config or render.render_plan_fingerprint != expected:
                return self._stale("DEPENDENCY_MISMATCH"), None
            self._render.validate(layout, render)
            return self._valid(), render
        except Exception:
            return self._corrupt("INVALID_RENDER_PLAN"), None

    def validate_pdf(self, job_id: str, input_manifest, render, *, deep: bool = True):
        if input_manifest is None or render is None:
            return self._stale("UPSTREAM_INVALID"), None
        if not self._workspace.document_pdf_manifest_path(job_id).exists() or not self._workspace.document_pdf_path(job_id).exists():
            return self._missing("ARTIFACT_MISSING"), None
        try:
            manifest = self._repository.load_pdf_manifest(job_id)
            if manifest.version != PDF_MANIFEST_VERSION or manifest.algorithm_version != PDF_RENDER_ALGORITHM_VERSION:
                return self._stale("ALGORITHM_VERSION_MISMATCH"), None
            config, expected = self._pdf.expected_fingerprint(input_manifest, render)
            if manifest.render_plan_fingerprint != render.render_plan_fingerprint or manifest.pdf_config_fingerprint != config or manifest.pdf_artifact_fingerprint != expected:
                return self._stale("DEPENDENCY_MISMATCH"), None
            self._pdf.validate_existing(job_id, manifest, render, deep=deep)
            return self._valid(), manifest
        except Exception:
            return self._corrupt("PDF_VALIDATION_FAILED"), None

    def inspect(self, job_id: str, *, deep_pdf: bool = True, deep_images: bool = True) -> Phase8CacheSnapshot:
        input_check, input_manifest = self.validate_input(job_id, verify_image_sha=deep_images)
        layout_check, layout = self.validate_layout(job_id, input_manifest)
        render_check, render = self.validate_render(job_id, input_manifest, layout)
        pdf_check, _ = self.validate_pdf(job_id, input_manifest, render, deep=deep_pdf)
        if not input_check.valid:
            resume = DocumentResumeStage.DOCUMENT_INPUT
        elif not layout_check.valid:
            resume = DocumentResumeStage.DOCUMENT_LAYOUT
        elif not render_check.valid:
            resume = DocumentResumeStage.DOCUMENT_RENDER_PLAN
        elif not pdf_check.valid:
            resume = DocumentResumeStage.PDF_GENERATION
        else:
            resume = DocumentResumeStage.DOCUMENT_READY
        order = [DocumentResumeStage.DOCUMENT_INPUT, DocumentResumeStage.DOCUMENT_LAYOUT, DocumentResumeStage.DOCUMENT_RENDER_PLAN, DocumentResumeStage.PDF_GENERATION]
        idx = order.index(resume) if resume in order else len(order)
        plan = Phase8ResumePlan(
            resume_stage=resume,
            rebuild_document_input=idx <= 0,
            rebuild_layout=idx <= 1,
            rebuild_render_plan=idx <= 2,
            rebuild_pdf=idx <= 3,
        )
        return Phase8CacheSnapshot(document_input=input_check, layout=layout_check, render_plan=render_check, pdf=pdf_check, resume_plan=plan)

    def repair_checkpoints(self, job_id: str, snapshot: Phase8CacheSnapshot) -> None:
        pairs = [
            (CP_DOCUMENT_INPUT_READY, snapshot.document_input.valid),
            (CP_DOCUMENT_LAYOUT_READY, snapshot.layout.valid),
            (CP_DOCUMENT_RENDER_PLAN_READY, snapshot.render_plan.valid),
            (CP_PDF_READY, snapshot.pdf.valid),
        ]
        upstream_ok = True
        for checkpoint, valid in pairs:
            stage_valid = upstream_ok and valid
            if stage_valid and not self._checkpoints.is_completed(job_id, checkpoint):
                self._checkpoints.mark_completed(job_id, checkpoint)
            if not stage_valid and self._checkpoints.is_completed(job_id, checkpoint):
                self._checkpoints.invalidate(job_id, checkpoint)
            upstream_ok = stage_valid
        if not upstream_ok and self._checkpoints.is_completed(job_id, CP_FINAL_DOCUMENT_READY):
            self._checkpoints.invalidate(job_id, CP_FINAL_DOCUMENT_READY)

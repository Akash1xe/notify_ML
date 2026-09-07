from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.core.config import AppSettings
from app.core.exceptions import PdfEmptyDocumentError, PdfGenerationError, PdfUnsupportedGlyphError, PdfValidationError
from app.document.models import (
    PDF_RENDER_ALGORITHM_VERSION,
    DocumentInputManifest,
    DocumentPdfManifest,
    DocumentPdfStats,
    DocumentRenderManifest,
    TextAlignment,
)
from app.document.repository import DocumentRepository
from app.document.utils import canonical_fingerprint, file_sha256, resolve_document_artifact_path
from app.jobs.checkpoints import CheckpointStore
from app.storage.workspace import WorkspaceManager

CP_PDF_READY = "PDF_READY"

_TEXT_HEAVY = {"CODE", "WHITEBOARD", "BLACKBOARD", "EQUATION", "DOCUMENT"}


class ReportLabPdfRenderer:
    """Backend adapter. Layout coordinates are top-left; ReportLab is bottom-left."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self.font_name = self._register_font()

    def _register_font(self) -> str:
        candidates = []
        if self._settings.pdf_unicode_font_path:
            candidates.append(Path(self._settings.pdf_unicode_font_path))
        candidates.extend([
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        ])
        for path in candidates:
            if path.exists() and path.is_file():
                try:
                    pdfmetrics.registerFont(TTFont("NotifyUnicode", str(path)))
                    return "NotifyUnicode"
                except Exception:
                    continue
        return "Helvetica"

    @staticmethod
    def backend_y(page_height: float, y_top: float, height: float) -> float:
        return page_height - y_top - height

    def _verify_text_supported(self, text: str) -> None:
        if self.font_name == "Helvetica" and any(ord(ch) > 255 for ch in text):
            raise PdfUnsupportedGlyphError("A Unicode-capable runtime font is required for this document text.")

    def _image_reader(self, path: Path, content_type: str):
        mode = self._settings.pdf_image_compression_mode
        use_jpeg = mode == "JPEG" or (mode == "AUTO" and content_type not in _TEXT_HEAVY)
        if not use_jpeg:
            return ImageReader(str(path))
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=self._settings.pdf_jpeg_quality, optimize=False)
                buffer.seek(0)
                return ImageReader(buffer)
        except Exception as exc:
            raise PdfGenerationError(f"Unable to prepare PDF image {path.name}.") from exc

    def _draw_text(self, pdf: canvas.Canvas, page_height: float, instruction) -> None:
        self._verify_text_supported(instruction.text)
        if not instruction.text:
            return
        size = instruction.font_size_points
        pdf.setFont(self.font_name, size)
        lines = instruction.text.splitlines() or [instruction.text]
        line_height = size * 1.2
        block_h = line_height * len(lines)
        if instruction.vertical_alignment.value == "TOP":
            top = instruction.y_points
        elif instruction.vertical_alignment.value == "BOTTOM":
            top = instruction.y_points + max(0.0, instruction.height_points - block_h)
        else:
            top = instruction.y_points + max(0.0, (instruction.height_points - block_h) / 2.0)
        baseline = page_height - top - size
        for line in lines:
            if instruction.alignment is TextAlignment.CENTER:
                pdf.drawCentredString(instruction.x_points + instruction.width_points / 2.0, baseline, line)
            elif instruction.alignment is TextAlignment.RIGHT:
                pdf.drawRightString(instruction.x_points + instruction.width_points, baseline, line)
            else:
                pdf.drawString(instruction.x_points, baseline, line)
            baseline -= line_height

    def render(self, path: Path, input_manifest: DocumentInputManifest, render_plan: DocumentRenderManifest, workspace: WorkspaceManager, job_id: str, cancel_check=None) -> None:
        if not render_plan.pages:
            raise PdfEmptyDocumentError("No useful visual screenshots were selected for the document.")
        path.parent.mkdir(parents=True, exist_ok=True)
        first = render_plan.pages[0]
        pdf = canvas.Canvas(str(path), pagesize=(first.page_width_points, first.page_height_points), pageCompression=1, invariant=1)
        pdf.setTitle(input_manifest.source.title or "Lecture Notes")
        pdf.setCreator("Notify")
        pdf.setProducer("Notify PDF Pipeline")
        try:
            for page in render_plan.pages:
                if cancel_check and cancel_check():
                    raise PdfGenerationError("PDF generation cancelled.")
                pdf.setPageSize((page.page_width_points, page.page_height_points))
                image_path = resolve_document_artifact_path(workspace, job_id, page.image.image_relative_path)
                if file_sha256(image_path) != page.image.file_sha256:
                    raise PdfGenerationError("PDF image SHA-256 no longer matches the render plan.")
                y = self.backend_y(page.page_height_points, page.image.y_points, page.image.height_points)
                reader = self._image_reader(image_path, page.content_type)
                pdf.drawImage(reader, page.image.x_points, y, page.image.width_points, page.image.height_points, preserveAspectRatio=True, anchor="c", mask="auto")
                for instruction in (page.header, page.caption, page.footer):
                    if instruction is not None:
                        self._draw_text(pdf, page.page_height_points, instruction)
                pdf.showPage()
            pdf.save()
        except Exception:
            raise


class DocumentPdfGenerator:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager, checkpoints: CheckpointStore, repository: DocumentRepository, renderer: ReportLabPdfRenderer | None = None) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = repository
        self._renderer = renderer or ReportLabPdfRenderer(settings)

    def config_fingerprint(self, title: str) -> str:
        s = self._settings
        return canonical_fingerprint({
            "algorithm_version": PDF_RENDER_ALGORITHM_VERSION,
            "compression_mode": s.pdf_image_compression_mode,
            "jpeg_quality": s.pdf_jpeg_quality,
            "unicode_font_path": str(s.pdf_unicode_font_path) if s.pdf_unicode_font_path else None,
            "title": title,
        })

    def expected_fingerprint(self, input_manifest: DocumentInputManifest, render_plan: DocumentRenderManifest) -> tuple[str, str]:
        config = self.config_fingerprint(input_manifest.source.title)
        return config, canonical_fingerprint({
            "render_plan_fingerprint": render_plan.render_plan_fingerprint,
            "pdf_config_fingerprint": config,
            "algorithm_version": PDF_RENDER_ALGORITHM_VERSION,
        })

    def _validate_pdf(self, path: Path, render_plan: DocumentRenderManifest) -> None:
        try:
            if not path.exists() or path.stat().st_size <= 4:
                raise PdfValidationError("Generated PDF signature is invalid.")
            with path.open("rb") as fh:
                if not fh.read(5).startswith(b"%PDF-"):
                    raise PdfValidationError("Generated PDF signature is invalid.")
            reader = PdfReader(str(path))
            if len(reader.pages) != len(render_plan.pages):
                raise PdfValidationError("Generated PDF page count does not match the render plan.")
            eps = self._settings.pdf_page_dimension_epsilon_points
            for actual, planned in zip(reader.pages, render_plan.pages, strict=True):
                width = float(actual.mediabox.width)
                height = float(actual.mediabox.height)
                if abs(width - planned.page_width_points) > eps or abs(height - planned.page_height_points) > eps:
                    raise PdfValidationError("Generated PDF page dimensions do not match the render plan.")
        except PdfValidationError:
            raise
        except Exception as exc:
            raise PdfValidationError("Generated PDF could not be parsed.") from exc

    def generate(self, job_id: str, input_manifest: DocumentInputManifest | None = None, render_plan: DocumentRenderManifest | None = None, cancel_check=None) -> DocumentPdfManifest:
        if not self._checkpoints.is_completed(job_id, "DOCUMENT_RENDER_PLAN_READY"):
            raise PdfGenerationError("DOCUMENT_RENDER_PLAN_READY is required before PDF generation.")
        input_manifest = input_manifest or self._repository.load_input_manifest(job_id)
        render_plan = render_plan or self._repository.load_render_manifest(job_id)
        if not render_plan.pages:
            raise PdfEmptyDocumentError("No useful visual screenshots were selected for the document.")
        temp = self._workspace.document_temp_pdf_path(job_id)
        final = self._workspace.document_pdf_path(job_id)
        temp.unlink(missing_ok=True)
        try:
            self._renderer.render(temp, input_manifest, render_plan, self._workspace, job_id, cancel_check=cancel_check)
            self._validate_pdf(temp, render_plan)
            sha = file_sha256(temp)
            file_size = temp.stat().st_size
            config_fp, artifact_fp = self.expected_fingerprint(input_manifest, render_plan)
            warnings = []
            if file_size > self._settings.pdf_large_file_warning_mb * 1024 * 1024:
                warnings.append("PDF_FILE_SIZE_HIGH")
            stats = DocumentPdfStats(
                page_count=len(render_plan.pages),
                file_size_bytes=file_size,
                portrait_page_count=sum(p.orientation.value == "PORTRAIT" for p in render_plan.pages),
                landscape_page_count=sum(p.orientation.value == "LANDSCAPE" for p in render_plan.pages),
                image_count=len(render_plan.pages),
                text_instruction_count=sum(sum(x is not None for x in (p.header, p.caption, p.footer)) for p in render_plan.pages),
                mean_pdf_bytes_per_page=file_size / len(render_plan.pages),
                warnings=warnings,
            )
            manifest = DocumentPdfManifest(
                render_plan_fingerprint=render_plan.render_plan_fingerprint,
                pdf_config_fingerprint=config_fp,
                pdf_artifact_fingerprint=artifact_fp,
                file_sha256=sha,
                file_size_bytes=file_size,
                page_count=len(render_plan.pages),
                title=input_manifest.source.title or "Lecture Notes",
                stats=stats,
            )
            backup = final.with_suffix(".previous.pdf")
            backup.unlink(missing_ok=True)
            had_previous = final.exists()
            if had_previous:
                os.replace(final, backup)
            try:
                os.replace(temp, final)
                self._repository.save_pdf_manifest(job_id, manifest)
            except Exception:
                final.unlink(missing_ok=True)
                if had_previous and backup.exists():
                    os.replace(backup, final)
                raise
            backup.unlink(missing_ok=True)
            self._checkpoints.mark_completed(job_id, CP_PDF_READY)
            return manifest
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def validate_existing(self, job_id: str, manifest: DocumentPdfManifest, render_plan: DocumentRenderManifest, *, deep: bool = True) -> None:
        path = self._workspace.document_pdf_path(job_id)
        if not path.exists() or not path.is_file() or path.stat().st_size != manifest.file_size_bytes:
            raise PdfValidationError("Stored PDF is missing or has the wrong size.")
        with path.open("rb") as fh:
            if not fh.read(5).startswith(b"%PDF-"):
                raise PdfValidationError("Stored PDF signature is invalid.")
        if deep:
            if file_sha256(path) != manifest.file_sha256:
                raise PdfValidationError("Stored PDF SHA-256 does not match its manifest.")
            self._validate_pdf(path, render_plan)

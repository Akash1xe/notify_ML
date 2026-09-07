from __future__ import annotations

from app.core.config import AppSettings
from app.core.exceptions import DocumentRenderPlanError
from app.document.models import (
    DOCUMENT_RENDER_PLAN_ALGORITHM_VERSION,
    DocumentInputManifest,
    DocumentLayoutManifest,
    DocumentPageRenderPlan,
    DocumentRenderManifest,
    DocumentRenderStats,
    ImageRenderInstruction,
    TextAlignment,
    TextRenderInstruction,
    TextRole,
    VerticalAlignment,
)
from app.document.repository import DocumentRepository
from app.document.utils import canonical_fingerprint, content_type_display_name, normalize_plain_text, truncate_words, wrap_approx
from app.jobs.checkpoints import CheckpointStore

CP_DOCUMENT_RENDER_PLAN_READY = "DOCUMENT_RENDER_PLAN_READY"


class DocumentRenderPlanner:
    def __init__(self, settings: AppSettings, checkpoints: CheckpointStore, repository: DocumentRepository) -> None:
        self._settings = settings
        self._checkpoints = checkpoints
        self._repository = repository

    def config_fingerprint(self) -> str:
        s = self._settings
        return canonical_fingerprint({
            "algorithm_version": DOCUMENT_RENDER_PLAN_ALGORITHM_VERSION,
            "show_timestamp": s.pdf_show_timestamp,
            "show_content_type": s.pdf_show_content_type,
            "caption_limits": [s.pdf_caption_max_characters, s.pdf_caption_max_words, s.pdf_caption_max_lines],
            "page_number_style": s.pdf_page_number_style,
            "header_enabled": s.pdf_header_enabled,
            "header_max_characters": s.pdf_header_max_characters,
            "font_sizes": [s.pdf_header_font_size_pt, s.pdf_caption_font_size_pt, s.pdf_footer_font_size_pt],
        })

    def expected_fingerprint(self, input_manifest: DocumentInputManifest, layout: DocumentLayoutManifest) -> tuple[str, str]:
        config = self.config_fingerprint()
        return config, canonical_fingerprint({
            "document_input_fingerprint": input_manifest.document_input_fingerprint,
            "layout_fingerprint": layout.layout_fingerprint,
            "render_config_fingerprint": config,
            "algorithm_version": DOCUMENT_RENDER_PLAN_ALGORITHM_VERSION,
        })

    def _caption(self, item) -> tuple[str, bool, bool, bool]:
        parts: list[str] = []
        has_timestamp = bool(self._settings.pdf_show_timestamp and item.timestamp_display)
        has_type = bool(self._settings.pdf_show_content_type)
        if has_timestamp:
            parts.append(item.timestamp_display)
        if has_type:
            parts.append(content_type_display_name(item.content_type))
        head = " · ".join(parts)
        body = normalize_plain_text(item.context_snippet or "")
        text = head
        if body:
            text = f"{head} — {body}" if head else body
        text, truncated = truncate_words(text, max_characters=self._settings.pdf_caption_max_characters, max_words=self._settings.pdf_caption_max_words)
        return text, truncated, has_timestamp, has_type

    def build(self, job_id: str, input_manifest: DocumentInputManifest | None = None, layout: DocumentLayoutManifest | None = None) -> DocumentRenderManifest:
        if not self._checkpoints.is_completed(job_id, "DOCUMENT_LAYOUT_READY"):
            raise DocumentRenderPlanError("DOCUMENT_LAYOUT_READY is required before render planning.")
        input_manifest = input_manifest or self._repository.load_input_manifest(job_id)
        layout = layout or self._repository.load_layout_manifest(job_id)
        if [p.candidate_id for p in layout.pages] != [x.candidate_id for x in input_manifest.items]:
            raise DocumentRenderPlanError("Layout does not match document input.")
        by_id = {x.candidate_id: x for x in input_manifest.items}
        pages: list[DocumentPageRenderPlan] = []
        caption_count = timestamp_count = type_count = header_count = footer_count = truncated_captions = truncated_headers = 0
        for page in layout.pages:
            item = by_id[page.candidate_id]
            image = ImageRenderInstruction(
                candidate_id=item.candidate_id,
                image_relative_path=item.image_relative_path,
                file_sha256=item.file_sha256,
                x_points=page.screenshot.x_points,
                y_points=page.screenshot.y_points,
                width_points=page.screenshot.width_points,
                height_points=page.screenshot.height_points,
            )
            caption = None
            caption_text, caption_truncated, has_ts, has_type = self._caption(item)
            if page.caption_bounds is not None and caption_text:
                wrapped, wrapped_truncated = wrap_approx(caption_text, width_points=page.caption_bounds.width_points, font_size=self._settings.pdf_caption_font_size_pt, max_lines=self._settings.pdf_caption_max_lines)
                caption = TextRenderInstruction(
                    role=TextRole.CAPTION, text=wrapped,
                    x_points=page.caption_bounds.x_points, y_points=page.caption_bounds.y_points,
                    width_points=page.caption_bounds.width_points, height_points=page.caption_bounds.height_points,
                    font_role="CAPTION", font_size_points=self._settings.pdf_caption_font_size_pt,
                    alignment=TextAlignment.LEFT, vertical_alignment=VerticalAlignment.CENTER,
                    max_lines=self._settings.pdf_caption_max_lines,
                )
                caption_count += 1
                timestamp_count += int(has_ts)
                type_count += int(has_type)
                truncated_captions += int(caption_truncated or wrapped_truncated)
            header = None
            if page.header_bounds is not None and self._settings.pdf_header_enabled:
                raw = normalize_plain_text(input_manifest.source.title)[:self._settings.pdf_header_max_characters]
                truncated = len(normalize_plain_text(input_manifest.source.title)) > len(raw)
                wrapped, wrapped_truncated = wrap_approx(raw, width_points=page.header_bounds.width_points, font_size=self._settings.pdf_header_font_size_pt, max_lines=1)
                header = TextRenderInstruction(
                    role=TextRole.HEADER, text=wrapped,
                    x_points=page.header_bounds.x_points, y_points=page.header_bounds.y_points,
                    width_points=page.header_bounds.width_points, height_points=page.header_bounds.height_points,
                    font_role="HEADER", font_size_points=self._settings.pdf_header_font_size_pt,
                    alignment=TextAlignment.CENTER, vertical_alignment=VerticalAlignment.CENTER,
                )
                header_count += 1
                truncated_headers += int(truncated or wrapped_truncated)
            footer = None
            if page.footer_bounds is not None and self._settings.pdf_footer_enabled:
                text = str(page.page_number) if self._settings.pdf_page_number_style == "SIMPLE" else f"{page.page_number} / {len(layout.pages)}"
                footer = TextRenderInstruction(
                    role=TextRole.FOOTER, text=text,
                    x_points=page.footer_bounds.x_points, y_points=page.footer_bounds.y_points,
                    width_points=page.footer_bounds.width_points, height_points=page.footer_bounds.height_points,
                    font_role="FOOTER", font_size_points=self._settings.pdf_footer_font_size_pt,
                    alignment=TextAlignment.CENTER, vertical_alignment=VerticalAlignment.CENTER,
                )
                footer_count += 1
            pages.append(DocumentPageRenderPlan(
                page_number=page.page_number,
                candidate_id=item.candidate_id,
                page_width_points=page.width_points,
                page_height_points=page.height_points,
                orientation=page.orientation,
                content_type=item.content_type,
                image=image,
                header=header,
                caption=caption,
                footer=footer,
            ))
        config_fp, render_fp = self.expected_fingerprint(input_manifest, layout)
        warnings = []
        if caption_count and truncated_captions / caption_count > 0.25:
            warnings.append("HIGH_CAPTION_TRUNCATION_RATE")
        output = DocumentRenderManifest(
            document_input_fingerprint=input_manifest.document_input_fingerprint,
            layout_fingerprint=layout.layout_fingerprint,
            render_config_fingerprint=config_fp,
            render_plan_fingerprint=render_fp,
            stats=DocumentRenderStats(
                page_count=len(pages), caption_page_count=caption_count, timestamp_page_count=timestamp_count,
                content_type_label_count=type_count, header_page_count=header_count, footer_page_count=footer_count,
                truncated_caption_count=truncated_captions, truncated_header_count=truncated_headers,
                warnings=warnings,
            ),
            pages=pages,
        )
        self.validate(layout, output)
        self._repository.save_render_manifest(job_id, output)
        self._checkpoints.mark_completed(job_id, CP_DOCUMENT_RENDER_PLAN_READY)
        return output

    @staticmethod
    def validate(layout: DocumentLayoutManifest, render: DocumentRenderManifest) -> None:
        if len(layout.pages) != len(render.pages):
            raise DocumentRenderPlanError("Render-plan page count does not match layout.")
        for planned, rendered in zip(layout.pages, render.pages, strict=True):
            if planned.page_number != rendered.page_number or planned.candidate_id != rendered.candidate_id:
                raise DocumentRenderPlanError("Render-plan page identity does not match layout.")
            shot = planned.screenshot
            image = rendered.image
            if (shot.x_points, shot.y_points, shot.width_points, shot.height_points) != (image.x_points, image.y_points, image.width_points, image.height_points):
                raise DocumentRenderPlanError("Render plan recalculated screenshot geometry.")

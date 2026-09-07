from __future__ import annotations

import statistics

from app.core.config import AppSettings
from app.core.exceptions import DocumentLayoutError
from app.document.models import (
    DOCUMENT_LAYOUT_ALGORITHM_VERSION,
    DocumentInputManifest,
    DocumentLayoutManifest,
    DocumentLayoutStats,
    DocumentOrientation,
    DocumentPageLayout,
    DocumentRect,
    LayoutStrategy,
    PdfPageOrientation,
    PdfPageSize,
    ScreenshotPlacement,
)
from app.document.repository import DocumentRepository
from app.document.utils import canonical_fingerprint, mm_to_points
from app.jobs.checkpoints import CheckpointStore

CP_DOCUMENT_LAYOUT_READY = "DOCUMENT_LAYOUT_READY"


class DocumentLayoutEngine:
    def __init__(self, settings: AppSettings, checkpoints: CheckpointStore, repository: DocumentRepository) -> None:
        self._settings = settings
        self._checkpoints = checkpoints
        self._repository = repository

    def config_fingerprint(self) -> str:
        s = self._settings
        return canonical_fingerprint({
            "algorithm_version": DOCUMENT_LAYOUT_ALGORITHM_VERSION,
            "page_size": s.pdf_page_size,
            "page_orientation": s.pdf_page_orientation,
            "square_page_orientation": s.pdf_square_page_orientation,
            "margins_mm": [s.pdf_margin_top_mm, s.pdf_margin_right_mm, s.pdf_margin_bottom_mm, s.pdf_margin_left_mm],
            "header": [s.pdf_header_enabled, s.pdf_header_height_mm],
            "footer": [s.pdf_footer_enabled, s.pdf_footer_height_mm],
            "caption": [s.pdf_caption_enabled, s.pdf_caption_height_mm],
            "allow_upscale": s.pdf_allow_image_upscale,
            "min_size_mm": [s.pdf_min_screenshot_width_mm, s.pdf_min_screenshot_height_mm],
        })

    @staticmethod
    def _base_page_mm(size: str) -> tuple[float, float]:
        return (210.0, 297.0) if size == "A4" else (215.9, 279.4)

    def _orientation_for(self, item) -> PdfPageOrientation:
        forced = self._settings.pdf_page_orientation
        if forced != "AUTO":
            return PdfPageOrientation(forced)
        if item.orientation is DocumentOrientation.LANDSCAPE:
            return PdfPageOrientation.LANDSCAPE
        if item.orientation is DocumentOrientation.PORTRAIT:
            return PdfPageOrientation.PORTRAIT
        return PdfPageOrientation(self._settings.pdf_square_page_orientation)

    def _page_points(self, orientation: PdfPageOrientation) -> tuple[float, float]:
        w, h = self._base_page_mm(self._settings.pdf_page_size)
        if orientation is PdfPageOrientation.LANDSCAPE:
            w, h = h, w
        return mm_to_points(w), mm_to_points(h)

    def expected_fingerprint(self, input_manifest: DocumentInputManifest) -> tuple[str, str]:
        config = self.config_fingerprint()
        fingerprint = canonical_fingerprint({
            "document_input_fingerprint": input_manifest.document_input_fingerprint,
            "layout_config_fingerprint": config,
            "algorithm_version": DOCUMENT_LAYOUT_ALGORITHM_VERSION,
        })
        return config, fingerprint

    def build(self, job_id: str, input_manifest: DocumentInputManifest | None = None) -> DocumentLayoutManifest:
        if not self._checkpoints.is_completed(job_id, "DOCUMENT_INPUT_READY"):
            raise DocumentLayoutError("DOCUMENT_INPUT_READY is required before page layout.")
        manifest = input_manifest or self._repository.load_input_manifest(job_id)
        s = self._settings
        pages: list[DocumentPageLayout] = []
        utilizations: list[float] = []
        scales: list[float] = []
        warnings: list[str] = []

        for item in manifest.items:
            orientation = self._orientation_for(item)
            page_w, page_h = self._page_points(orientation)
            left = mm_to_points(s.pdf_margin_left_mm)
            right = mm_to_points(s.pdf_margin_right_mm)
            top = mm_to_points(s.pdf_margin_top_mm)
            bottom = mm_to_points(s.pdf_margin_bottom_mm)
            header_h = mm_to_points(s.pdf_header_height_mm) if s.pdf_header_enabled else 0.0
            footer_h = mm_to_points(s.pdf_footer_height_mm) if s.pdf_footer_enabled else 0.0
            caption_h = mm_to_points(s.pdf_caption_height_mm) if s.pdf_caption_enabled else 0.0
            usable_w = page_w - left - right
            usable_h = page_h - top - bottom - header_h - footer_h - caption_h
            if usable_w <= 0 or usable_h <= 0:
                raise DocumentLayoutError("PDF configuration leaves no usable screenshot region.")

            image_y = top + header_h
            scale = min(usable_w / item.width, usable_h / item.height)
            if not s.pdf_allow_image_upscale:
                scale = min(scale, 1.0)
            render_w = item.width * scale
            render_h = item.height * scale
            if render_w + 1e-6 < mm_to_points(s.pdf_min_screenshot_width_mm) or render_h + 1e-6 < mm_to_points(s.pdf_min_screenshot_height_mm):
                warnings.append("SCREENSHOT_RENDER_SIZE_TOO_SMALL")
            x = left + (usable_w - render_w) / 2.0
            y = image_y + (usable_h - render_h) / 2.0
            content = DocumentRect(x_points=left, y_points=image_y, width_points=usable_w, height_points=usable_h)
            screenshot = ScreenshotPlacement(
                candidate_id=item.candidate_id,
                x_points=x,
                y_points=y,
                width_points=render_w,
                height_points=render_h,
                source_width=item.width,
                source_height=item.height,
                scale=scale,
            )
            header_bounds = DocumentRect(x_points=left, y_points=top, width_points=usable_w, height_points=header_h) if header_h else None
            caption_y = image_y + usable_h
            caption_bounds = DocumentRect(x_points=left, y_points=caption_y, width_points=usable_w, height_points=caption_h) if caption_h else None
            footer_y = page_h - bottom - footer_h
            footer_bounds = DocumentRect(x_points=left, y_points=footer_y, width_points=usable_w, height_points=footer_h) if footer_h else None
            page = DocumentPageLayout(
                page_number=item.order,
                document_input_order=item.order,
                candidate_id=item.candidate_id,
                page_size=PdfPageSize(s.pdf_page_size),
                orientation=orientation,
                width_points=page_w,
                height_points=page_h,
                content_bounds=content,
                screenshot=screenshot,
                header_bounds=header_bounds,
                caption_bounds=caption_bounds,
                footer_bounds=footer_bounds,
            )
            pages.append(page)
            scales.append(scale)
            utilizations.append((render_w * render_h) / (usable_w * usable_h))

        if pages and statistics.mean(utilizations) < 0.45:
            warnings.append("LOW_PAGE_UTILIZATION")
        config_fp, layout_fp = self.expected_fingerprint(manifest)
        stats = DocumentLayoutStats(
            page_count=len(pages),
            portrait_page_count=sum(p.orientation is PdfPageOrientation.PORTRAIT for p in pages),
            landscape_page_count=sum(p.orientation is PdfPageOrientation.LANDSCAPE for p in pages),
            mean_image_scale=statistics.mean(scales) if scales else None,
            min_image_scale=min(scales) if scales else None,
            max_image_scale=max(scales) if scales else None,
            mean_page_utilization=statistics.mean(utilizations) if utilizations else None,
            median_page_utilization=statistics.median(utilizations) if utilizations else None,
            warnings=list(dict.fromkeys(warnings)),
        )
        output = DocumentLayoutManifest(
            document_input_fingerprint=manifest.document_input_fingerprint,
            layout_config_fingerprint=config_fp,
            layout_fingerprint=layout_fp,
            layout_strategy=LayoutStrategy.SINGLE_SCREENSHOT,
            stats=stats,
            pages=pages,
        )
        self.validate(manifest, output)
        self._repository.save_layout_manifest(job_id, output)
        self._checkpoints.mark_completed(job_id, CP_DOCUMENT_LAYOUT_READY)
        return output

    def validate(self, input_manifest: DocumentInputManifest, layout: DocumentLayoutManifest) -> None:
        if len(layout.pages) != len(input_manifest.items):
            raise DocumentLayoutError("Layout page count does not match document input.")
        if [p.page_number for p in layout.pages] != list(range(1, len(layout.pages) + 1)):
            raise DocumentLayoutError("Layout page numbers are not contiguous.")
        if [p.candidate_id for p in layout.pages] != [x.candidate_id for x in input_manifest.items]:
            raise DocumentLayoutError("Layout candidate order does not match document input.")
        eps = self._settings.pdf_layout_epsilon_points
        for page in layout.pages:
            shot = page.screenshot
            bounds = page.content_bounds
            if shot.x_points + eps < bounds.x_points or shot.y_points + eps < bounds.y_points:
                raise DocumentLayoutError("Screenshot placement escaped content bounds.")
            if shot.x_points + shot.width_points > bounds.x_points + bounds.width_points + eps:
                raise DocumentLayoutError("Screenshot placement exceeded content width.")
            if shot.y_points + shot.height_points > bounds.y_points + bounds.height_points + eps:
                raise DocumentLayoutError("Screenshot placement exceeded content height.")
            source_ratio = shot.source_width / shot.source_height
            render_ratio = shot.width_points / shot.height_points
            if abs(source_ratio - render_ratio) > 1e-6:
                raise DocumentLayoutError("Screenshot aspect ratio was not preserved.")
            if not self._settings.pdf_allow_image_upscale and shot.scale > 1.0 + eps:
                raise DocumentLayoutError("Screenshot was upscaled despite configuration.")

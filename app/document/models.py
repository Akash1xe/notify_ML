from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

DOCUMENT_INPUT_MANIFEST_VERSION = "1"
DOCUMENT_INPUT_ALGORITHM_VERSION = "1"
DOCUMENT_LAYOUT_MANIFEST_VERSION = "1"
DOCUMENT_LAYOUT_ALGORITHM_VERSION = "1"
DOCUMENT_RENDER_MANIFEST_VERSION = "1"
DOCUMENT_RENDER_PLAN_ALGORITHM_VERSION = "1"
PDF_MANIFEST_VERSION = "1"
PDF_RENDER_ALGORITHM_VERSION = "1"


class DocumentOrientation(str, Enum):
    LANDSCAPE = "LANDSCAPE"
    PORTRAIT = "PORTRAIT"
    SQUARE = "SQUARE"


class PdfPageSize(str, Enum):
    A4 = "A4"
    LETTER = "LETTER"


class PdfPageOrientation(str, Enum):
    PORTRAIT = "PORTRAIT"
    LANDSCAPE = "LANDSCAPE"


class LayoutStrategy(str, Enum):
    SINGLE_SCREENSHOT = "SINGLE_SCREENSHOT"


class ImageFit(str, Enum):
    CONTAIN = "CONTAIN"


class TextAlignment(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class VerticalAlignment(str, Enum):
    TOP = "TOP"
    CENTER = "CENTER"
    BOTTOM = "BOTTOM"


class TextRole(str, Enum):
    HEADER = "HEADER"
    CAPTION = "CAPTION"
    FOOTER = "FOOTER"


class DocumentCacheState(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    STALE = "STALE"
    CORRUPT = "CORRUPT"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class DocumentResumeStage(str, Enum):
    DOCUMENT_INPUT = "DOCUMENT_INPUT"
    DOCUMENT_LAYOUT = "DOCUMENT_LAYOUT"
    DOCUMENT_RENDER_PLAN = "DOCUMENT_RENDER_PLAN"
    PDF_GENERATION = "PDF_GENERATION"
    DOCUMENT_READY = "DOCUMENT_READY"


class DocumentApiStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    STALE = "STALE"
    CORRUPT = "CORRUPT"
    FAILED = "FAILED"
    EMPTY = "EMPTY"


class DocumentSourceMetadata(BaseModel):
    title: str = "Lecture Notes"
    url: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    video_width: int | None = Field(default=None, ge=1)
    video_height: int | None = Field(default=None, ge=1)


class DocumentInputItem(BaseModel):
    order: int = Field(ge=1)
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    duplicate_group_id: str
    timestamp_seconds: float = Field(ge=0)
    timestamp_display: str | None = None
    image_relative_path: str
    file_sha256: str
    file_size_bytes: int = Field(ge=1)
    image_format: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    aspect_ratio: float = Field(gt=0)
    orientation: DocumentOrientation
    content_type: str
    semantic_decision_score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    final_selection_score: float = Field(ge=0, le=1)
    context_snippet: str | None = None


class DocumentInputStats(BaseModel):
    item_count: int = Field(ge=0)
    total_image_bytes: int = Field(ge=0)
    landscape_count: int = Field(ge=0)
    portrait_count: int = Field(ge=0)
    square_count: int = Field(ge=0)
    first_timestamp_seconds: float | None = Field(default=None, ge=0)
    last_timestamp_seconds: float | None = Field(default=None, ge=0)
    content_type_counts: dict[str, int] = Field(default_factory=dict)
    min_width: int | None = Field(default=None, ge=1)
    max_width: int | None = Field(default=None, ge=1)
    min_height: int | None = Field(default=None, ge=1)
    max_height: int | None = Field(default=None, ge=1)
    unique_resolution_count: int = Field(default=0, ge=0)
    mean_quality_score: float | None = Field(default=None, ge=0, le=1)
    median_quality_score: float | None = Field(default=None, ge=0, le=1)
    mean_semantic_score: float | None = Field(default=None, ge=0, le=1)


class DocumentInputManifest(BaseModel):
    version: str = DOCUMENT_INPUT_MANIFEST_VERSION
    algorithm_version: str = DOCUMENT_INPUT_ALGORITHM_VERSION
    job_id: str
    source: DocumentSourceMetadata
    source_final_selection_fingerprint: str
    config_fingerprint: str
    document_input_fingerprint: str
    stats: DocumentInputStats
    items: list[DocumentInputItem]


class DocumentRect(BaseModel):
    x_points: float = Field(ge=0)
    y_points: float = Field(ge=0)
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)


class ScreenshotPlacement(BaseModel):
    candidate_id: int = Field(ge=1)
    x_points: float = Field(ge=0)
    y_points: float = Field(ge=0)
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    source_width: int = Field(ge=1)
    source_height: int = Field(ge=1)
    scale: float = Field(gt=0)
    fit_mode: ImageFit = ImageFit.CONTAIN


class DocumentPageLayout(BaseModel):
    page_number: int = Field(ge=1)
    document_input_order: int = Field(ge=1)
    candidate_id: int = Field(ge=1)
    page_size: PdfPageSize
    orientation: PdfPageOrientation
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    content_bounds: DocumentRect
    screenshot: ScreenshotPlacement
    header_bounds: DocumentRect | None = None
    caption_bounds: DocumentRect | None = None
    footer_bounds: DocumentRect | None = None


class DocumentLayoutStats(BaseModel):
    page_count: int = Field(ge=0)
    portrait_page_count: int = Field(ge=0)
    landscape_page_count: int = Field(ge=0)
    mean_image_scale: float | None = Field(default=None, gt=0)
    min_image_scale: float | None = Field(default=None, gt=0)
    max_image_scale: float | None = Field(default=None, gt=0)
    mean_page_utilization: float | None = Field(default=None, ge=0, le=1)
    median_page_utilization: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class DocumentLayoutManifest(BaseModel):
    version: str = DOCUMENT_LAYOUT_MANIFEST_VERSION
    algorithm_version: str = DOCUMENT_LAYOUT_ALGORITHM_VERSION
    document_input_fingerprint: str
    layout_config_fingerprint: str
    layout_fingerprint: str
    layout_strategy: LayoutStrategy = LayoutStrategy.SINGLE_SCREENSHOT
    stats: DocumentLayoutStats
    pages: list[DocumentPageLayout]


class ImageRenderInstruction(BaseModel):
    candidate_id: int = Field(ge=1)
    image_relative_path: str
    file_sha256: str
    x_points: float = Field(ge=0)
    y_points: float = Field(ge=0)
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    fit_mode: ImageFit = ImageFit.CONTAIN
    preserve_aspect_ratio: bool = True


class TextRenderInstruction(BaseModel):
    role: TextRole
    text: str
    x_points: float = Field(ge=0)
    y_points: float = Field(ge=0)
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    font_role: str
    font_size_points: float = Field(gt=0)
    alignment: TextAlignment
    vertical_alignment: VerticalAlignment = VerticalAlignment.CENTER
    max_lines: int = Field(default=1, ge=1)


class DocumentPageRenderPlan(BaseModel):
    page_number: int = Field(ge=1)
    candidate_id: int = Field(ge=1)
    page_width_points: float = Field(gt=0)
    page_height_points: float = Field(gt=0)
    orientation: PdfPageOrientation
    content_type: str
    image: ImageRenderInstruction
    header: TextRenderInstruction | None = None
    caption: TextRenderInstruction | None = None
    footer: TextRenderInstruction | None = None


class DocumentRenderStats(BaseModel):
    page_count: int = Field(ge=0)
    caption_page_count: int = Field(ge=0)
    timestamp_page_count: int = Field(ge=0)
    content_type_label_count: int = Field(ge=0)
    header_page_count: int = Field(ge=0)
    footer_page_count: int = Field(ge=0)
    truncated_caption_count: int = Field(ge=0)
    truncated_header_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class DocumentRenderManifest(BaseModel):
    version: str = DOCUMENT_RENDER_MANIFEST_VERSION
    algorithm_version: str = DOCUMENT_RENDER_PLAN_ALGORITHM_VERSION
    document_input_fingerprint: str
    layout_fingerprint: str
    render_config_fingerprint: str
    render_plan_fingerprint: str
    stats: DocumentRenderStats
    pages: list[DocumentPageRenderPlan]


class DocumentPdfStats(BaseModel):
    page_count: int = Field(ge=0)
    file_size_bytes: int = Field(ge=0)
    portrait_page_count: int = Field(ge=0)
    landscape_page_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    text_instruction_count: int = Field(ge=0)
    mean_pdf_bytes_per_page: float | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class DocumentPdfManifest(BaseModel):
    version: str = PDF_MANIFEST_VERSION
    algorithm_version: str = PDF_RENDER_ALGORITHM_VERSION
    render_plan_fingerprint: str
    pdf_config_fingerprint: str
    pdf_artifact_fingerprint: str
    relative_path: str = "document/final.pdf"
    file_sha256: str
    file_size_bytes: int = Field(gt=0)
    page_count: int = Field(ge=1)
    title: str
    stats: DocumentPdfStats


class DocumentStageCheck(BaseModel):
    state: DocumentCacheState
    reasons: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.state is DocumentCacheState.VALID


class Phase8ResumePlan(BaseModel):
    resume_stage: DocumentResumeStage
    rebuild_document_input: bool = False
    rebuild_layout: bool = False
    rebuild_render_plan: bool = False
    rebuild_pdf: bool = False
    cleanup_temp_artifacts: bool = False


class Phase8CacheSnapshot(BaseModel):
    document_input: DocumentStageCheck
    layout: DocumentStageCheck
    render_plan: DocumentStageCheck
    pdf: DocumentStageCheck
    resume_plan: Phase8ResumePlan


class FinalDocumentSummary(BaseModel):
    document_ready: bool
    document_title: str
    page_count: int = Field(ge=0)
    final_screenshot_count: int = Field(ge=0)
    suppressed_duplicate_count: int = Field(ge=0)
    file_size_bytes: int = Field(ge=0)
    first_timestamp_seconds: float | None = Field(default=None, ge=0)
    last_timestamp_seconds: float | None = Field(default=None, ge=0)
    content_type_counts: dict[str, int] = Field(default_factory=dict)
    cache_hits: dict[str, bool] = Field(default_factory=dict)


class DocumentStatusResponse(BaseModel):
    job_id: str
    status: DocumentApiStatus
    checkpoint: str | None = None
    document_title: str = "Lecture Notes"
    page_count: int = Field(default=0, ge=0)
    file_size_bytes: int = Field(default=0, ge=0)
    final_screenshot_count: int = Field(default=0, ge=0)
    download_available: bool = False
    error_code: str | None = None


class DocumentSummaryResponse(BaseModel):
    document_title: str
    page_count: int = Field(ge=0)
    final_screenshot_count: int = Field(ge=0)
    suppressed_duplicate_count: int = Field(ge=0)
    file_size_bytes: int = Field(ge=0)
    first_timestamp_seconds: float | None = Field(default=None, ge=0)
    last_timestamp_seconds: float | None = Field(default=None, ge=0)
    content_type_counts: dict[str, int] = Field(default_factory=dict)


class DocumentScreenshotResponse(BaseModel):
    order: int = Field(ge=1)
    candidate_id: int = Field(ge=1)
    timestamp_seconds: float = Field(ge=0)
    timestamp_display: str | None = None
    content_type: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    quality_score: float = Field(ge=0, le=1)
    semantic_score: float = Field(ge=0, le=1)
    preview_available: bool = True


class DocumentScreenshotListResponse(BaseModel):
    items: list[DocumentScreenshotResponse]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)


class DocumentDownloadDescriptor(BaseModel):
    path: str
    filename: str
    content_type: Literal["application/pdf"] = "application/pdf"
    file_size_bytes: int = Field(gt=0)
    sha256: str
    etag: str

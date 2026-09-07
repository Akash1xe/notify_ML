from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

STRESS_REPORT_VERSION = "1"
STRESS_ALGORITHM_VERSION = "1"


class LimitType(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"
    WARNING_ONLY = "WARNING_ONLY"


class StressStatus(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    FAILED = "FAILED"


class ScalingClassification(str, Enum):
    SUBLINEAR = "SUBLINEAR"
    APPROX_LINEAR = "APPROX_LINEAR"
    SUPERLINEAR = "SUPERLINEAR"
    EXPLOSIVE = "EXPLOSIVE"
    UNKNOWN = "UNKNOWN"


class ResourceLimits(BaseModel):
    max_video_duration_seconds: float | None = Field(default=3 * 3600, gt=0)
    max_source_file_bytes: int | None = Field(default=4 * 1024**3, gt=0)
    max_workspace_bytes: int | None = Field(default=20 * 1024**3, gt=0)
    max_sampled_frames: int | None = Field(default=20_000, gt=0)
    max_candidates: int | None = Field(default=1000, gt=0)
    max_semantic_candidates: int | None = Field(default=1000, gt=0)
    max_final_screenshots: int | None = Field(default=1000, gt=0)
    max_transcript_segments: int | None = Field(default=100_000, gt=0)
    max_pdf_pages: int | None = Field(default=1000, gt=0)
    max_pdf_bytes: int | None = Field(default=500 * 1024**2, gt=0)
    max_peak_rss_bytes: int | None = Field(default=None, gt=0)
    max_peak_vram_bytes: int | None = Field(default=None, gt=0)
    max_processing_seconds: float | None = Field(default=None, gt=0)


class ResourceLimitDecision(BaseModel):
    limit_name: str
    limit_type: LimitType
    observed_value: float
    configured_limit: float
    exceeded: bool
    action: str
    message: str


class StressScenario(BaseModel):
    scenario_id: str
    description: str
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    change_density: str
    expected_candidate_scale: int = Field(default=1, ge=0)
    expected_screenshot_scale: int = Field(default=1, ge=0)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    required_stages: list[str] = Field(default_factory=list)


class ScalingAnalysis(BaseModel):
    metric: str
    input_scale: float = Field(gt=0)
    observed_scale: float = Field(ge=0)
    classification: ScalingClassification


class StressTestReport(BaseModel):
    version: str = STRESS_REPORT_VERSION
    algorithm_version: str = STRESS_ALGORITHM_VERSION
    scenario: StressScenario
    pipeline_fingerprint: str
    total_runtime_seconds: float = Field(ge=0)
    real_time_factor: float | None = Field(default=None, ge=0)
    peak_rss_bytes: int = Field(default=0, ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    workspace_bytes: int = Field(default=0, ge=0)
    source_file_bytes: int = Field(default=0, ge=0)
    temp_bytes: int = Field(default=0, ge=0)
    sampled_frames: int = Field(default=0, ge=0)
    candidates: int = Field(default=0, ge=0)
    semantic_candidates: int = Field(default=0, ge=0)
    final_screenshots: int = Field(default=0, ge=0)
    pdf_pages: int = Field(default=0, ge=0)
    pdf_bytes: int = Field(default=0, ge=0)
    limit_decisions: list[ResourceLimitDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: StressStatus = StressStatus.PASS

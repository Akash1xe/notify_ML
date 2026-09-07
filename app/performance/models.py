from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.models import stable_fingerprint

PERFORMANCE_REPORT_VERSION = "1"
PERFORMANCE_ALGORITHM_VERSION = "1"


class PerformanceMode(str, Enum):
    COLD_RUN = "COLD_RUN"
    WARM_RUN = "WARM_RUN"
    CACHE_HIT_RUN = "CACHE_HIT_RUN"


class StagePerformanceMetrics(BaseModel):
    stage: str
    duration_seconds: float = Field(ge=0)
    cpu_time_seconds: float | None = Field(default=None, ge=0)
    cpu_percent_mean: float | None = Field(default=None, ge=0)
    cpu_percent_peak: float | None = Field(default=None, ge=0)
    rss_memory_start_bytes: int | None = Field(default=None, ge=0)
    rss_memory_peak_bytes: int | None = Field(default=None, ge=0)
    rss_memory_delta_bytes: int | None = None
    gpu_memory_peak_bytes: int | None = Field(default=None, ge=0)
    gpu_utilization_mean: float | None = Field(default=None, ge=0)
    disk_read_bytes: int | None = Field(default=None, ge=0)
    disk_write_bytes: int | None = Field(default=None, ge=0)
    input_count: int | None = Field(default=None, ge=0)
    output_count: int | None = Field(default=None, ge=0)
    cache_hit: bool = False


class HardwareSummary(BaseModel):
    cpu_count_logical: int = Field(ge=1)
    ram_total_bytes: int = Field(ge=0)
    platform: str
    python_version: str
    gpu_name: str | None = None
    gpu_vram_bytes: int | None = Field(default=None, ge=0)


class ArtifactMetrics(BaseModel):
    source_video_bytes: int = Field(default=0, ge=0)
    normalized_media_bytes: int = Field(default=0, ge=0)
    sampled_frame_bytes: int = Field(default=0, ge=0)
    transcript_bytes: int = Field(default=0, ge=0)
    semantic_artifact_bytes: int = Field(default=0, ge=0)
    screenshot_bytes: int = Field(default=0, ge=0)
    pdf_bytes: int = Field(default=0, ge=0)
    workspace_bytes: int = Field(default=0, ge=0)


class PerformanceProfilingConfig(BaseModel):
    resource_sample_interval_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    collect_cpu: bool = True
    collect_memory: bool = True
    collect_gpu: bool = True
    collect_disk: bool = True
    repeat_count: int = Field(default=1, ge=1, le=20)
    warmup_count: int = Field(default=0, ge=0, le=10)
    min_meaningful_delta_ratio: float = Field(default=0.03, ge=0, le=1)
    max_memory_regression_ratio: float = Field(default=0.20, ge=0, le=10)
    max_disk_regression_ratio: float = Field(default=0.20, ge=0, le=10)


class PerformanceRun(BaseModel):
    version: str = PERFORMANCE_REPORT_VERSION
    algorithm_version: str = PERFORMANCE_ALGORITHM_VERSION
    run_id: str
    sample_id: str
    mode: PerformanceMode
    pipeline_fingerprint: str
    performance_config_fingerprint: str
    hardware: HardwareSummary
    stage_metrics: list[StagePerformanceMetrics]
    artifact_metrics: ArtifactMetrics = Field(default_factory=ArtifactMetrics)
    video_duration_seconds: float = Field(default=0, ge=0)
    counts: dict[str, int] = Field(default_factory=dict)
    quality_metrics_reference: str | None = None

    @property
    def total_runtime_seconds(self) -> float:
        return round(sum(item.duration_seconds for item in self.stage_metrics), 6)

    @property
    def real_time_factor(self) -> float | None:
        if self.video_duration_seconds <= 0:
            return None
        return round(self.total_runtime_seconds / self.video_duration_seconds, 6)

    @property
    def processing_seconds_per_video_minute(self) -> float | None:
        if self.video_duration_seconds <= 0:
            return None
        return round(self.total_runtime_seconds / (self.video_duration_seconds / 60.0), 6)

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "algorithm": self.algorithm_version,
                "sample_id": self.sample_id,
                "mode": self.mode,
                "pipeline": self.pipeline_fingerprint,
                "config": self.performance_config_fingerprint,
                "counts": self.counts,
            }
        )


class PerformanceOptimizationScorecard(BaseModel):
    runtime_improved: bool
    memory_within_guardrail: bool
    disk_within_guardrail: bool
    quality_passed: bool
    critical_regressions_zero: bool

    @property
    def accepted(self) -> bool:
        return all(self.model_dump().values())


class PerformanceComparison(BaseModel):
    baseline_runtime_seconds: float
    candidate_runtime_seconds: float
    runtime_delta_seconds: float
    runtime_improvement_ratio: float
    baseline_peak_memory_bytes: int
    candidate_peak_memory_bytes: int
    memory_regression_ratio: float
    baseline_workspace_bytes: int
    candidate_workspace_bytes: int
    disk_regression_ratio: float
    scorecard: PerformanceOptimizationScorecard


class OptimizationRun(BaseModel):
    optimization_id: str
    description: str
    affected_stage: str
    before_fingerprint: str
    after_fingerprint: str
    comparison: PerformanceComparison
    provenance: dict[str, Any] = Field(default_factory=dict)

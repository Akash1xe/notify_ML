from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.media.models import FileFingerprint


class SampledFrame(BaseModel):
    index: int = Field(ge=1)
    timestamp_seconds: float = Field(ge=0)
    relative_path: str
    file_size_bytes: int = Field(gt=0)


class SamplingManifest(BaseModel):
    algorithm_version: str
    source_video: str
    source_fingerprint: FileFingerprint
    sample_fps: float = Field(gt=0)
    interval_seconds: float = Field(gt=0)
    video_duration_seconds: float = Field(gt=0)
    expected_frame_count: int = Field(ge=1)
    actual_frame_count: int = Field(ge=1)
    image_format: str = "jpg"
    jpeg_quality: int = Field(ge=1, le=100)
    config_fingerprint: str
    artifact_fingerprint: str
    frames: list[SampledFrame]
    sampling_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_count(self) -> "SamplingManifest":
        if self.actual_frame_count != len(self.frames):
            raise ValueError("Sampling frame count does not match frame records")
        return self


class FrameQuality(BaseModel):
    index: int = Field(ge=1)
    timestamp_seconds: float = Field(ge=0)
    source_path: str
    processed_path: str | None = None
    source_width: int | None = Field(default=None, ge=1)
    source_height: int | None = Field(default=None, ge=1)
    analysis_width: int | None = Field(default=None, ge=1)
    analysis_height: int | None = Field(default=None, ge=1)
    brightness: float | None = Field(default=None, ge=0, le=1)
    contrast: float | None = Field(default=None, ge=0, le=1)
    sharpness_score: float | None = Field(default=None, ge=0)
    edge_density: float | None = Field(default=None, ge=0, le=1)
    black_pixel_ratio: float | None = Field(default=None, ge=0, le=1)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    is_valid: bool = True
    is_black: bool = False
    is_very_dark: bool = False
    is_blurry: bool = False
    is_low_information: bool = False
    invalid_reason: str | None = None


class PreprocessingStats(BaseModel):
    total_frames: int = Field(ge=0)
    valid_frames: int = Field(ge=0)
    corrupt_frames: int = Field(ge=0)
    black_frames: int = Field(ge=0)
    blurry_frames: int = Field(ge=0)
    low_information_frames: int = Field(ge=0)


class PreprocessingManifest(BaseModel):
    algorithm_version: str
    sampling_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    analysis_width: int = Field(ge=1)
    stats: PreprocessingStats
    frames: list[FrameQuality]
    preprocessing_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DifferenceRecord(BaseModel):
    previous_frame_index: int = Field(ge=1)
    current_frame_index: int = Field(ge=1)
    previous_timestamp_seconds: float = Field(ge=0)
    current_timestamp_seconds: float = Field(ge=0)
    delta_seconds: float = Field(ge=0)
    pixel_difference: float | None = Field(default=None, ge=0, le=1)
    ssim_similarity: float | None = Field(default=None, ge=0, le=1)
    ssim_difference: float | None = Field(default=None, ge=0, le=1)
    phash_difference: float | None = Field(default=None, ge=0, le=1)
    edge_difference: float | None = Field(default=None, ge=0, le=1)
    difference_score: float | None = Field(default=None, ge=0, le=1)
    previous_is_black: bool = False
    current_is_black: bool = False
    valid: bool = True
    reason: str | None = None


class DifferenceStats(BaseModel):
    total_comparisons: int = Field(ge=0)
    valid_comparisons: int = Field(ge=0)
    invalid_comparisons: int = Field(ge=0)
    mean_difference_score: float = Field(default=0.0, ge=0, le=1)
    median_difference_score: float = Field(default=0.0, ge=0, le=1)
    max_difference_score: float = Field(default=0.0, ge=0, le=1)
    p50: float = Field(default=0.0, ge=0, le=1)
    p75: float = Field(default=0.0, ge=0, le=1)
    p90: float = Field(default=0.0, ge=0, le=1)
    p95: float = Field(default=0.0, ge=0, le=1)
    p99: float = Field(default=0.0, ge=0, le=1)


class DifferenceManifest(BaseModel):
    algorithm_version: str
    preprocessing_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: DifferenceStats
    comparisons: list[DifferenceRecord]
    comparison_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MajorEventType(str, Enum):
    MAJOR_VISUAL_CHANGE = "MAJOR_VISUAL_CHANGE"
    VERY_MAJOR_VISUAL_CHANGE = "VERY_MAJOR_VISUAL_CHANGE"
    BLACK_TRANSITION = "BLACK_TRANSITION"
    INVALID_GAP = "INVALID_GAP"


class MajorChangeThresholds(BaseModel):
    median_score: float = Field(ge=0, le=1)
    mad: float = Field(ge=0)
    p90: float = Field(ge=0, le=1)
    p95: float = Field(ge=0, le=1)
    p99: float = Field(ge=0, le=1)
    major_threshold: float = Field(ge=0, le=1)
    very_major_threshold: float = Field(ge=0, le=1)


class MajorChangeEvent(BaseModel):
    event_id: int = Field(ge=1)
    event_type: MajorEventType
    timestamp_seconds: float = Field(ge=0)
    start_timestamp_seconds: float = Field(ge=0)
    end_timestamp_seconds: float = Field(ge=0)
    previous_frame_index: int | None = Field(default=None, ge=1)
    current_frame_index: int | None = Field(default=None, ge=1)
    difference_score: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)
    robust_score: float | None = None
    metric_agreement: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)


class MajorChangeStats(BaseModel):
    total_comparisons: int = Field(ge=0)
    major_events: int = Field(ge=0)
    very_major_events: int = Field(ge=0)
    black_transitions: int = Field(ge=0)
    invalid_gaps: int = Field(ge=0)
    average_seconds_between_major_changes: float | None = Field(default=None, ge=0)
    median_seconds_between_major_changes: float | None = Field(default=None, ge=0)
    event_ratio: float = Field(default=0.0, ge=0)


class MajorChangesManifest(BaseModel):
    algorithm_version: str
    differences_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    thresholds: MajorChangeThresholds
    stats: MajorChangeStats
    events: list[MajorChangeEvent]
    detection_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TimelineState(str, Enum):
    STABLE = "STABLE"
    CHANGING = "CHANGING"
    MAJOR_TRANSITION = "MAJOR_TRANSITION"
    BLACK_TRANSITION = "BLACK_TRANSITION"
    INVALID = "INVALID"


class TimelineSegment(BaseModel):
    segment_id: int = Field(ge=1)
    state: TimelineState
    start_timestamp_seconds: float = Field(ge=0)
    end_timestamp_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    start_frame_index: int | None = Field(default=None, ge=1)
    end_frame_index: int | None = Field(default=None, ge=1)
    mean_difference_score: float | None = Field(default=None, ge=0, le=1)
    median_difference_score: float | None = Field(default=None, ge=0, le=1)
    min_difference_score: float | None = Field(default=None, ge=0, le=1)
    max_difference_score: float | None = Field(default=None, ge=0, le=1)
    change_intensity: float | None = Field(default=None, ge=0, le=1)
    comparison_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> "TimelineSegment":
        if self.end_timestamp_seconds < self.start_timestamp_seconds:
            raise ValueError("Timeline segment end must be >= start")
        return self


class TimelineThresholds(BaseModel):
    enter_stable_threshold: float = Field(ge=0, le=1)
    exit_stable_threshold: float = Field(ge=0, le=1)
    major_threshold: float = Field(ge=0, le=1)


class TimelineStats(BaseModel):
    total_duration_seconds: float = Field(default=0.0, ge=0)
    stable_duration_seconds: float = Field(default=0.0, ge=0)
    changing_duration_seconds: float = Field(default=0.0, ge=0)
    transition_duration_seconds: float = Field(default=0.0, ge=0)
    invalid_duration_seconds: float = Field(default=0.0, ge=0)
    stable_segment_count: int = Field(default=0, ge=0)
    changing_segment_count: int = Field(default=0, ge=0)
    transition_segment_count: int = Field(default=0, ge=0)
    invalid_segment_count: int = Field(default=0, ge=0)
    stable_ratio: float = Field(default=0.0, ge=0, le=1)
    changing_ratio: float = Field(default=0.0, ge=0, le=1)
    invalid_ratio: float = Field(default=0.0, ge=0, le=1)


class TimelineManifest(BaseModel):
    algorithm_version: str
    differences_fingerprint: str
    major_changes_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    thresholds: TimelineThresholds
    stats: TimelineStats
    segments: list[TimelineSegment]
    timeline_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_order(self) -> "TimelineManifest":
        previous_end: float | None = None
        for segment in self.segments:
            if previous_end is not None and segment.start_timestamp_seconds < previous_end - 1e-6:
                raise ValueError("Timeline segments overlap or are not chronological")
            previous_end = segment.end_timestamp_seconds
        return self


class Phase3Timings(BaseModel):
    sampling_seconds: float = Field(default=0.0, ge=0)
    preprocessing_seconds: float = Field(default=0.0, ge=0)
    difference_seconds: float = Field(default=0.0, ge=0)
    major_change_seconds: float = Field(default=0.0, ge=0)
    timeline_seconds: float = Field(default=0.0, ge=0)
    total_phase3_seconds: float = Field(default=0.0, ge=0)


class Phase3DiskUsage(BaseModel):
    sampled_frames_bytes: int = Field(default=0, ge=0)
    processed_frames_bytes: int = Field(default=0, ge=0)
    analysis_metadata_bytes: int = Field(default=0, ge=0)
    phase3_total_bytes: int = Field(default=0, ge=0)


class FrameAnalysisSummary(BaseModel):
    sampled_frame_count: int = Field(ge=0)
    processed_frame_count: int = Field(ge=0)
    invalid_frame_count: int = Field(ge=0)
    difference_count: int = Field(ge=0)
    major_event_count: int = Field(ge=0)
    stable_segment_count: int = Field(ge=0)
    changing_segment_count: int = Field(ge=0)
    analysis_start_seconds: float = Field(default=0.0, ge=0)
    analysis_end_seconds: float = Field(default=0.0, ge=0)
    timings: Phase3Timings
    disk_usage: Phase3DiskUsage
    sampling_fingerprint: str
    preprocessing_fingerprint: str
    differences_fingerprint: str
    major_changes_fingerprint: str
    timeline_fingerprint: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AnalysisArtifactState(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    STALE = "STALE"
    CORRUPT = "CORRUPT"
    PARTIAL = "PARTIAL"


class AnalysisArtifactCheck(BaseModel):
    state: AnalysisArtifactState
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.state is AnalysisArtifactState.VALID


class Phase3ResumeStage(str, Enum):
    SAMPLING_FRAMES = "SAMPLING_FRAMES"
    PREPROCESSING_FRAMES = "PREPROCESSING_FRAMES"
    DETECTING_CHANGES = "DETECTING_CHANGES"
    DETECTING_MAJOR_CHANGES = "DETECTING_MAJOR_CHANGES"
    DETECTING_STABILITY = "DETECTING_STABILITY"
    PHASE3_READY = "PHASE3_READY"


class AnalysisCacheSnapshot(BaseModel):
    sampling: AnalysisArtifactCheck
    preprocessing: AnalysisArtifactCheck
    differences: AnalysisArtifactCheck
    major_changes: AnalysisArtifactCheck
    timeline: AnalysisArtifactCheck
    frame_analysis_complete: AnalysisArtifactCheck
    resume_stage: Phase3ResumeStage


class EvaluationWarning(BaseModel):
    code: str
    message: str


class Phase3EvaluationReport(BaseModel):
    lecture_duration_seconds: float = Field(ge=0)
    sampled_frames: int = Field(ge=0)
    valid_frames: int = Field(ge=0)
    invalid_frames: int = Field(ge=0)
    black_frames: int = Field(ge=0)
    blurry_frames: int = Field(ge=0)
    low_information_frames: int = Field(ge=0)
    score_distribution: dict[str, float]
    major_events: int = Field(ge=0)
    events_per_minute: float = Field(ge=0)
    stable_ratio: float = Field(ge=0, le=1)
    changing_ratio: float = Field(ge=0, le=1)
    segments_per_minute: float = Field(ge=0)
    mean_stable_segment_seconds: float = Field(default=0.0, ge=0)
    median_stable_segment_seconds: float = Field(default=0.0, ge=0)
    mean_changing_segment_seconds: float = Field(default=0.0, ge=0)
    median_changing_segment_seconds: float = Field(default=0.0, ge=0)
    example_event_timestamps: list[float] = []
    warnings: list[EvaluationWarning] = []
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ArtifactEnvelope(BaseModel):
    """Small helper for future schema migration/debug endpoints."""

    kind: str
    payload: dict[str, Any]

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

BENCHMARK_DATASET_MANIFEST_VERSION = "1"
BENCHMARK_ALGORITHM_VERSION = "1"
BENCHMARK_REPORT_VERSION = "1"
QUALITY_BASELINE_REPORT_VERSION = "1"
QUALITY_EVALUATION_ALGORITHM_VERSION = "1"


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def stable_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class SourceType(str, Enum):
    LOCAL_FILE = "LOCAL_FILE"
    YOUTUBE_REFERENCE = "YOUTUBE_REFERENCE"
    SYNTHETIC = "SYNTHETIC"


class ContentType(str, Enum):
    SLIDES = "SLIDES"
    SLIDE = "SLIDE"
    WHITEBOARD = "WHITEBOARD"
    BLACKBOARD = "BLACKBOARD"
    CODE = "CODE"
    DIAGRAM = "DIAGRAM"
    EQUATION = "EQUATION"
    DOCUMENT = "DOCUMENT"
    UI_DEMO = "UI_DEMO"
    MIXED = "MIXED"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class Importance(str, Enum):
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    IGNORE = "IGNORE"


class CompletionState(str, Enum):
    INCOMPLETE = "INCOMPLETE"
    COMPLETE = "COMPLETE"
    TRANSITION = "TRANSITION"


class NegativeReason(str, Enum):
    INCOMPLETE_WRITING = "INCOMPLETE_WRITING"
    TRANSITION = "TRANSITION"
    BLACK_FRAME = "BLACK_FRAME"
    EMPTY = "EMPTY"
    BLURRY = "BLURRY"
    OCCLUDED = "OCCLUDED"
    LOW_INFORMATION = "LOW_INFORMATION"
    TEMPORARY_UI = "TEMPORARY_UI"
    DUPLICATE = "DUPLICATE"
    OTHER = "OTHER"


class AnnotationReviewStatus(str, Enum):
    DRAFT = "DRAFT"
    REVIEWED = "REVIEWED"
    LOCKED = "LOCKED"


class PredictionLevel(str, Enum):
    CANDIDATE = "CANDIDATE"
    SEMANTIC = "SEMANTIC"
    FINAL_SCREENSHOT = "FINAL_SCREENSHOT"
    DOCUMENT = "DOCUMENT"


class DatasetSplit(str, Enum):
    CALIBRATION = "CALIBRATION"
    VALIDATION = "VALIDATION"
    CI = "CI"
    REGRESSION = "REGRESSION"
    NO_SPLIT = "NO_SPLIT"


class SampleSource(BaseModel):
    relative_path: str | None = None
    url: str | None = None
    sha256: str | None = None


class BenchmarkSample(BaseModel):
    sample_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_type: SourceType
    source: SampleSource = Field(default_factory=SampleSource)
    duration_seconds: float = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps: float = Field(gt=0)
    categories: list[ContentType] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    annotation_path: str
    split: DatasetSplit = DatasetSplit.NO_SPLIT


class DatasetManifest(BaseModel):
    version: str = BENCHMARK_DATASET_MANIFEST_VERSION
    samples: list[BenchmarkSample]

    @model_validator(mode="after")
    def unique_samples(self) -> "DatasetManifest":
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sample_id in dataset manifest")
        return self


class GroundTruthVisualState(BaseModel):
    state_id: str = Field(min_length=1)
    target_timestamp_seconds: float = Field(ge=0)
    acceptable_start_seconds: float = Field(ge=0)
    acceptable_end_seconds: float = Field(ge=0)
    content_type: ContentType = ContentType.UNKNOWN
    importance: Importance = Importance.REQUIRED
    completion: CompletionState = CompletionState.COMPLETE
    meaningful: bool = True
    duplicate_group: str | None = None
    acceptable_equivalence_group: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def valid_window(self) -> "GroundTruthVisualState":
        if not (
            self.acceptable_start_seconds
            <= self.target_timestamp_seconds
            <= self.acceptable_end_seconds
        ):
            raise ValueError("ground-truth target must lie inside acceptable window")
        return self


class GroundTruthNegativeState(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    reason: NegativeReason
    notes: str = ""

    @model_validator(mode="after")
    def valid_window(self) -> "GroundTruthNegativeState":
        if self.end_seconds < self.start_seconds:
            raise ValueError("negative-state end must be >= start")
        return self


class GroundTruthAnnotation(BaseModel):
    version: str = "1"
    sample_id: str
    annotator: str = "manual"
    annotation_version: str = "1"
    review_status: AnnotationReviewStatus = AnnotationReviewStatus.DRAFT
    states: list[GroundTruthVisualState] = Field(default_factory=list)
    negative_states: list[GroundTruthNegativeState] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def unique_state_ids(self) -> "GroundTruthAnnotation":
        ids = [state.state_id for state in self.states]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ground-truth state_id")
        return self


class PredictionRecord(BaseModel):
    prediction_id: str
    timestamp_seconds: float = Field(ge=0)
    content_type: ContentType = ContentType.UNKNOWN
    candidate_id: str | int | None = None
    duplicate_group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchRecord(BaseModel):
    ground_truth_state_id: str
    prediction_id: str
    timestamp_error_seconds: float = Field(ge=0)


class FailureRecord(BaseModel):
    sample_id: str
    failure_type: str
    severity: Literal["CRITICAL", "MAJOR", "MINOR", "INFO"]
    ground_truth_state_id: str | None = None
    prediction_id: str | None = None
    target_timestamp_seconds: float | None = None
    prediction_timestamp_seconds: float | None = None
    notes: str = ""


class MetricSummary(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    required_state_recall: float = 0.0
    negative_state_violation_rate: float = 0.0
    timestamp_error_mean: float | None = None
    timestamp_error_median: float | None = None
    timestamp_error_p90: float | None = None
    timestamp_error_max: float | None = None


class StageEvaluationResult(BaseModel):
    stage: PredictionLevel
    prediction_count: int
    ground_truth_required_count: int
    metrics: MetricSummary
    matches: list[MatchRecord] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)
    false_negatives: list[str] = Field(default_factory=list)


class RequiredStateOutcome(BaseModel):
    ground_truth_state_id: str
    candidate_found: bool = False
    semantic_kept: bool = False
    final_screenshot_found: bool = False
    document_found: bool = False
    earliest_loss_stage: str | None = None


class SampleQualityEvaluation(BaseModel):
    sample_id: str
    stages: dict[PredictionLevel, StageEvaluationResult]
    required_state_outcomes: list[RequiredStateOutcome] = Field(default_factory=list)
    failure_records: list[FailureRecord] = Field(default_factory=list)


class BenchmarkConfig(BaseModel):
    prediction_level: PredictionLevel = PredictionLevel.FINAL_SCREENSHOT
    include_optional_states: bool = True
    allowed_review_status: tuple[AnnotationReviewStatus, ...] = (
        AnnotationReviewStatus.REVIEWED,
        AnnotationReviewStatus.LOCKED,
    )
    strict: bool = True


class BenchmarkReport(BaseModel):
    version: str = BENCHMARK_REPORT_VERSION
    algorithm_version: str = BENCHMARK_ALGORITHM_VERSION
    dataset_fingerprint: str
    benchmark_input_fingerprint: str
    prediction_level: PredictionLevel
    aggregate: MetricSummary
    by_category: dict[str, MetricSummary] = Field(default_factory=dict)
    by_tag: dict[str, MetricSummary] = Field(default_factory=dict)
    samples: list[SampleQualityEvaluation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QualityBaselineReport(BaseModel):
    version: str = QUALITY_BASELINE_REPORT_VERSION
    algorithm_version: str = QUALITY_EVALUATION_ALGORITHM_VERSION
    dataset_fingerprint: str
    pipeline_fingerprint: str
    overall: MetricSummary
    candidate: MetricSummary
    semantic: MetricSummary
    final_screenshot: MetricSummary
    document: MetricSummary
    required_visual_state_recall: float
    final_visual_note_precision: float
    final_visual_note_f1: float
    stage_loss_counts: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, MetricSummary] = Field(default_factory=dict)
    by_tag: dict[str, MetricSummary] = Field(default_factory=dict)
    samples: list[SampleQualityEvaluation] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: Literal["DRAFT", "LOCKED"] = "DRAFT"

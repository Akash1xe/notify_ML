from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.candidate_analysis.models import BoundaryType, SelectionRole
from app.transcription.models import AlignmentType
from app.video_analysis.models import AnalysisArtifactCheck, TimelineState


SEMANTIC_INPUT_ALGORITHM_VERSION = "1"
TEMPORAL_CONTEXT_ALGORITHM_VERSION = "1"
VLM_RUNTIME_ALGORITHM_VERSION = "1"
SEMANTIC_PROMPT_VERSION = "1"
SEMANTIC_ANALYSIS_ALGORITHM_VERSION = "1"
SEMANTIC_DECISION_ALGORITHM_VERSION = "1"


class SemanticFrameRef(BaseModel):
    frame_index: int = Field(ge=1)
    timestamp_seconds: float = Field(ge=0)
    relative_path: str
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    sharpness_score: float | None = Field(default=None, ge=0)
    is_valid: bool = True
    is_black: bool = False


class SemanticInputRecord(BaseModel):
    candidate_id: int = Field(ge=1)
    selection_role: SelectionRole
    candidate_timestamp_seconds: float = Field(ge=0)
    semantic_frame: SemanticFrameRef
    sampled_frame_path: str
    processed_frame_path: str
    ranking_score: float = Field(ge=0, le=1)
    ranking_position: int | None = Field(default=None, ge=1)
    selection_confidence: float = Field(ge=0, le=1)
    is_ambiguous: bool = False
    stable_window_id: int = Field(ge=1)
    stable_window_start_seconds: float = Field(ge=0)
    stable_window_end_seconds: float = Field(ge=0)
    relative_position_in_stable_window: float = Field(ge=0, le=1)
    boundary_id: int | None = Field(default=None, ge=1)
    boundary_type: BoundaryType | None = None
    visual_boundary_timestamp_seconds: float | None = Field(default=None, ge=0)
    boundary_score: float | None = Field(default=None, ge=0, le=1)
    seconds_after_boundary: float | None = None
    alignment_type: AlignmentType
    speech_proximity_score: float = Field(ge=0, le=1)
    nearest_speech_gap_seconds: float | None = Field(default=None, ge=0)
    before_text: str = ""
    current_text: str = ""
    after_text: str = ""
    has_transcript_context: bool = False
    comparison_group_id: str
    semantic_input_fingerprint: str

    @model_validator(mode="after")
    def validate_window(self) -> "SemanticInputRecord":
        if self.stable_window_end_seconds < self.stable_window_start_seconds:
            raise ValueError("stable-window end must be >= start")
        return self


class SemanticInputStats(BaseModel):
    input_count: int = Field(default=0, ge=0)
    primary_input_count: int = Field(default=0, ge=0)
    alternate_input_count: int = Field(default=0, ge=0)
    inputs_with_transcript: int = Field(default=0, ge=0)
    inputs_without_transcript: int = Field(default=0, ge=0)
    ambiguous_input_count: int = Field(default=0, ge=0)
    unique_stable_window_count: int = Field(default=0, ge=0)
    mean_context_characters: float = Field(default=0.0, ge=0)
    median_context_characters: float = Field(default=0.0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class SemanticInputManifest(BaseModel):
    algorithm_version: str = SEMANTIC_INPUT_ALGORITHM_VERSION
    candidate_selections_fingerprint: str
    ranked_candidates_fingerprint: str
    transcript_contexts_fingerprint: str
    transcript_alignment_fingerprint: str
    frame_manifest_fingerprint: str
    preprocessing_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: SemanticInputStats
    inputs: list[SemanticInputRecord]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VLMDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class VLMQuantization(str, Enum):
    NONE = "none"
    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"


class HardwareCapability(BaseModel):
    torch_available: bool = False
    cuda_available: bool = False
    cuda_device_count: int = Field(default=0, ge=0)
    gpu_name: str | None = None
    gpu_total_memory_mb: int | None = Field(default=None, ge=0)
    gpu_free_memory_mb: int | None = Field(default=None, ge=0)
    system_ram_total_mb: int | None = Field(default=None, ge=0)
    system_ram_available_mb: int | None = Field(default=None, ge=0)
    cpu_count: int | None = Field(default=None, ge=0)


class VLGenerationConfig(BaseModel):
    max_new_tokens: int = Field(default=512, ge=32, le=4096)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    do_sample: bool = False
    repetition_penalty: float = Field(default=1.0, gt=0, le=5)


class VLMRuntimeMetadata(BaseModel):
    configured_model_tier: str
    configured_model_name: str
    primary_model: str
    resolved_model: str
    fallback_model: str | None = None
    fallback_enabled: bool = True
    fallback_used: bool = False
    fallback_reason: str | None = None
    configured_device: str
    resolved_device: str
    resolved_dtype: str
    quantization: str
    model_revision: str | None = None
    model_loaded: bool = False
    transformers_version: str | None = None
    torch_version: str | None = None
    runtime_fingerprint: str


class VLMGenerationResult(BaseModel):
    text: str
    model_name: str
    resolved_device: str
    resolved_dtype: str
    generation_seconds: float = Field(default=0.0, ge=0)
    input_image_count: int = Field(default=0, ge=0)
    output_token_count: int | None = Field(default=None, ge=0)
    fallback_used: bool = False
    fallback_reason: str | None = None
    runtime_fingerprint: str


class TemporalFrameRole(str, Enum):
    PREVIOUS = "PREVIOUS"
    CURRENT = "CURRENT"
    NEXT = "NEXT"


class TemporalContextType(str, Enum):
    CURRENT_ONLY = "CURRENT_ONLY"
    PREVIOUS_CURRENT = "PREVIOUS_CURRENT"
    CURRENT_NEXT = "CURRENT_NEXT"
    TRIPLET = "TRIPLET"


class TemporalFrame(BaseModel):
    role: TemporalFrameRole
    frame_index: int = Field(ge=1)
    timestamp_seconds: float = Field(ge=0)
    relative_path: str
    timeline_state: TimelineState | None = None
    quality_score: float | None = Field(default=None, ge=0, le=1)
    sharpness_score: float | None = Field(default=None, ge=0)
    relative_position_in_window: float | None = Field(default=None, ge=0, le=1)
    seconds_relative_to_boundary: float | None = None


class TemporalVisualContext(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    current: TemporalFrame
    previous: TemporalFrame | None = None
    next: TemporalFrame | None = None
    context_type: TemporalContextType
    previous_gap_seconds: float | None = Field(default=None, ge=0)
    next_gap_seconds: float | None = Field(default=None, ge=0)
    previous_to_current_change_score: float | None = Field(default=None, ge=0, le=1)
    current_to_next_change_score: float | None = Field(default=None, ge=0, le=1)
    temporal_context_fingerprint: str

    @model_validator(mode="after")
    def validate_distinct(self) -> "TemporalVisualContext":
        ids = [self.current.frame_index]
        if self.previous:
            ids.append(self.previous.frame_index)
        if self.next:
            ids.append(self.next.frame_index)
        if len(ids) != len(set(ids)):
            raise ValueError("temporal context frame IDs must be distinct")
        return self


class TemporalContextStats(BaseModel):
    context_count: int = Field(default=0, ge=0)
    triplet_count: int = Field(default=0, ge=0)
    previous_current_count: int = Field(default=0, ge=0)
    current_next_count: int = Field(default=0, ge=0)
    current_only_count: int = Field(default=0, ge=0)
    mean_previous_gap_seconds: float = Field(default=0.0, ge=0)
    mean_next_gap_seconds: float = Field(default=0.0, ge=0)
    median_previous_gap_seconds: float = Field(default=0.0, ge=0)
    median_next_gap_seconds: float = Field(default=0.0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class TemporalContextsManifest(BaseModel):
    algorithm_version: str = TEMPORAL_CONTEXT_ALGORITHM_VERSION
    semantic_input_fingerprint: str
    frame_manifest_fingerprint: str
    preprocessing_fingerprint: str
    timeline_fingerprint: str
    differences_fingerprint: str
    stability_windows_fingerprint: str
    boundaries_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: TemporalContextStats
    contexts: list[TemporalVisualContext]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SemanticContentType(str, Enum):
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
    EMPTY_OR_LOW_INFORMATION = "EMPTY_OR_LOW_INFORMATION"
    UNKNOWN = "UNKNOWN"


class CompletionState(str, Enum):
    COMPLETE = "COMPLETE"
    MOSTLY_COMPLETE = "MOSTLY_COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    TRANSITION = "TRANSITION"
    UNCERTAIN = "UNCERTAIN"


class EducationalUsefulness(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class VisualChangeState(str, Enum):
    STILL_CHANGING = "STILL_CHANGING"
    MOSTLY_SETTLED = "MOSTLY_SETTLED"
    SETTLED = "SETTLED"
    UNCERTAIN = "UNCERTAIN"


class TranscriptVisualConsistency(str, Enum):
    CONSISTENT = "CONSISTENT"
    PARTIALLY_CONSISTENT = "PARTIALLY_CONSISTENT"
    UNRELATED = "UNRELATED"
    NO_TRANSCRIPT = "NO_TRANSCRIPT"
    UNCERTAIN = "UNCERTAIN"


class SemanticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: int = Field(ge=1)
    content_type: SemanticContentType
    completion_state: CompletionState
    completion_score: float = Field(ge=0, le=1)
    visual_change_state: VisualChangeState
    educational_usefulness: EducationalUsefulness
    usefulness_score: float = Field(ge=0, le=1)
    transition_probability: float = Field(ge=0, le=1)
    has_meaningful_visual_content: bool
    teacher_still_writing_likely: bool
    transcript_visual_consistency: TranscriptVisualConsistency
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list, max_length=12)
    short_rationale: str | None = Field(default=None, max_length=240)


class SemanticParseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_JSON = "INVALID_JSON"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    RUNTIME_ERROR = "RUNTIME_ERROR"


class CandidateSemanticArtifact(BaseModel):
    candidate_id: int = Field(ge=1)
    semantic_input_fingerprint: str
    temporal_context_fingerprint: str
    inference_fingerprint: str
    prompt_version: str
    analysis_algorithm_version: str
    prompt_fingerprint: str
    raw_text: str
    parse_status: SemanticParseStatus
    semantic_result: SemanticResult | None = None
    resolved_model: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    runtime_fingerprint: str
    inference_seconds: float = Field(default=0.0, ge=0)
    attempt_count: int = Field(default=1, ge=1)
    artifact_fingerprint: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SemanticAnalysisStats(BaseModel):
    candidate_count: int = Field(default=0, ge=0)
    successful_analysis_count: int = Field(default=0, ge=0)
    failed_analysis_count: int = Field(default=0, ge=0)
    cached_analysis_count: int = Field(default=0, ge=0)
    complete_count: int = Field(default=0, ge=0)
    mostly_complete_count: int = Field(default=0, ge=0)
    incomplete_count: int = Field(default=0, ge=0)
    transition_count: int = Field(default=0, ge=0)
    uncertain_count: int = Field(default=0, ge=0)
    mean_confidence: float = Field(default=0.0, ge=0, le=1)
    median_confidence: float = Field(default=0.0, ge=0, le=1)
    mean_inference_seconds: float = Field(default=0.0, ge=0)
    median_inference_seconds: float = Field(default=0.0, ge=0)
    p90_inference_seconds: float = Field(default=0.0, ge=0)
    total_inference_seconds: float = Field(default=0.0, ge=0)
    invalid_json_count: int = Field(default=0, ge=0)
    schema_error_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    retry_success_count: int = Field(default=0, ge=0)
    fallback_candidate_count: int = Field(default=0, ge=0)
    model_usage_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SemanticResultsManifest(BaseModel):
    algorithm_version: str = SEMANTIC_ANALYSIS_ALGORITHM_VERSION
    prompt_version: str = SEMANTIC_PROMPT_VERSION
    semantic_input_fingerprint: str
    temporal_contexts_fingerprint: str
    inference_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: SemanticAnalysisStats
    results: list[SemanticResult]
    candidate_artifact_fingerprints: dict[str, str] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandidateDecision(str, Enum):
    KEEP = "KEEP"
    REJECT_INCOMPLETE = "REJECT_INCOMPLETE"
    REJECT_LOW_VALUE = "REJECT_LOW_VALUE"
    REJECT_TRANSITION = "REJECT_TRANSITION"
    REJECT_LOW_CONFIDENCE = "REJECT_LOW_CONFIDENCE"
    REJECT_INVALID_SEMANTIC_RESULT = "REJECT_INVALID_SEMANTIC_RESULT"
    REVIEW_ALTERNATE = "REVIEW_ALTERNATE"


class WindowDecisionType(str, Enum):
    SELECT_PRIMARY = "SELECT_PRIMARY"
    SELECT_ALTERNATE = "SELECT_ALTERNATE"
    SELECT_SINGLE = "SELECT_SINGLE"
    REJECT_WINDOW = "REJECT_WINDOW"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class SemanticCandidateDecision(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    selection_role: SelectionRole
    eligible: bool
    semantic_decision_score: float = Field(ge=0, le=1)
    decision: CandidateDecision
    reason_codes: list[str] = Field(default_factory=list)


class SemanticWindowDecision(BaseModel):
    stable_window_id: int = Field(ge=1)
    primary_candidate_id: int | None = Field(default=None, ge=1)
    alternate_candidate_id: int | None = Field(default=None, ge=1)
    selected_candidate_id: int | None = Field(default=None, ge=1)
    decision: WindowDecisionType
    score_gap: float = Field(default=0.0, ge=0, le=1)
    is_semantically_ambiguous: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class SemanticSelectionStats(BaseModel):
    window_count: int = Field(default=0, ge=0)
    selected_window_count: int = Field(default=0, ge=0)
    rejected_window_count: int = Field(default=0, ge=0)
    primary_selected_count: int = Field(default=0, ge=0)
    alternate_selected_count: int = Field(default=0, ge=0)
    candidate_keep_count: int = Field(default=0, ge=0)
    candidate_reject_count: int = Field(default=0, ge=0)
    semantic_ambiguous_window_count: int = Field(default=0, ge=0)
    rejected_incomplete_count: int = Field(default=0, ge=0)
    rejected_transition_count: int = Field(default=0, ge=0)
    rejected_low_value_count: int = Field(default=0, ge=0)
    rejected_low_confidence_count: int = Field(default=0, ge=0)
    mean_selected_semantic_score: float = Field(default=0.0, ge=0, le=1)
    median_selected_semantic_score: float = Field(default=0.0, ge=0, le=1)
    alternate_switch_rate: float = Field(default=0.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class SemanticSelectionsManifest(BaseModel):
    algorithm_version: str = SEMANTIC_DECISION_ALGORITHM_VERSION
    semantic_results_fingerprint: str
    ranked_candidates_fingerprint: str
    candidate_selections_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: SemanticSelectionStats
    candidate_decisions: list[SemanticCandidateDecision]
    window_decisions: list[SemanticWindowDecision]
    selected_candidate_ids: list[int]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Phase6ResumeStage(str, Enum):
    SEMANTIC_INPUT_PREPARATION = "SEMANTIC_INPUT_PREPARATION"
    TEMPORAL_CONTEXT_CONSTRUCTION = "TEMPORAL_CONTEXT_CONSTRUCTION"
    SEMANTIC_ANALYSIS = "SEMANTIC_ANALYSIS"
    SEMANTIC_DECISION = "SEMANTIC_DECISION"
    PHASE6_READY = "PHASE6_READY"


class CandidateCacheState(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    STALE = "STALE"
    CORRUPT = "CORRUPT"
    FAILED = "FAILED"


class CandidateCacheCheck(BaseModel):
    candidate_id: int = Field(ge=1)
    state: CandidateCacheState
    reason: str | None = None


class Phase6ResumePlan(BaseModel):
    resume_stage: Phase6ResumeStage
    reuse_semantic_input: bool = False
    reuse_temporal_context: bool = False
    candidate_ids_to_analyze: list[int] = Field(default_factory=list)
    candidate_ids_to_reparse: list[int] = Field(default_factory=list)
    rebuild_semantic_aggregate: bool = False
    rerun_decision_engine: bool = False


class SemanticCacheSnapshot(BaseModel):
    semantic_input: AnalysisArtifactCheck
    temporal_context: AnalysisArtifactCheck
    semantic_analysis: AnalysisArtifactCheck
    semantic_decision: AnalysisArtifactCheck
    semantic_candidates_ready: AnalysisArtifactCheck
    candidate_checks: list[CandidateCacheCheck] = Field(default_factory=list)
    resume_plan: Phase6ResumePlan


class Phase6Timings(BaseModel):
    semantic_input_seconds: float = Field(default=0.0, ge=0)
    temporal_context_seconds: float = Field(default=0.0, ge=0)
    semantic_inference_seconds: float = Field(default=0.0, ge=0)
    semantic_decision_seconds: float = Field(default=0.0, ge=0)
    total_phase6_seconds: float = Field(default=0.0, ge=0)


class SemanticSummary(BaseModel):
    semantic_input_count: int = Field(default=0, ge=0)
    semantic_analysis_count: int = Field(default=0, ge=0)
    stable_window_count: int = Field(default=0, ge=0)
    selected_candidate_count: int = Field(default=0, ge=0)
    rejected_window_count: int = Field(default=0, ge=0)
    primary_selected_count: int = Field(default=0, ge=0)
    alternate_selected_count: int = Field(default=0, ge=0)
    semantic_ambiguous_window_count: int = Field(default=0, ge=0)
    mean_selected_semantic_score: float = Field(default=0.0, ge=0, le=1)
    model_usage_counts: dict[str, int] = Field(default_factory=dict)
    fallback_candidate_count: int = Field(default=0, ge=0)
    mixed_models_used: bool = False
    semantic_input_fingerprint: str
    temporal_contexts_fingerprint: str
    semantic_results_fingerprint: str
    semantic_selections_fingerprint: str
    timings: Phase6Timings
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Phase7CandidateHandoff(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    candidate_timestamp_seconds: float = Field(ge=0)
    analysis_frame_path: str
    content_type: SemanticContentType
    completion_state: CompletionState
    educational_usefulness: EducationalUsefulness
    semantic_decision_score: float = Field(ge=0, le=1)
    selection_role: SelectionRole
    semantic_window_decision: WindowDecisionType


class Phase6EvaluationReport(BaseModel):
    semantic_input_count: int = Field(default=0, ge=0)
    primary_input_count: int = Field(default=0, ge=0)
    alternate_input_count: int = Field(default=0, ge=0)
    triplet_count: int = Field(default=0, ge=0)
    current_only_count: int = Field(default=0, ge=0)
    completion_distribution: dict[str, int] = Field(default_factory=dict)
    usefulness_distribution: dict[str, int] = Field(default_factory=dict)
    content_type_distribution: dict[str, int] = Field(default_factory=dict)
    selected_windows: int = Field(default=0, ge=0)
    rejected_windows: int = Field(default=0, ge=0)
    primary_selected: int = Field(default=0, ge=0)
    alternate_selected: int = Field(default=0, ge=0)
    alternate_switch_rate: float = Field(default=0.0, ge=0, le=1)
    mean_confidence: float = Field(default=0.0, ge=0, le=1)
    mean_inference_seconds: float = Field(default=0.0, ge=0)
    fallback_candidate_count: int = Field(default=0, ge=0)
    cache_hit_ratio: float = Field(default=0.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

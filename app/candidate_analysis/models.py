from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.video_analysis.models import AnalysisArtifactCheck, TimelineState


class StabilityRejectionReason(str, Enum):
    TOO_SHORT = "TOO_SHORT"
    TOO_MANY_INVALID_FRAMES = "TOO_MANY_INVALID_FRAMES"
    TOO_MANY_BLACK_FRAMES = "TOO_MANY_BLACK_FRAMES"
    INSUFFICIENT_VALID_COMPARISONS = "INSUFFICIENT_VALID_COMPARISONS"
    INVALID_TIMESTAMPS = "INVALID_TIMESTAMPS"
    LOW_QUALITY = "LOW_QUALITY"


class StabilityWindow(BaseModel):
    window_id: int = Field(ge=1)
    source_segment_id: int = Field(ge=1)
    start_timestamp_seconds: float = Field(ge=0)
    end_timestamp_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    start_frame_index: int | None = Field(default=None, ge=1)
    end_frame_index: int | None = Field(default=None, ge=1)
    comparison_count: int = Field(default=0, ge=0)
    mean_difference_score: float = Field(default=0.0, ge=0, le=1)
    median_difference_score: float = Field(default=0.0, ge=0, le=1)
    max_difference_score: float = Field(default=0.0, ge=0, le=1)
    difference_stddev: float = Field(default=0.0, ge=0)
    stability_score: float = Field(default=0.0, ge=0, le=1)
    quality_score: float = Field(default=0.0, ge=0, le=1)
    window_confidence: float = Field(default=0.0, ge=0, le=1)
    frame_count: int = Field(default=0, ge=0)
    valid_frame_ratio: float = Field(default=0.0, ge=0, le=1)
    black_frame_ratio: float = Field(default=0.0, ge=0, le=1)
    blurry_frame_ratio: float = Field(default=0.0, ge=0, le=1)
    low_information_ratio: float = Field(default=0.0, ge=0, le=1)
    average_sharpness: float = Field(default=0.0, ge=0)
    average_brightness: float = Field(default=0.0, ge=0, le=1)
    average_contrast: float = Field(default=0.0, ge=0, le=1)
    average_edge_density: float = Field(default=0.0, ge=0, le=1)
    previous_segment_state: TimelineState | None = None
    previous_segment_start_seconds: float | None = Field(default=None, ge=0)
    previous_segment_end_seconds: float | None = Field(default=None, ge=0)
    previous_segment_duration_seconds: float | None = Field(default=None, ge=0)
    next_segment_state: TimelineState | None = None
    next_segment_start_seconds: float | None = Field(default=None, ge=0)
    next_segment_end_seconds: float | None = Field(default=None, ge=0)
    seconds_since_previous_major_transition: float | None = Field(default=None, ge=0)
    seconds_until_next_major_transition: float | None = Field(default=None, ge=0)
    is_valid: bool = True
    rejection_reasons: list[StabilityRejectionReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_interval(self) -> "StabilityWindow":
        if self.end_timestamp_seconds < self.start_timestamp_seconds:
            raise ValueError("Stability window end must be >= start")
        expected = self.end_timestamp_seconds - self.start_timestamp_seconds
        if abs(expected - self.duration_seconds) > 1e-3:
            raise ValueError("Stability window duration does not match timestamps")
        return self


class StabilityWindowStats(BaseModel):
    total_stable_segments: int = Field(default=0, ge=0)
    valid_stability_windows: int = Field(default=0, ge=0)
    rejected_stability_windows: int = Field(default=0, ge=0)
    total_stable_duration_seconds: float = Field(default=0.0, ge=0)
    valid_stability_duration_seconds: float = Field(default=0.0, ge=0)
    mean_window_duration_seconds: float = Field(default=0.0, ge=0)
    median_window_duration_seconds: float = Field(default=0.0, ge=0)
    p75_window_duration_seconds: float = Field(default=0.0, ge=0)
    p90_window_duration_seconds: float = Field(default=0.0, ge=0)
    mean_stability_score: float = Field(default=0.0, ge=0, le=1)
    median_stability_score: float = Field(default=0.0, ge=0, le=1)
    p90_stability_score: float = Field(default=0.0, ge=0, le=1)
    rejected_too_short: int = Field(default=0, ge=0)
    rejected_black: int = Field(default=0, ge=0)
    rejected_invalid: int = Field(default=0, ge=0)


class StabilityWindowsManifest(BaseModel):
    algorithm_version: str
    timeline_fingerprint: str
    differences_fingerprint: str
    preprocessing_fingerprint: str
    major_changes_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: StabilityWindowStats
    windows: list[StabilityWindow]
    detection_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BoundaryType(str, Enum):
    CHANGING_TO_STABLE = "CHANGING_TO_STABLE"
    MAJOR_TRANSITION_TO_STABLE = "MAJOR_TRANSITION_TO_STABLE"
    BLACK_TRANSITION_TO_STABLE = "BLACK_TRANSITION_TO_STABLE"
    STABLE_AT_VIDEO_START = "STABLE_AT_VIDEO_START"
    INVALID_TO_STABLE = "INVALID_TO_STABLE"
    OTHER_TO_STABLE = "OTHER_TO_STABLE"


class BoundaryRejectionReason(str, Enum):
    NO_MEANINGFUL_ACTIVITY_DROP = "NO_MEANINGFUL_ACTIVITY_DROP"
    STABLE_WINDOW_TOO_WEAK = "STABLE_WINDOW_TOO_WEAK"
    STABLE_WINDOW_LOW_QUALITY = "STABLE_WINDOW_LOW_QUALITY"
    INSUFFICIENT_PRE_BOUNDARY_DATA = "INSUFFICIENT_PRE_BOUNDARY_DATA"
    INSUFFICIENT_POST_BOUNDARY_DATA = "INSUFFICIENT_POST_BOUNDARY_DATA"
    INVALID_CONTEXT = "INVALID_CONTEXT"
    UNTRUSTED_PREVIOUS_STATE = "UNTRUSTED_PREVIOUS_STATE"
    BOUNDARY_SCORE_TOO_LOW = "BOUNDARY_SCORE_TOO_LOW"


class StableBoundary(BaseModel):
    boundary_id: int = Field(ge=1)
    boundary_type: BoundaryType
    timestamp_seconds: float = Field(ge=0)
    previous_segment_start_seconds: float | None = Field(default=None, ge=0)
    previous_segment_end_seconds: float | None = Field(default=None, ge=0)
    previous_segment_duration_seconds: float | None = Field(default=None, ge=0)
    stable_window_id: int = Field(ge=1)
    stable_window_start_seconds: float = Field(ge=0)
    stable_window_end_seconds: float = Field(ge=0)
    last_changing_frame_index: int | None = Field(default=None, ge=1)
    first_stable_frame_index: int | None = Field(default=None, ge=1)
    pre_boundary_mean_difference: float = Field(default=0.0, ge=0, le=1)
    pre_boundary_max_difference: float = Field(default=0.0, ge=0, le=1)
    post_boundary_mean_difference: float = Field(default=0.0, ge=0, le=1)
    post_boundary_max_difference: float = Field(default=0.0, ge=0, le=1)
    activity_drop_score: float = Field(default=0.0, ge=0, le=1)
    preceding_activity_score: float = Field(default=0.0, ge=0, le=1)
    preceding_changing_duration_seconds: float = Field(default=0.0, ge=0)
    stability_score: float = Field(default=0.0, ge=0, le=1)
    quality_score: float = Field(default=0.0, ge=0, le=1)
    boundary_score: float = Field(default=0.0, ge=0, le=1)
    opportunity_duration_seconds: float = Field(default=0.0, ge=0)
    stable_exit_timestamp_seconds: float = Field(ge=0)
    is_valid: bool = True
    rejection_reasons: list[BoundaryRejectionReason] = Field(default_factory=list)


class BoundaryStats(BaseModel):
    total_boundaries: int = Field(default=0, ge=0)
    valid_boundaries: int = Field(default=0, ge=0)
    rejected_boundaries: int = Field(default=0, ge=0)
    changing_to_stable_count: int = Field(default=0, ge=0)
    major_transition_to_stable_count: int = Field(default=0, ge=0)
    black_transition_to_stable_count: int = Field(default=0, ge=0)
    stable_at_start_count: int = Field(default=0, ge=0)
    mean_boundary_score: float = Field(default=0.0, ge=0, le=1)
    median_boundary_score: float = Field(default=0.0, ge=0, le=1)
    p90_boundary_score: float = Field(default=0.0, ge=0, le=1)
    p95_boundary_score: float = Field(default=0.0, ge=0, le=1)
    mean_activity_drop_score: float = Field(default=0.0, ge=0, le=1)


class BoundariesManifest(BaseModel):
    algorithm_version: str
    stability_windows_fingerprint: str
    timeline_fingerprint: str
    differences_fingerprint: str
    major_changes_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: BoundaryStats
    boundaries: list[StableBoundary]
    detection_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandidateType(str, Enum):
    SETTLED_START = "SETTLED_START"
    EARLY_STABLE = "EARLY_STABLE"
    MID_STABLE = "MID_STABLE"
    LATE_STABLE = "LATE_STABLE"
    PRE_EXIT = "PRE_EXIT"


class CandidateSourceType(str, Enum):
    BOUNDARY = "BOUNDARY"
    WINDOW_ONLY = "WINDOW_ONLY"


class CandidateRejectionReason(str, Enum):
    NO_VALID_FRAME_NEAR_TIMESTAMP = "NO_VALID_FRAME_NEAR_TIMESTAMP"
    BLACK_FRAME = "BLACK_FRAME"
    INVALID_FRAME = "INVALID_FRAME"
    OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    INSUFFICIENT_SPACING = "INSUFFICIENT_SPACING"
    GLOBAL_CANDIDATE_CAP = "GLOBAL_CANDIDATE_CAP"


class GeneratedCandidate(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    boundary_id: int | None = Field(default=None, ge=1)
    candidate_type: CandidateType
    source_type: CandidateSourceType
    boundary_type: BoundaryType | None = None
    requested_timestamp_seconds: float = Field(ge=0)
    frame_timestamp_seconds: float | None = Field(default=None, ge=0)
    frame_index: int | None = Field(default=None, ge=1)
    sampled_frame_path: str | None = None
    processed_frame_path: str | None = None
    stable_window_start_seconds: float = Field(ge=0)
    stable_window_end_seconds: float = Field(ge=0)
    stable_window_duration_seconds: float = Field(ge=0)
    relative_position: float = Field(default=0.0, ge=0, le=1)
    seconds_after_boundary: float | None = Field(default=None, ge=0)
    seconds_before_stable_exit: float = Field(default=0.0, ge=0)
    boundary_score: float = Field(default=0.0, ge=0, le=1)
    activity_drop_score: float = Field(default=0.0, ge=0, le=1)
    preceding_activity_score: float = Field(default=0.0, ge=0, le=1)
    preceding_changing_duration_seconds: float = Field(default=0.0, ge=0)
    stability_score: float = Field(default=0.0, ge=0, le=1)
    window_quality_score: float = Field(default=0.0, ge=0, le=1)
    brightness: float | None = Field(default=None, ge=0, le=1)
    contrast: float | None = Field(default=None, ge=0, le=1)
    sharpness_score: float | None = Field(default=None, ge=0)
    edge_density: float | None = Field(default=None, ge=0, le=1)
    frame_quality_score: float | None = Field(default=None, ge=0, le=1)
    is_blurry: bool = False
    is_black: bool = False
    is_low_information: bool = False
    is_valid: bool = True
    rejection_reasons: list[CandidateRejectionReason] = Field(default_factory=list)


class CandidateGenerationStats(BaseModel):
    valid_boundaries: int = Field(default=0, ge=0)
    windows_considered: int = Field(default=0, ge=0)
    candidates_generated: int = Field(default=0, ge=0)
    valid_candidates: int = Field(default=0, ge=0)
    rejected_candidates: int = Field(default=0, ge=0)
    mean_candidates_per_window: float = Field(default=0.0, ge=0)
    candidates_per_minute: float = Field(default=0.0, ge=0)
    shifted_to_nearby_frame_count: int = Field(default=0, ge=0)
    candidate_type_distribution: dict[str, int] = Field(default_factory=dict)


class GeneratedCandidatesManifest(BaseModel):
    algorithm_version: str
    stability_windows_fingerprint: str
    boundaries_fingerprint: str
    sampling_fingerprint: str
    preprocessing_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: CandidateGenerationStats
    candidates: list[GeneratedCandidate]
    generation_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HeuristicRejectionReason(str, Enum):
    INVALID_FRAME = "INVALID_FRAME"
    BLACK_FRAME = "BLACK_FRAME"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    EXTREME_BLUR = "EXTREME_BLUR"
    VERY_LOW_VISUAL_QUALITY = "VERY_LOW_VISUAL_QUALITY"


class ScoredCandidate(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    boundary_id: int | None = Field(default=None, ge=1)
    candidate_type: CandidateType
    source_type: CandidateSourceType
    frame_timestamp_seconds: float | None = Field(default=None, ge=0)
    relative_position: float = Field(ge=0, le=1)
    boundary_score: float = Field(default=0.0, ge=0, le=1)
    activity_drop_score: float = Field(default=0.0, ge=0, le=1)
    stability_score: float = Field(default=0.0, ge=0, le=1)
    window_quality_score: float = Field(default=0.0, ge=0, le=1)
    visual_quality_score: float = Field(default=0.0, ge=0, le=1)
    content_density_score: float = Field(default=0.0, ge=0, le=1)
    local_stability_score: float = Field(default=0.0, ge=0, le=1)
    transition_risk_score: float = Field(default=0.0, ge=0, le=1)
    transition_safety_score: float = Field(default=0.0, ge=0, le=1)
    preceding_activity_strength: float = Field(default=0.0, ge=0, le=1)
    content_accumulation_score: float = Field(default=0.0, ge=0, le=1)
    relative_content_gain: float = Field(default=0.0, ge=0, le=1)
    structural_progress_score: float = Field(default=0.0, ge=0, le=1)
    positional_completeness_score: float = Field(default=0.0, ge=0, le=1)
    completeness_heuristic_score: float = Field(default=0.0, ge=0, le=1)
    is_valid: bool = True
    rejection_reasons: list[HeuristicRejectionReason] = Field(default_factory=list)


class CandidateHeuristicStats(BaseModel):
    total_candidates: int = Field(default=0, ge=0)
    valid_candidates: int = Field(default=0, ge=0)
    rejected_candidates: int = Field(default=0, ge=0)
    mean_visual_quality: float = Field(default=0.0, ge=0, le=1)
    mean_content_density: float = Field(default=0.0, ge=0, le=1)
    mean_completeness: float = Field(default=0.0, ge=0, le=1)
    mean_transition_risk: float = Field(default=0.0, ge=0, le=1)
    p50_completeness: float = Field(default=0.0, ge=0, le=1)
    p75_completeness: float = Field(default=0.0, ge=0, le=1)
    p90_completeness: float = Field(default=0.0, ge=0, le=1)
    p95_completeness: float = Field(default=0.0, ge=0, le=1)
    candidate_type_mean_completeness: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ScoredCandidatesManifest(BaseModel):
    algorithm_version: str
    generated_candidates_fingerprint: str
    boundaries_fingerprint: str
    stability_windows_fingerprint: str
    preprocessing_fingerprint: str
    differences_fingerprint: str
    timeline_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: CandidateHeuristicStats
    candidates: list[ScoredCandidate]
    scoring_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SelectionRole(str, Enum):
    NONE = "NONE"
    PRIMARY = "PRIMARY"
    ALTERNATE = "ALTERNATE"


class RankedCandidate(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    ranking_score: float = Field(ge=0, le=1)
    rank_within_window: int = Field(ge=1)
    selection_role: SelectionRole = SelectionRole.NONE
    score_gap_from_best: float = Field(default=0.0, ge=0, le=1)
    candidate_type: CandidateType
    source_type: CandidateSourceType
    frame_timestamp_seconds: float = Field(ge=0)
    relative_position: float = Field(ge=0, le=1)
    visual_quality_score: float = Field(ge=0, le=1)
    completeness_heuristic_score: float = Field(ge=0, le=1)
    transition_safety_score: float = Field(ge=0, le=1)
    local_stability_score: float = Field(ge=0, le=1)
    boundary_score: float = Field(ge=0, le=1)
    content_density_score: float = Field(ge=0, le=1)
    content_accumulation_score: float = Field(ge=0, le=1)


class WindowSelection(BaseModel):
    stable_window_id: int = Field(ge=1)
    boundary_id: int | None = Field(default=None, ge=1)
    candidate_count: int = Field(default=0, ge=0)
    valid_candidate_count: int = Field(default=0, ge=0)
    primary_candidate_id: int | None = Field(default=None, ge=1)
    alternate_candidate_ids: list[int] = Field(default_factory=list)
    primary_score: float = Field(default=0.0, ge=0, le=1)
    second_best_score: float | None = Field(default=None, ge=0, le=1)
    score_gap: float = Field(default=0.0, ge=0, le=1)
    selection_confidence: float = Field(default=0.0, ge=0, le=1)
    is_ambiguous: bool = False
    clear_winner: bool = False
    failure_reason: str | None = None


class RankingStats(BaseModel):
    windows_considered: int = Field(default=0, ge=0)
    windows_with_primary: int = Field(default=0, ge=0)
    primary_candidates: int = Field(default=0, ge=0)
    alternate_candidates: int = Field(default=0, ge=0)
    ambiguous_windows: int = Field(default=0, ge=0)
    clear_winner_windows: int = Field(default=0, ge=0)
    windows_without_candidate: int = Field(default=0, ge=0)
    mean_ranking_score: float = Field(default=0.0, ge=0, le=1)
    median_ranking_score: float = Field(default=0.0, ge=0, le=1)
    p90_ranking_score: float = Field(default=0.0, ge=0, le=1)
    mean_score_gap: float = Field(default=0.0, ge=0, le=1)
    ambiguity_ratio: float = Field(default=0.0, ge=0, le=1)
    winner_type_distribution: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RankedCandidatesManifest(BaseModel):
    algorithm_version: str
    scored_candidates_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: RankingStats
    candidates: list[RankedCandidate]
    ranking_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SelectionsManifest(BaseModel):
    algorithm_version: str
    ranked_candidates_fingerprint: str
    scored_candidates_fingerprint: str
    stability_windows_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: RankingStats
    windows: list[WindowSelection]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandidateResumeStage(str, Enum):
    STABILITY_WINDOWS = "STABILITY_WINDOWS"
    BOUNDARY_DETECTION = "BOUNDARY_DETECTION"
    CANDIDATE_GENERATION = "CANDIDATE_GENERATION"
    CANDIDATE_HEURISTICS = "CANDIDATE_HEURISTICS"
    CANDIDATE_RANKING = "CANDIDATE_RANKING"
    PHASE4_READY = "PHASE4_READY"


class CandidateCacheSnapshot(BaseModel):
    stability_windows: AnalysisArtifactCheck
    boundaries: AnalysisArtifactCheck
    generated_candidates: AnalysisArtifactCheck
    scored_candidates: AnalysisArtifactCheck
    ranking: AnalysisArtifactCheck
    candidates_ready: AnalysisArtifactCheck
    resume_stage: CandidateResumeStage


class Phase4Timings(BaseModel):
    stability_window_seconds: float = Field(default=0.0, ge=0)
    boundary_detection_seconds: float = Field(default=0.0, ge=0)
    candidate_generation_seconds: float = Field(default=0.0, ge=0)
    heuristic_scoring_seconds: float = Field(default=0.0, ge=0)
    ranking_seconds: float = Field(default=0.0, ge=0)
    total_phase4_seconds: float = Field(default=0.0, ge=0)


class CandidateAnalysisSummary(BaseModel):
    stability_window_count: int = Field(default=0, ge=0)
    valid_stability_window_count: int = Field(default=0, ge=0)
    valid_boundary_count: int = Field(default=0, ge=0)
    generated_candidate_count: int = Field(default=0, ge=0)
    valid_candidate_count: int = Field(default=0, ge=0)
    primary_candidate_count: int = Field(default=0, ge=0)
    alternate_candidate_count: int = Field(default=0, ge=0)
    ambiguous_window_count: int = Field(default=0, ge=0)
    clear_winner_count: int = Field(default=0, ge=0)
    window_without_candidate_count: int = Field(default=0, ge=0)
    primary_candidates_per_minute: float = Field(default=0.0, ge=0)
    generated_candidates_per_minute: float = Field(default=0.0, ge=0)
    timings: Phase4Timings
    stability_windows_fingerprint: str
    boundaries_fingerprint: str
    generated_candidates_fingerprint: str
    scored_candidates_fingerprint: str
    ranked_candidates_fingerprint: str
    selections_fingerprint: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandidateHandoff(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    boundary_id: int | None = Field(default=None, ge=1)
    timestamp_seconds: float = Field(ge=0)
    frame_index: int = Field(ge=1)
    sampled_frame_path: str
    processed_frame_path: str
    selection_role: SelectionRole
    ranking_score: float = Field(ge=0, le=1)
    selection_confidence: float = Field(ge=0, le=1)
    is_ambiguous: bool
    boundary_score: float = Field(ge=0, le=1)
    completeness_heuristic_score: float = Field(ge=0, le=1)
    visual_quality_score: float = Field(ge=0, le=1)
    transition_safety_score: float = Field(ge=0, le=1)


class Phase4EvaluationReport(BaseModel):
    lecture_duration_seconds: float = Field(default=0.0, ge=0)
    stability_windows: int = Field(default=0, ge=0)
    valid_boundaries: int = Field(default=0, ge=0)
    generated_candidates: int = Field(default=0, ge=0)
    valid_candidates: int = Field(default=0, ge=0)
    primary_candidates: int = Field(default=0, ge=0)
    alternate_candidates: int = Field(default=0, ge=0)
    ambiguous_windows: int = Field(default=0, ge=0)
    clear_winners: int = Field(default=0, ge=0)
    primary_density_per_minute: float = Field(default=0.0, ge=0)
    generated_density_per_minute: float = Field(default=0.0, ge=0)
    boundary_type_distribution: dict[str, int] = Field(default_factory=dict)
    candidate_type_distribution: dict[str, int] = Field(default_factory=dict)
    winner_type_distribution: dict[str, int] = Field(default_factory=dict)
    mean_window_duration_seconds: float = Field(default=0.0, ge=0)
    mean_boundary_score: float = Field(default=0.0, ge=0, le=1)
    mean_activity_drop: float = Field(default=0.0, ge=0, le=1)
    mean_visual_quality: float = Field(default=0.0, ge=0, le=1)
    mean_completeness: float = Field(default=0.0, ge=0, le=1)
    mean_transition_risk: float = Field(default=0.0, ge=0, le=1)
    mean_primary_ranking_score: float = Field(default=0.0, ge=0, le=1)
    mean_score_gap: float = Field(default=0.0, ge=0, le=1)
    ambiguity_ratio: float = Field(default=0.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

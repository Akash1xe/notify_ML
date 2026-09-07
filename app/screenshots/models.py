from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.semantic_analysis.models import CompletionState, SemanticContentType
from app.video_analysis.models import AnalysisArtifactCheck


SOURCE_SCREENSHOT_EXTRACTION_ALGORITHM_VERSION = "1"
SCREENSHOT_QUALITY_ALGORITHM_VERSION = "1"
VISUAL_FINGERPRINT_ALGORITHM_VERSION = "1"
CROSS_WINDOW_DUPLICATE_ALGORITHM_VERSION = "1"
FINAL_SCREENSHOT_SELECTION_ALGORITHM_VERSION = "1"


class ScreenshotExtractionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    CORRUPT = "CORRUPT"
    STALE = "STALE"


class ExposureState(str, Enum):
    NORMAL = "NORMAL"
    UNDEREXPOSED = "UNDEREXPOSED"
    OVEREXPOSED = "OVEREXPOSED"
    UNCERTAIN = "UNCERTAIN"


class ScreenshotQualityState(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    INVALID = "INVALID"


class PairDecision(str, Enum):
    DUPLICATE = "DUPLICATE"
    POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"
    DISTINCT = "DISTINCT"
    NOT_COMPARED = "NOT_COMPARED"


class FinalMemberDecision(str, Enum):
    WINNER = "WINNER"
    DUPLICATE_SUPPRESSED = "DUPLICATE_SUPPRESSED"
    INELIGIBLE = "INELIGIBLE"


class Phase7CacheState(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    STALE = "STALE"
    CORRUPT = "CORRUPT"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class Phase7ResumeStage(str, Enum):
    SOURCE_EXTRACTION = "SOURCE_EXTRACTION"
    QUALITY_VALIDATION = "QUALITY_VALIDATION"
    FINGERPRINT_GENERATION = "FINGERPRINT_GENERATION"
    DUPLICATE_DETECTION = "DUPLICATE_DETECTION"
    FINAL_SCREENSHOT_SELECTION = "FINAL_SCREENSHOT_SELECTION"
    PHASE7_READY = "PHASE7_READY"


class SourceScreenshotRecord(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    requested_timestamp_seconds: float = Field(ge=0)
    resolved_timestamp_seconds: float | None = Field(default=None, ge=0)
    source_video_relative_path: str
    screenshot_relative_path: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    format: str = "png"
    file_size_bytes: int = Field(ge=1)
    file_sha256: str
    source_fingerprint: str
    extraction_config_fingerprint: str
    extraction_fingerprint: str
    extraction_status: ScreenshotExtractionStatus = ScreenshotExtractionStatus.SUCCESS
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExtractionStats(BaseModel):
    selected_candidate_count: int = Field(default=0, ge=0)
    successful_extraction_count: int = Field(default=0, ge=0)
    failed_extraction_count: int = Field(default=0, ge=0)
    cached_extraction_count: int = Field(default=0, ge=0)
    total_output_bytes: int = Field(default=0, ge=0)
    min_width: int | None = Field(default=None, ge=1)
    max_width: int | None = Field(default=None, ge=1)
    min_height: int | None = Field(default=None, ge=1)
    max_height: int | None = Field(default=None, ge=1)
    unique_resolutions: list[str] = Field(default_factory=list)
    first_selected_timestamp: float | None = Field(default=None, ge=0)
    last_selected_timestamp: float | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ExtractionManifest(BaseModel):
    algorithm_version: str = SOURCE_SCREENSHOT_EXTRACTION_ALGORITHM_VERSION
    semantic_selections_fingerprint: str
    source_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: ExtractionStats
    screenshots: list[SourceScreenshotRecord]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScreenshotQualityMetrics(BaseModel):
    sharpness_raw: float = Field(ge=0)
    sharpness_score: float = Field(ge=0, le=1)
    brightness_mean: float = Field(ge=0, le=1)
    brightness_std: float = Field(ge=0, le=1)
    contrast_score: float = Field(ge=0, le=1)
    black_ratio: float = Field(ge=0, le=1)
    white_ratio: float = Field(ge=0, le=1)
    edge_density: float = Field(ge=0, le=1)
    content_density: float = Field(ge=0, le=1)
    exposure_state: ExposureState
    is_black_or_blank: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False
    is_blurry: bool = False
    quality_score: float = Field(ge=0, le=1)
    quality_state: ScreenshotQualityState


class QualityAttemptRecord(BaseModel):
    timestamp_seconds: float = Field(ge=0)
    distance_seconds: float = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)
    quality_state: ScreenshotQualityState
    selected: bool = False


class QualitySelectionRecord(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    semantic_timestamp_seconds: float = Field(ge=0)
    selected_timestamp_seconds: float = Field(ge=0)
    timestamp_shift_seconds: float
    exact_quality_score: float = Field(ge=0, le=1)
    selected_quality_score: float = Field(ge=0, le=1)
    selected_relative_path: str
    selected_file_sha256: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    quality_state: ScreenshotQualityState
    metrics: ScreenshotQualityMetrics
    exact_metrics: ScreenshotQualityMetrics
    used_timestamp_fallback: bool = False
    extraction_fingerprint: str
    quality_config_fingerprint: str
    quality_fingerprint: str
    attempts: list[QualityAttemptRecord] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class QualityStats(BaseModel):
    candidate_count: int = Field(default=0, ge=0)
    exact_accepted_count: int = Field(default=0, ge=0)
    fallback_search_count: int = Field(default=0, ge=0)
    fallback_selected_count: int = Field(default=0, ge=0)
    poor_remaining_count: int = Field(default=0, ge=0)
    invalid_count: int = Field(default=0, ge=0)
    cached_quality_count: int = Field(default=0, ge=0)
    mean_absolute_timestamp_shift: float = Field(default=0.0, ge=0)
    median_absolute_timestamp_shift: float = Field(default=0.0, ge=0)
    max_absolute_timestamp_shift: float = Field(default=0.0, ge=0)
    mean_exact_quality_score: float = Field(default=0.0, ge=0, le=1)
    mean_selected_quality_score: float = Field(default=0.0, ge=0, le=1)
    median_selected_quality_score: float = Field(default=0.0, ge=0, le=1)
    mean_quality_improvement: float = 0.0
    max_quality_improvement: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class QualityManifest(BaseModel):
    algorithm_version: str = SCREENSHOT_QUALITY_ALGORITHM_VERSION
    extraction_manifest_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: QualityStats
    screenshots: list[QualitySelectionRecord]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VisualFingerprintRecord(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    selected_timestamp_seconds: float = Field(ge=0)
    screenshot_relative_path: str
    source_file_sha256: str
    phash: str
    dhash: str
    edge_hash: str
    thumbnail_relative_path: str
    thumbnail_file_sha256: str
    edge_map_relative_path: str
    edge_map_file_sha256: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fingerprint_config_fingerprint: str
    visual_fingerprint: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FingerprintStats(BaseModel):
    candidate_count: int = Field(default=0, ge=0)
    fingerprint_count: int = Field(default=0, ge=0)
    cached_fingerprint_count: int = Field(default=0, ge=0)
    failed_fingerprint_count: int = Field(default=0, ge=0)
    identical_phash_count: int = Field(default=0, ge=0)
    identical_dhash_count: int = Field(default=0, ge=0)
    total_thumbnail_bytes: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class FingerprintManifest(BaseModel):
    algorithm_version: str = VISUAL_FINGERPRINT_ALGORITHM_VERSION
    quality_manifest_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: FingerprintStats
    fingerprints: list[VisualFingerprintRecord]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScreenshotPairSimilarity(BaseModel):
    candidate_a_id: int = Field(ge=1)
    candidate_b_id: int = Field(ge=1)
    timestamp_gap_seconds: float = Field(ge=0)
    phash_distance: float = Field(ge=0, le=1)
    dhash_distance: float = Field(ge=0, le=1)
    thumbnail_ssim: float = Field(ge=0, le=1)
    edge_similarity: float = Field(ge=0, le=1)
    changed_pixel_ratio: float = Field(ge=0, le=1)
    a_to_b_edge_addition_ratio: float = Field(ge=0, le=1)
    b_to_a_edge_addition_ratio: float = Field(ge=0, le=1)
    content_type_compatible: bool
    content_compatibility_score: float = Field(ge=0, le=1)
    duplicate_score: float = Field(ge=0, le=1)
    decision: PairDecision
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def canonical_pair(self) -> "ScreenshotPairSimilarity":
        if self.candidate_a_id >= self.candidate_b_id:
            raise ValueError("pair candidate IDs must be in ascending canonical order")
        return self


class DuplicatePairStats(BaseModel):
    candidate_count: int = Field(default=0, ge=0)
    comparison_pair_count: int = Field(default=0, ge=0)
    duplicate_pair_count: int = Field(default=0, ge=0)
    possible_duplicate_pair_count: int = Field(default=0, ge=0)
    distinct_pair_count: int = Field(default=0, ge=0)


class DuplicatePairsManifest(BaseModel):
    algorithm_version: str = CROSS_WINDOW_DUPLICATE_ALGORITHM_VERSION
    fingerprint_manifest_fingerprint: str
    content_types_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: DuplicatePairStats
    pairs: list[ScreenshotPairSimilarity]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DuplicateGroup(BaseModel):
    group_id: str
    candidate_ids: list[int] = Field(min_length=1)
    representative_candidate_id: int = Field(ge=1)
    group_size: int = Field(ge=1)
    earliest_timestamp_seconds: float = Field(ge=0)
    latest_timestamp_seconds: float = Field(ge=0)
    minimum_internal_duplicate_score: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_members(self) -> "DuplicateGroup":
        if self.group_size != len(self.candidate_ids):
            raise ValueError("group_size does not match member count")
        if self.representative_candidate_id not in self.candidate_ids:
            raise ValueError("representative must belong to group")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("duplicate candidate in group")
        return self


class DuplicateGroupStats(BaseModel):
    candidate_count: int = Field(default=0, ge=0)
    duplicate_group_count: int = Field(default=0, ge=0)
    singleton_group_count: int = Field(default=0, ge=0)
    multi_member_group_count: int = Field(default=0, ge=0)
    potential_duplicate_removals: int = Field(default=0, ge=0)
    duplicate_ratio: float = Field(default=0.0, ge=0, le=1)
    mean_group_size: float = Field(default=0.0, ge=0)
    max_group_size: int = Field(default=0, ge=0)
    mean_duplicate_group_time_span: float = Field(default=0.0, ge=0)
    max_duplicate_group_time_span: float = Field(default=0.0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class DuplicateGroupsManifest(BaseModel):
    algorithm_version: str = CROSS_WINDOW_DUPLICATE_ALGORITHM_VERSION
    duplicate_pairs_fingerprint: str
    fingerprint_manifest_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: DuplicateGroupStats
    groups: list[DuplicateGroup]
    possible_duplicate_pairs: list[ScreenshotPairSimilarity] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FinalSelectionMember(BaseModel):
    candidate_id: int = Field(ge=1)
    eligible: bool
    final_selection_score: float = Field(ge=0, le=1)
    semantic_decision_score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    completion_score: float = Field(ge=0, le=1)
    semantic_confidence: float = Field(ge=0, le=1)
    sharpness_score: float = Field(ge=0, le=1)
    timestamp_proximity_score: float = Field(ge=0, le=1)
    decision: FinalMemberDecision
    reason_codes: list[str] = Field(default_factory=list)


class FinalGroupSelection(BaseModel):
    group_id: str
    winner_candidate_id: int = Field(ge=1)
    winner_relative_path: str
    winner_selected_timestamp_seconds: float = Field(ge=0)
    final_selection_score: float = Field(ge=0, le=1)
    group_size: int = Field(ge=1)
    reason_codes: list[str] = Field(default_factory=list)
    members: list[FinalSelectionMember]


class FinalScreenshotRecord(BaseModel):
    order: int = Field(ge=1)
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    duplicate_group_id: str
    timestamp_seconds: float = Field(ge=0)
    semantic_timestamp_seconds: float = Field(ge=0)
    image_relative_path: str
    file_sha256: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    content_type: SemanticContentType
    completion_state: CompletionState
    semantic_decision_score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    final_selection_score: float = Field(ge=0, le=1)


class FinalSelectionStats(BaseModel):
    duplicate_group_count: int = Field(default=0, ge=0)
    final_screenshot_count: int = Field(default=0, ge=0)
    singleton_selected_count: int = Field(default=0, ge=0)
    multi_member_group_count: int = Field(default=0, ge=0)
    suppressed_duplicate_count: int = Field(default=0, ge=0)
    invalid_member_count: int = Field(default=0, ge=0)
    final_dedup_ratio: float = Field(default=0.0, ge=0, le=1)
    mean_winner_semantic_score: float = Field(default=0.0, ge=0, le=1)
    median_winner_semantic_score: float = Field(default=0.0, ge=0, le=1)
    mean_winner_quality_score: float = Field(default=0.0, ge=0, le=1)
    median_winner_quality_score: float = Field(default=0.0, ge=0, le=1)
    mean_final_selection_score: float = Field(default=0.0, ge=0, le=1)
    median_final_selection_score: float = Field(default=0.0, ge=0, le=1)
    p10_final_selection_score: float = Field(default=0.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class FinalSelectionsManifest(BaseModel):
    algorithm_version: str = FINAL_SCREENSHOT_SELECTION_ALGORITHM_VERSION
    duplicate_groups_fingerprint: str
    semantic_dependency_fingerprint: str
    quality_manifest_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: FinalSelectionStats
    groups: list[FinalGroupSelection]
    final_screenshots: list[FinalScreenshotRecord]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Phase7CandidateCacheCheck(BaseModel):
    candidate_id: int = Field(ge=1)
    state: Phase7CacheState
    reason: str | None = None


class Phase7ResumePlan(BaseModel):
    resume_stage: Phase7ResumeStage
    candidate_ids_to_extract: list[int] = Field(default_factory=list)
    candidate_ids_to_quality_check: list[int] = Field(default_factory=list)
    candidate_ids_to_fingerprint: list[int] = Field(default_factory=list)
    rebuild_extraction_manifest: bool = False
    rebuild_quality_manifest: bool = False
    rebuild_fingerprint_manifest: bool = False
    rerun_duplicate_detection: bool = False
    rerun_final_selection: bool = False
    cleanup_orphan_candidate_ids: list[int] = Field(default_factory=list)


class Phase7CacheSnapshot(BaseModel):
    extraction: AnalysisArtifactCheck
    quality: AnalysisArtifactCheck
    fingerprints: AnalysisArtifactCheck
    duplicates: AnalysisArtifactCheck
    final_selection: AnalysisArtifactCheck
    final_screenshots_ready: AnalysisArtifactCheck
    extraction_candidates: list[Phase7CandidateCacheCheck] = Field(default_factory=list)
    quality_candidates: list[Phase7CandidateCacheCheck] = Field(default_factory=list)
    fingerprint_candidates: list[Phase7CandidateCacheCheck] = Field(default_factory=list)
    resume_plan: Phase7ResumePlan


class Phase7Timings(BaseModel):
    source_extraction_seconds: float = Field(default=0.0, ge=0)
    quality_validation_seconds: float = Field(default=0.0, ge=0)
    fingerprint_seconds: float = Field(default=0.0, ge=0)
    duplicate_detection_seconds: float = Field(default=0.0, ge=0)
    final_selection_seconds: float = Field(default=0.0, ge=0)
    total_phase7_seconds: float = Field(default=0.0, ge=0)


class Phase7Summary(BaseModel):
    semantic_candidate_count: int = Field(default=0, ge=0)
    source_extraction_count: int = Field(default=0, ge=0)
    quality_selected_count: int = Field(default=0, ge=0)
    fingerprint_count: int = Field(default=0, ge=0)
    duplicate_group_count: int = Field(default=0, ge=0)
    final_screenshot_count: int = Field(default=0, ge=0)
    suppressed_duplicate_count: int = Field(default=0, ge=0)
    exact_timestamp_winner_count: int = Field(default=0, ge=0)
    refined_timestamp_winner_count: int = Field(default=0, ge=0)
    mean_final_quality_score: float = Field(default=0.0, ge=0, le=1)
    mean_final_semantic_score: float = Field(default=0.0, ge=0, le=1)
    mean_final_selection_score: float = Field(default=0.0, ge=0, le=1)
    final_dedup_ratio: float = Field(default=0.0, ge=0, le=1)
    content_type_distribution: dict[str, int] = Field(default_factory=dict)
    cache_stats: dict[str, int | float | bool] = Field(default_factory=dict)
    extraction_manifest_fingerprint: str
    quality_manifest_fingerprint: str
    fingerprint_manifest_fingerprint: str
    duplicate_groups_fingerprint: str
    final_selections_fingerprint: str
    timings: Phase7Timings
    warnings: list[str] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Phase7EvaluationReport(BaseModel):
    semantic_candidate_count: int = Field(default=0, ge=0)
    final_screenshot_count: int = Field(default=0, ge=0)
    quality_state_distribution: dict[str, int] = Field(default_factory=dict)
    exact_retained_count: int = Field(default=0, ge=0)
    refined_timestamp_count: int = Field(default=0, ge=0)
    mean_absolute_timestamp_shift: float = Field(default=0.0, ge=0)
    duplicate_pair_distribution: dict[str, int] = Field(default_factory=dict)
    duplicate_group_count: int = Field(default=0, ge=0)
    max_duplicate_group_size: int = Field(default=0, ge=0)
    dedup_ratio: float = Field(default=0.0, ge=0, le=1)
    mean_final_quality_score: float = Field(default=0.0, ge=0, le=1)
    mean_final_semantic_score: float = Field(default=0.0, ge=0, le=1)
    quality_override_count: int = Field(default=0, ge=0)
    cache_hit_ratio: float = Field(default=0.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.candidate_analysis.models import SelectionRole
from app.video_analysis.models import AnalysisArtifactCheck


class AudioPreparationWarning(str, Enum):
    AUDIO_VIDEO_MINOR_DURATION_MISMATCH = "AUDIO_VIDEO_MINOR_DURATION_MISMATCH"
    HIGH_SILENCE_RATIO = "HIGH_SILENCE_RATIO"
    HIGH_CLIPPING_RATIO = "HIGH_CLIPPING_RATIO"
    LOW_OVERALL_ENERGY = "LOW_OVERALL_ENERGY"


class AudioPreparationRejection(str, Enum):
    AUDIO_FILE_MISSING = "AUDIO_FILE_MISSING"
    AUDIO_MANIFEST_MISSING = "AUDIO_MANIFEST_MISSING"
    AUDIO_DECODE_FAILED = "AUDIO_DECODE_FAILED"
    EMPTY_AUDIO = "EMPTY_AUDIO"
    UNSUPPORTED_AUDIO_FORMAT = "UNSUPPORTED_AUDIO_FORMAT"
    INVALID_SAMPLE_RATE = "INVALID_SAMPLE_RATE"
    INVALID_CHANNEL_COUNT = "INVALID_CHANNEL_COUNT"
    INVALID_SAMPLE_WIDTH = "INVALID_SAMPLE_WIDTH"
    AUDIO_VIDEO_DURATION_MISMATCH = "AUDIO_VIDEO_DURATION_MISMATCH"
    AUDIO_FILE_SIZE_IMPLAUSIBLE = "AUDIO_FILE_SIZE_IMPLAUSIBLE"
    AUDIO_EFFECTIVELY_SILENT = "AUDIO_EFFECTIVELY_SILENT"
    UNSAFE_AUDIO_PATH = "UNSAFE_AUDIO_PATH"


class AudioFormatInfo(BaseModel):
    codec: str = "pcm_s16le"
    container: str = "wav"
    sample_rate_hz: int = Field(ge=1)
    channels: int = Field(ge=1)
    bits_per_sample: int = Field(ge=1)
    duration_seconds: float = Field(ge=0)
    file_size_bytes: int = Field(ge=0)
    frame_count: int = Field(ge=0)


class AudioDiagnostics(BaseModel):
    mean_rms: float = Field(default=0.0, ge=0, le=1)
    median_rms: float = Field(default=0.0, ge=0, le=1)
    peak_amplitude: float = Field(default=0.0, ge=0, le=1)
    silent_window_ratio: float = Field(default=0.0, ge=0, le=1)
    clipping_ratio: float = Field(default=0.0, ge=0, le=1)
    dc_offset: float = Field(default=0.0, ge=-1, le=1)
    effectively_silent: bool = False


class TranscriptionChunk(BaseModel):
    chunk_id: int = Field(ge=1)
    logical_start_seconds: float = Field(ge=0)
    logical_end_seconds: float = Field(ge=0)
    decode_start_seconds: float = Field(ge=0)
    decode_end_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    contains_candidate_timestamps: bool = False
    candidate_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "TranscriptionChunk":
        if self.logical_start_seconds >= self.logical_end_seconds:
            raise ValueError("logical chunk start must be < end")
        if self.decode_start_seconds > self.logical_start_seconds:
            raise ValueError("decode start must be <= logical start")
        if self.decode_end_seconds < self.logical_end_seconds:
            raise ValueError("decode end must be >= logical end")
        return self


class AudioPreparationStats(BaseModel):
    chunk_count: int = Field(default=0, ge=0)
    candidate_annotated_chunk_count: int = Field(default=0, ge=0)
    duration_delta_seconds: float = Field(default=0.0, ge=0)
    duration_delta_ratio: float = Field(default=0.0, ge=0)


class TranscriptionPreparationManifest(BaseModel):
    algorithm_version: str
    audio_fingerprint: str
    media_fingerprint: str
    candidate_summary_fingerprint: str | None = None
    config_fingerprint: str
    artifact_fingerprint: str
    audio_path: str
    audio: AudioFormatInfo
    diagnostics: AudioDiagnostics
    video_duration_seconds: float = Field(ge=0)
    stats: AudioPreparationStats
    warnings: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    is_valid: bool = True
    chunks: list[TranscriptionChunk]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RawWord(BaseModel):
    word: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    probability: float | None = Field(default=None, ge=0, le=1)


class RawTranscriptSegment(BaseModel):
    segment_id: str
    chunk_id: int = Field(ge=1)
    source_ordinal: int = Field(ge=0)
    local_start_seconds: float = Field(ge=0)
    local_end_seconds: float = Field(ge=0)
    absolute_start_seconds: float = Field(ge=0)
    absolute_end_seconds: float = Field(ge=0)
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = Field(default=None, ge=0, le=1)
    compression_ratio: float | None = Field(default=None, ge=0)
    is_within_logical_range: bool = True
    words: list[RawWord] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_times(self) -> "RawTranscriptSegment":
        if self.local_start_seconds > self.local_end_seconds:
            raise ValueError("local segment start must be <= end")
        if self.absolute_start_seconds > self.absolute_end_seconds:
            raise ValueError("absolute segment start must be <= end")
        return self


class RawChunkStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RawChunkTranscript(BaseModel):
    algorithm_version: str
    chunk: TranscriptionChunk
    audio_fingerprint: str
    config_fingerprint: str
    chunk_fingerprint: str
    artifact_fingerprint: str
    status: RawChunkStatus = RawChunkStatus.COMPLETED
    language: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    segments: list[RawTranscriptSegment]
    empty_segment_count: int = Field(default=0, ge=0)
    transcription_seconds: float = Field(default=0.0, ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RawTranscriptStats(BaseModel):
    audio_duration_seconds: float = Field(ge=0)
    chunk_count: int = Field(default=0, ge=0)
    completed_chunk_count: int = Field(default=0, ge=0)
    raw_segment_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    empty_segment_count: int = Field(default=0, ge=0)
    transcribed_speech_duration_seconds: float = Field(default=0.0, ge=0)
    wall_clock_transcription_seconds: float = Field(default=0.0, ge=0)
    real_time_factor: float = Field(default=0.0, ge=0)


class RawTranscriptManifest(BaseModel):
    algorithm_version: str
    engine: str = "faster-whisper"
    model_name: str
    device: str
    compute_type: str
    language: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    config_fingerprint: str
    preparation_fingerprint: str
    audio_fingerprint: str
    chunk_fingerprints: list[str]
    artifact_fingerprint: str
    stats: RawTranscriptStats
    warnings: list[str] = Field(default_factory=list)
    segments: list[RawTranscriptSegment]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NormalizationFlag(str, Enum):
    DEDUPLICATED_OVERLAP = "DEDUPLICATED_OVERLAP"
    MERGED_ADJACENT = "MERGED_ADJACENT"
    TRIMMED_WHITESPACE = "TRIMMED_WHITESPACE"
    TIMESTAMP_CLAMPED = "TIMESTAMP_CLAMPED"
    LONG_SEGMENT_PRESERVED = "LONG_SEGMENT_PRESERVED"


class NormalizedTranscriptSegment(BaseModel):
    segment_id: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str
    source_segment_ids: list[str]
    source_chunk_ids: list[int]
    word_count: int = Field(default=0, ge=0)
    language: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = Field(default=None, ge=0, le=1)
    normalization_flags: list[str] = Field(default_factory=list)
    words: list[RawWord] = Field(default_factory=list)


class TranscriptStats(BaseModel):
    raw_segment_count: int = Field(default=0, ge=0)
    normalized_segment_count: int = Field(default=0, ge=0)
    deduplicated_segment_count: int = Field(default=0, ge=0)
    merged_segment_count: int = Field(default=0, ge=0)
    rejected_segment_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    speech_start_seconds: float | None = Field(default=None, ge=0)
    speech_end_seconds: float | None = Field(default=None, ge=0)
    total_transcribed_span_seconds: float = Field(default=0.0, ge=0)
    speech_coverage_ratio: float = Field(default=0.0, ge=0, le=1)
    mean_gap_seconds: float = Field(default=0.0, ge=0)
    median_gap_seconds: float = Field(default=0.0, ge=0)
    max_gap_seconds: float = Field(default=0.0, ge=0)


class NormalizedTranscriptManifest(BaseModel):
    algorithm_version: str
    raw_transcript_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    primary_language: str | None = None
    observed_languages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: TranscriptStats
    segments: list[NormalizedTranscriptSegment]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AlignmentType(str, Enum):
    OVERLAPPING_SPEECH = "OVERLAPPING_SPEECH"
    BETWEEN_SPEECH = "BETWEEN_SPEECH"
    BEFORE_FIRST_SPEECH = "BEFORE_FIRST_SPEECH"
    AFTER_LAST_SPEECH = "AFTER_LAST_SPEECH"
    NO_TRANSCRIPT = "NO_TRANSCRIPT"


class SpeechDirection(str, Enum):
    OVERLAPPING = "OVERLAPPING"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    EQUAL_DISTANCE = "EQUAL_DISTANCE"
    NONE = "NONE"


class CandidateTranscriptAlignment(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    boundary_id: int | None = Field(default=None, ge=1)
    selection_role: SelectionRole
    candidate_timestamp_seconds: float = Field(ge=0)
    alignment_type: AlignmentType
    overlapping_segment_ids: list[int] = Field(default_factory=list)
    primary_segment_id: int | None = Field(default=None, ge=1)
    previous_segment_id: int | None = Field(default=None, ge=1)
    next_segment_id: int | None = Field(default=None, ge=1)
    previous_gap_seconds: float | None = Field(default=None, ge=0)
    next_gap_seconds: float | None = Field(default=None, ge=0)
    nearest_speech_gap_seconds: float | None = Field(default=None, ge=0)
    nearest_speech_direction: SpeechDirection = SpeechDirection.NONE
    speech_proximity_score: float = Field(default=0.0, ge=0, le=1)
    segment_relative_position: float | None = Field(default=None, ge=0, le=1)
    distance_from_segment_midpoint_seconds: float | None = Field(default=None, ge=0)
    nearest_word: str | None = None
    nearest_word_start_seconds: float | None = Field(default=None, ge=0)
    nearest_word_end_seconds: float | None = Field(default=None, ge=0)
    nearest_word_gap_seconds: float | None = Field(default=None, ge=0)
    candidate_overlaps_word: bool = False
    stable_window_has_speech: bool = False
    boundary_speech_segment_id: int | None = Field(default=None, ge=1)


class AlignmentStats(BaseModel):
    candidate_count: int = Field(default=0, ge=0)
    primary_candidate_count: int = Field(default=0, ge=0)
    alternate_candidate_count: int = Field(default=0, ge=0)
    overlapping_speech_count: int = Field(default=0, ge=0)
    between_speech_count: int = Field(default=0, ge=0)
    before_first_speech_count: int = Field(default=0, ge=0)
    after_last_speech_count: int = Field(default=0, ge=0)
    no_transcript_count: int = Field(default=0, ge=0)
    mean_nearest_speech_gap_seconds: float = Field(default=0.0, ge=0)
    median_nearest_speech_gap_seconds: float = Field(default=0.0, ge=0)
    p90_nearest_speech_gap_seconds: float = Field(default=0.0, ge=0)
    max_nearest_speech_gap_seconds: float = Field(default=0.0, ge=0)
    mean_speech_proximity_score: float = Field(default=0.0, ge=0, le=1)


class CandidateAlignmentManifest(BaseModel):
    algorithm_version: str
    normalized_transcript_fingerprint: str
    selections_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: AlignmentStats
    alignments: list[CandidateTranscriptAlignment]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandidateTranscriptContext(BaseModel):
    candidate_id: int = Field(ge=1)
    stable_window_id: int = Field(ge=1)
    boundary_id: int | None = Field(default=None, ge=1)
    selection_role: SelectionRole
    candidate_timestamp_seconds: float = Field(ge=0)
    stable_window_start_seconds: float = Field(ge=0)
    stable_window_end_seconds: float = Field(ge=0)
    visual_boundary_timestamp_seconds: float | None = Field(default=None, ge=0)
    boundary_to_candidate_seconds: float | None = Field(default=None, ge=0)
    ranking_score: float = Field(ge=0, le=1)
    selection_confidence: float = Field(ge=0, le=1)
    is_ambiguous: bool = False
    sampled_frame_path: str
    processed_frame_path: str
    alignment_type: AlignmentType
    speech_proximity_score: float = Field(ge=0, le=1)
    nearest_speech_gap_seconds: float | None = Field(default=None, ge=0)
    requested_context_start_seconds: float = Field(ge=0)
    requested_context_end_seconds: float = Field(ge=0)
    actual_context_start_seconds: float = Field(ge=0)
    actual_context_end_seconds: float = Field(ge=0)
    before_segment_ids: list[int] = Field(default_factory=list)
    current_segment_ids: list[int] = Field(default_factory=list)
    after_segment_ids: list[int] = Field(default_factory=list)
    before_text: str = ""
    current_text: str = ""
    after_text: str = ""
    combined_text: str = ""
    speech_duration_seconds: float = Field(default=0.0, ge=0)
    context_speech_ratio: float = Field(default=0.0, ge=0, le=1)
    has_transcript_context: bool = False
    is_sparse_context: bool = False
    truncated_before: bool = False
    truncated_after: bool = False
    context_over_budget_due_to_current: bool = False
    context_signature: str


class ContextStats(BaseModel):
    context_count: int = Field(default=0, ge=0)
    primary_context_count: int = Field(default=0, ge=0)
    alternate_context_count: int = Field(default=0, ge=0)
    contexts_with_speech: int = Field(default=0, ge=0)
    contexts_without_speech: int = Field(default=0, ge=0)
    sparse_context_count: int = Field(default=0, ge=0)
    contexts_with_current_speech: int = Field(default=0, ge=0)
    contexts_without_current_speech: int = Field(default=0, ge=0)
    truncated_context_count: int = Field(default=0, ge=0)
    truncated_before_count: int = Field(default=0, ge=0)
    truncated_after_count: int = Field(default=0, ge=0)
    mean_context_characters: float = Field(default=0.0, ge=0)
    median_context_characters: float = Field(default=0.0, ge=0)
    p90_context_characters: float = Field(default=0.0, ge=0)
    mean_context_words: float = Field(default=0.0, ge=0)
    mean_actual_context_duration_seconds: float = Field(default=0.0, ge=0)


class CandidateContextsManifest(BaseModel):
    algorithm_version: str
    normalized_transcript_fingerprint: str
    alignment_fingerprint: str
    selections_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    stats: ContextStats
    contexts: list[CandidateTranscriptContext]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TranscriptResumeStage(str, Enum):
    AUDIO_PREPARATION = "AUDIO_PREPARATION"
    RAW_TRANSCRIPTION = "RAW_TRANSCRIPTION"
    NORMALIZATION = "NORMALIZATION"
    CANDIDATE_ALIGNMENT = "CANDIDATE_ALIGNMENT"
    CONTEXT_EXTRACTION = "CONTEXT_EXTRACTION"
    PHASE5_READY = "PHASE5_READY"


class TranscriptCacheSnapshot(BaseModel):
    preparation: AnalysisArtifactCheck
    raw_transcription: AnalysisArtifactCheck
    normalized_transcript: AnalysisArtifactCheck
    candidate_alignment: AnalysisArtifactCheck
    contexts: AnalysisArtifactCheck
    transcript_context_ready: AnalysisArtifactCheck
    valid_raw_chunks: int = Field(default=0, ge=0)
    total_raw_chunks: int = Field(default=0, ge=0)
    invalid_raw_chunk_ids: list[int] = Field(default_factory=list)
    resume_stage: TranscriptResumeStage


class Phase5Timings(BaseModel):
    audio_preparation_seconds: float = Field(default=0.0, ge=0)
    raw_transcription_seconds: float = Field(default=0.0, ge=0)
    normalization_seconds: float = Field(default=0.0, ge=0)
    alignment_seconds: float = Field(default=0.0, ge=0)
    context_extraction_seconds: float = Field(default=0.0, ge=0)
    total_phase5_seconds: float = Field(default=0.0, ge=0)


class TranscriptSummary(BaseModel):
    audio_duration_seconds: float = Field(default=0.0, ge=0)
    transcription_chunk_count: int = Field(default=0, ge=0)
    raw_segment_count: int = Field(default=0, ge=0)
    normalized_segment_count: int = Field(default=0, ge=0)
    primary_candidate_count: int = Field(default=0, ge=0)
    alternate_candidate_count: int = Field(default=0, ge=0)
    aligned_candidate_count: int = Field(default=0, ge=0)
    contexts_with_speech: int = Field(default=0, ge=0)
    contexts_without_speech: int = Field(default=0, ge=0)
    speech_coverage_ratio: float = Field(default=0.0, ge=0, le=1)
    detected_language: str | None = None
    real_time_factor: float = Field(default=0.0, ge=0)
    preparation_fingerprint: str
    raw_transcript_fingerprint: str
    normalized_transcript_fingerprint: str
    alignment_fingerprint: str
    contexts_fingerprint: str
    timings: Phase5Timings
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SemanticCandidateInput(BaseModel):
    candidate_id: int = Field(ge=1)
    selection_role: SelectionRole
    candidate_timestamp_seconds: float = Field(ge=0)
    stable_window_start_seconds: float = Field(ge=0)
    stable_window_end_seconds: float = Field(ge=0)
    visual_boundary_timestamp_seconds: float | None = Field(default=None, ge=0)
    sampled_frame_path: str
    processed_frame_path: str
    ranking_score: float = Field(ge=0, le=1)
    selection_confidence: float = Field(ge=0, le=1)
    is_ambiguous: bool
    alignment_type: AlignmentType
    speech_proximity_score: float = Field(ge=0, le=1)
    before_text: str = ""
    current_text: str = ""
    after_text: str = ""


class Phase5EvaluationReport(BaseModel):
    audio_duration_seconds: float = Field(default=0.0, ge=0)
    sample_rate_hz: int = Field(default=0, ge=0)
    channels: int = Field(default=0, ge=0)
    silent_window_ratio: float = Field(default=0.0, ge=0, le=1)
    clipping_ratio: float = Field(default=0.0, ge=0, le=1)
    chunk_count: int = Field(default=0, ge=0)
    raw_segments: int = Field(default=0, ge=0)
    normalized_segments: int = Field(default=0, ge=0)
    duplicates_removed: int = Field(default=0, ge=0)
    segments_merged: int = Field(default=0, ge=0)
    speech_coverage_ratio: float = Field(default=0.0, ge=0, le=1)
    candidate_count: int = Field(default=0, ge=0)
    overlapping_speech_count: int = Field(default=0, ge=0)
    contexts_with_speech: int = Field(default=0, ge=0)
    empty_contexts: int = Field(default=0, ge=0)
    sparse_contexts: int = Field(default=0, ge=0)
    truncated_contexts: int = Field(default=0, ge=0)
    mean_context_characters: float = Field(default=0.0, ge=0)
    real_time_factor: float = Field(default=0.0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

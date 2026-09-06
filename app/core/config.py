from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Notify"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    storage_root: Path = Path("storage/jobs")
    job_retention_hours: int = Field(default=72, ge=1)
    max_concurrent_jobs: int = Field(default=2, ge=1, le=32)
    log_level: str = "INFO"
    processor_mode: Literal["transcription", "candidates", "analysis", "ingestion", "fake"] = "analysis"

    # Phase-1 compatibility/testing processor.
    fake_processor_step_delay: float = Field(default=0.15, ge=0.0, le=60.0)
    simulated_failure_at_progress: int | None = Field(default=None, ge=1, le=99)

    # YouTube ingestion.
    video_max_height: int = Field(default=1080, ge=144, le=2160)
    video_preferred_container: str = "mp4"
    max_video_download_gb: float = Field(default=4.0, gt=0)
    video_download_retries: int = Field(default=3, ge=0, le=10)
    video_fragment_retries: int = Field(default=3, ge=0, le=10)
    download_disk_safety_margin_mb: int = Field(default=512, ge=0)

    # FFmpeg/ffprobe. Empty means auto-detect from PATH.
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None
    media_probe_timeout_seconds: int = Field(default=30, ge=1, le=300)

    # Normalized transcription audio operational limits.
    audio_extraction_timeout_multiplier: float = Field(default=3.0, ge=1.0, le=20.0)
    audio_disk_safety_margin_mb: int = Field(default=256, ge=0)
    audio_duration_tolerance_ratio: float = Field(default=0.01, ge=0.0, le=0.2)


    # Phase-3 frame sampling.
    frame_sample_fps: float = Field(default=1.0, gt=0.0, le=5.0)
    frame_sample_max_fps: float = Field(default=5.0, gt=0.0, le=10.0)
    max_sampled_frames: int = Field(default=20_000, ge=1, le=200_000)
    frame_estimated_size_kb: int = Field(default=150, ge=1, le=10_000)
    frame_disk_safety_margin_mb: int = Field(default=512, ge=0)
    frame_sampling_timeout_multiplier: float = Field(default=2.0, ge=0.25, le=20.0)
    frame_sampling_base_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    frame_jpeg_quality: int = Field(default=90, ge=40, le=100)

    # Phase-3 frame preprocessing.
    analysis_frame_width: int = Field(default=640, ge=64, le=3840)
    max_invalid_frame_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    dark_frame_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    black_pixel_value_threshold: int = Field(default=10, ge=0, le=255)
    black_pixel_ratio_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    blur_variance_threshold: float = Field(default=50.0, ge=0.0)
    low_information_edge_density: float = Field(default=0.01, ge=0.0, le=1.0)
    low_information_contrast: float = Field(default=0.04, ge=0.0, le=1.0)
    preprocessing_progress_step_percent: int = Field(default=1, ge=1, le=25)

    # Phase-3 visual difference scoring.
    diff_pixel_weight: float = Field(default=0.25, ge=0.0)
    diff_ssim_weight: float = Field(default=0.35, ge=0.0)
    diff_phash_weight: float = Field(default=0.20, ge=0.0)
    diff_edge_weight: float = Field(default=0.20, ge=0.0)
    diff_phash_size: int = Field(default=8, ge=4, le=32)
    diff_canny_low: int = Field(default=100, ge=0, le=255)
    diff_canny_high: int = Field(default=200, ge=0, le=255)
    max_invalid_comparison_ratio: float = Field(default=0.02, ge=0.0, le=1.0)

    # Phase-3 major-change detection.
    major_change_min_score: float = Field(default=0.30, ge=0.0, le=1.0)
    very_major_change_min_score: float = Field(default=0.60, ge=0.0, le=1.0)
    major_change_robust_multiplier: float = Field(default=6.0, ge=0.0, le=50.0)
    major_change_merge_window_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    max_major_event_ratio_warning: float = Field(default=0.30, ge=0.0, le=1.0)

    # Phase-3 temporal timeline.
    timeline_stable_min_score: float = Field(default=0.03, ge=0.0, le=1.0)
    timeline_stable_max_score: float = Field(default=0.10, ge=0.0, le=1.0)
    timeline_stable_robust_multiplier: float = Field(default=1.5, ge=0.0, le=20.0)
    timeline_exit_stable_factor: float = Field(default=1.6, ge=1.0, le=10.0)
    min_stable_duration_seconds: float = Field(default=2.0, ge=0.0, le=600.0)
    min_changing_duration_seconds: float = Field(default=1.0, ge=0.0, le=600.0)
    max_stable_gap_seconds: float = Field(default=0.5, ge=0.0, le=60.0)

    # Phase-4.1 stability windows.
    stability_window_min_duration_seconds: float = Field(default=2.0, ge=0.0, le=600.0)
    stability_max_invalid_frame_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    stability_max_black_frame_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    stability_duration_saturation_seconds: float = Field(default=10.0, gt=0.0, le=3600.0)
    stability_difference_weight: float = Field(default=0.55, ge=0.0)
    stability_variance_weight: float = Field(default=0.15, ge=0.0)
    stability_max_spike_weight: float = Field(default=0.15, ge=0.0)
    stability_duration_weight: float = Field(default=0.15, ge=0.0)

    # Phase-4.2 stable-boundary detection.
    boundary_context_seconds: float = Field(default=4.0, gt=0.0, le=60.0)
    boundary_min_activity_drop: float = Field(default=0.10, ge=0.0, le=1.0)
    boundary_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    boundary_duration_saturation_seconds: float = Field(default=10.0, gt=0.0, le=3600.0)
    boundary_drop_weight: float = Field(default=0.40, ge=0.0)
    boundary_stability_weight: float = Field(default=0.25, ge=0.0)
    boundary_quality_weight: float = Field(default=0.20, ge=0.0)
    boundary_duration_weight: float = Field(default=0.10, ge=0.0)
    boundary_type_weight: float = Field(default=0.05, ge=0.0)

    # Phase-4.3 candidate timestamp generation.
    candidate_settle_delay_seconds: float = Field(default=1.0, ge=0.0, le=30.0)
    candidate_exit_margin_seconds: float = Field(default=0.75, ge=0.0, le=30.0)
    min_candidate_spacing_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
    max_candidates_per_window: int = Field(default=5, ge=1, le=20)
    candidate_frame_search_radius_seconds: float = Field(default=1.5, ge=0.0, le=30.0)
    allow_window_only_candidates: bool = True
    max_total_candidates: int = Field(default=1000, ge=1, le=100_000)

    # Phase-4.4 explainable candidate heuristics.
    candidate_sharpness_saturation: float = Field(default=200.0, gt=0.0)
    candidate_local_context_seconds: float = Field(default=1.5, gt=0.0, le=60.0)
    candidate_transition_safe_margin_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    completeness_boundary_weight: float = Field(default=0.20, ge=0.0)
    completeness_stability_weight: float = Field(default=0.18, ge=0.0)
    completeness_activity_weight: float = Field(default=0.14, ge=0.0)
    completeness_accumulation_weight: float = Field(default=0.14, ge=0.0)
    completeness_density_weight: float = Field(default=0.10, ge=0.0)
    completeness_position_weight: float = Field(default=0.08, ge=0.0)
    completeness_quality_weight: float = Field(default=0.08, ge=0.0)
    completeness_safety_weight: float = Field(default=0.08, ge=0.0)

    # Phase-4.5 ranking and per-window selection.
    rank_quality_weight: float = Field(default=0.22, ge=0.0)
    rank_completeness_weight: float = Field(default=0.28, ge=0.0)
    rank_safety_weight: float = Field(default=0.18, ge=0.0)
    rank_local_stability_weight: float = Field(default=0.14, ge=0.0)
    rank_boundary_weight: float = Field(default=0.08, ge=0.0)
    rank_density_weight: float = Field(default=0.06, ge=0.0)
    rank_accumulation_weight: float = Field(default=0.04, ge=0.0)
    top_candidates_per_window: int = Field(default=2, ge=1, le=5)
    min_primary_ranking_score: float = Field(default=0.0, ge=0.0, le=1.0)
    min_alternate_ranking_score: float = Field(default=0.25, ge=0.0, le=1.0)
    ambiguous_ranking_gap: float = Field(default=0.05, ge=0.0, le=1.0)
    ranking_tie_epsilon: float = Field(default=1e-6, gt=0.0, le=0.1)

    # Phase 5.1 - audio validation and transcription preparation.
    audio_video_duration_tolerance_seconds: float = Field(default=3.0, ge=0.0, le=120.0)
    audio_video_duration_tolerance_ratio: float = Field(default=0.01, ge=0.0, le=0.2)
    audio_min_duration_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    audio_diagnostic_window_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    audio_silence_rms_threshold: float = Field(default=0.003, ge=0.0, le=1.0)
    audio_effectively_silent_ratio: float = Field(default=0.98, ge=0.0, le=1.0)
    audio_high_clipping_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    transcription_chunk_seconds: float = Field(default=600.0, gt=1.0, le=7200.0)
    transcription_chunk_overlap_seconds: float = Field(default=3.0, ge=0.0, le=60.0)

    # Phase 5.2 - local faster-whisper transcription.
    whisper_model_size: str = "small"
    whisper_device: Literal["cpu", "cuda", "auto"] = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "auto"
    whisper_beam_size: int = Field(default=5, ge=1, le=20)
    whisper_vad_filter: bool = True
    whisper_word_timestamps: bool = True
    whisper_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    whisper_condition_on_previous_text: bool = False
    whisper_chunk_max_retries: int = Field(default=1, ge=0, le=5)
    whisper_model_cache_dir: Path | None = None
    max_concurrent_transcriptions: int = Field(default=1, ge=1, le=8)

    # Phase 5.3 - transcript normalization.
    transcript_overlap_time_tolerance_seconds: float = Field(default=4.0, ge=0.0, le=30.0)
    transcript_duplicate_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    transcript_merge_max_gap_seconds: float = Field(default=0.75, ge=0.0, le=10.0)
    transcript_max_segment_duration_seconds: float = Field(default=20.0, gt=0.0, le=300.0)
    transcript_max_segment_characters: int = Field(default=400, ge=20, le=5000)
    transcript_short_segment_max_duration_seconds: float = Field(default=1.5, ge=0.0, le=30.0)

    # Phase 5.4 - visual candidate/transcript alignment.
    alignment_timestamp_epsilon_seconds: float = Field(default=0.05, ge=0.0, le=2.0)
    alignment_proximity_saturation_seconds: float = Field(default=10.0, gt=0.0, le=300.0)
    alignment_word_level_enabled: bool = True

    # Phase 5.5 - compact transcript context.
    transcript_context_before_seconds: float = Field(default=12.0, ge=0.0, le=300.0)
    transcript_context_after_seconds: float = Field(default=8.0, ge=0.0, le=300.0)
    transcript_context_max_boundary_extension_seconds: float = Field(default=3.0, ge=0.0, le=30.0)
    transcript_context_max_characters: int = Field(default=4000, ge=100, le=50000)
    transcript_context_max_words: int = Field(default=500, ge=10, le=10000)
    transcript_context_min_words_for_non_sparse: int = Field(default=8, ge=0, le=1000)

    # Phase 5.7 diagnostics only.
    phase5_low_speech_coverage_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    phase5_high_empty_context_ratio: float = Field(default=0.50, ge=0.0, le=1.0)

    # Phase-4.7 diagnostics only; these values never auto-tune production logic.
    phase4_high_primary_density_per_minute: float = Field(default=4.0, gt=0.0)
    phase4_high_ambiguity_ratio: float = Field(default=0.80, ge=0.0, le=1.0)
    phase4_winner_type_skew_ratio: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper().strip()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return normalized

    @field_validator("ffmpeg_path", "ffprobe_path", "whisper_model_cache_dir", mode="before")
    @classmethod
    def empty_tool_path_is_none(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @field_validator("video_preferred_container")
    @classmethod
    def normalize_container(cls, value: str) -> str:
        value = value.lower().strip().lstrip(".")
        if value not in {"mp4", "webm", "mkv"}:
            raise ValueError("VIDEO_PREFERRED_CONTAINER must be mp4, webm, or mkv")
        return value

    @model_validator(mode="after")
    def validate_phase3_configuration(self) -> "AppSettings":
        if self.frame_sample_fps > self.frame_sample_max_fps:
            raise ValueError("FRAME_SAMPLE_FPS must be <= FRAME_SAMPLE_MAX_FPS")
        if (
            self.diff_pixel_weight
            + self.diff_ssim_weight
            + self.diff_phash_weight
            + self.diff_edge_weight
        ) <= 0:
            raise ValueError("At least one visual difference metric weight must be > 0")
        if self.diff_canny_low > self.diff_canny_high:
            raise ValueError("DIFF_CANNY_LOW must be <= DIFF_CANNY_HIGH")
        if self.timeline_stable_min_score > self.timeline_stable_max_score:
            raise ValueError("TIMELINE_STABLE_MIN_SCORE must be <= TIMELINE_STABLE_MAX_SCORE")
        if self.major_change_min_score > self.very_major_change_min_score:
            raise ValueError("MAJOR_CHANGE_MIN_SCORE must be <= VERY_MAJOR_CHANGE_MIN_SCORE")
        phase4_weight_groups = {
            "stability": (
                self.stability_difference_weight,
                self.stability_variance_weight,
                self.stability_max_spike_weight,
                self.stability_duration_weight,
            ),
            "boundary": (
                self.boundary_drop_weight,
                self.boundary_stability_weight,
                self.boundary_quality_weight,
                self.boundary_duration_weight,
                self.boundary_type_weight,
            ),
            "completeness": (
                self.completeness_boundary_weight,
                self.completeness_stability_weight,
                self.completeness_activity_weight,
                self.completeness_accumulation_weight,
                self.completeness_density_weight,
                self.completeness_position_weight,
                self.completeness_quality_weight,
                self.completeness_safety_weight,
            ),
            "ranking": (
                self.rank_quality_weight,
                self.rank_completeness_weight,
                self.rank_safety_weight,
                self.rank_local_stability_weight,
                self.rank_boundary_weight,
                self.rank_density_weight,
                self.rank_accumulation_weight,
            ),
        }
        for name, weights in phase4_weight_groups.items():
            if sum(weights) <= 0:
                raise ValueError(f"At least one {name} weight must be > 0")
        if self.transcription_chunk_overlap_seconds >= self.transcription_chunk_seconds:
            raise ValueError("TRANSCRIPTION_CHUNK_OVERLAP_SECONDS must be smaller than TRANSCRIPTION_CHUNK_SECONDS")
        if not self.whisper_model_size.strip():
            raise ValueError("WHISPER_MODEL_SIZE must not be empty")
        if not self.whisper_compute_type.strip():
            raise ValueError("WHISPER_COMPUTE_TYPE must not be empty")
        if self.min_alternate_ranking_score < self.min_primary_ranking_score:
            # Alternates may be held to a stricter threshold, never a looser one.
            raise ValueError("MIN_ALTERNATE_RANKING_SCORE must be >= MIN_PRIMARY_RANKING_SCORE")
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()

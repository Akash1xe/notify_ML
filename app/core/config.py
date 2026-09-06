from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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
    processor_mode: Literal["ingestion", "fake"] = "ingestion"

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

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper().strip()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return normalized

    @field_validator("ffmpeg_path", "ffprobe_path", mode="before")
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


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()

from __future__ import annotations

from datetime import UTC, datetime
from pydantic import BaseModel, Field


class StageTimings(BaseModel):
    metadata_seconds: float = 0.0
    download_seconds: float = 0.0
    inspection_seconds: float = 0.0
    audio_seconds: float = 0.0
    total_seconds: float = 0.0


class IngestionResult(BaseModel):
    video_id: str
    title: str
    source_video: str
    audio_file: str
    duration_seconds: float | None
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_sample_rate: int
    audio_channels: int
    video_size_bytes: int
    audio_size_bytes: int
    workspace_size_bytes: int
    timings: StageTimings
    ingestion_completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

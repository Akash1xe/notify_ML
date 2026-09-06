from __future__ import annotations

from datetime import UTC, datetime
from pydantic import BaseModel, Field


class FileFingerprint(BaseModel):
    file_size_bytes: int
    mtime_ns: int


class VideoStreamInfo(BaseModel):
    codec: str | None = None
    codec_profile: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    pixel_format: str | None = None
    bitrate: int | None = None
    duration_seconds: float | None = None
    rotation: int | None = None


class AudioStreamInfo(BaseModel):
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    bitrate: int | None = None
    duration_seconds: float | None = None


class MediaInspection(BaseModel):
    file_path: str
    container: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    file_size_bytes: int
    overall_bitrate: int | None = None
    video: VideoStreamInfo | None = None
    audio: AudioStreamInfo | None = None
    source_fingerprint: FileFingerprint
    inspected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MediaToolInfo(BaseModel):
    available: bool
    path: str | None = None
    version: str | None = None


class MediaToolsCapabilities(BaseModel):
    ffmpeg: MediaToolInfo
    ffprobe: MediaToolInfo


class AudioResult(BaseModel):
    path: str
    format: str = "wav"
    codec: str = "pcm_s16le"
    sample_rate: int = 16000
    channels: int = 1
    duration_seconds: float
    file_size_bytes: int
    source_video_id: str
    source_fingerprint: FileFingerprint
    audio_fingerprint: FileFingerprint
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

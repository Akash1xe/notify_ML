from __future__ import annotations

from datetime import UTC, date, datetime

from pydantic import BaseModel, Field


class NormalizedYouTubeURL(BaseModel):
    video_id: str
    canonical_url: str


class YouTubeFormat(BaseModel):
    format_id: str
    ext: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    filesize_bytes: int | None = None
    has_video: bool = False
    has_audio: bool = False
    vcodec: str | None = None
    acodec: str | None = None


class YouTubeMetadata(BaseModel):
    video_id: str
    canonical_url: str
    title: str
    description: str | None = None
    channel: str | None = None
    channel_id: str | None = None
    uploader: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    thumbnail_url: str | None = None
    upload_date: date | None = None
    availability: str | None = None
    live_status: str | None = None
    is_live: bool = False
    was_live: bool = False
    age_limit: int | None = None
    view_count: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    format_count: int = 0
    subtitles_available: bool = False
    automatic_captions_available: bool = False
    formats: list[YouTubeFormat] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DownloadResult(BaseModel):
    video_id: str
    path: str
    container: str
    format_id: str | None = None
    filesize_bytes: int
    resolution: str | None = None
    height: int | None = None
    width: int | None = None
    fps: float | None = None
    downloaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DownloadProgress(BaseModel):
    bytes_downloaded: int | None = None
    total_bytes: int | None = None
    estimated_total_bytes: int | None = None
    download_percent: float | None = None
    download_speed: float | None = None
    eta_seconds: float | None = None

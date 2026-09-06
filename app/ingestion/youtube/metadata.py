from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from app.core.exceptions import (
    LiveVideoNotSupportedError,
    MetadataExtractionError,
    PrivateVideoError,
    VideoUnavailableError,
)
from app.ingestion.youtube.models import YouTubeFormat, YouTubeMetadata
from app.ingestion.youtube.validator import normalize_youtube_url


class MetadataBackend(Protocol):
    def extract(self, url: str) -> dict[str, Any]: ...


class YtDlpMetadataBackend:
    def extract(self, url: str) -> dict[str, Any]:
        try:
            import yt_dlp  # type: ignore
        except ImportError as exc:
            raise MetadataExtractionError(
                "yt-dlp is not installed. Install project dependencies before processing videos."
            ) from exc
        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
            "extract_flat": False,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                result = ydl.extract_info(url, download=False)
        except Exception as exc:  # yt-dlp intentionally kept behind this boundary
            message = str(exc).lower()
            if "private video" in message or "private" in message:
                raise PrivateVideoError("The YouTube video is private and cannot be accessed.") from exc
            if any(term in message for term in ("unavailable", "deleted", "not available", "sign in")):
                raise VideoUnavailableError("The YouTube video is unavailable.") from exc
            raise MetadataExtractionError("Unable to fetch YouTube video metadata.") from exc
        if not isinstance(result, dict):
            raise MetadataExtractionError("YouTube metadata response was invalid.")
        return result


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _format(raw: dict[str, Any]) -> YouTubeFormat:
    vcodec = raw.get("vcodec")
    acodec = raw.get("acodec")
    size = _as_int(raw.get("filesize")) or _as_int(raw.get("filesize_approx"))
    return YouTubeFormat(
        format_id=str(raw.get("format_id", "unknown")),
        ext=raw.get("ext"),
        width=_as_int(raw.get("width")),
        height=_as_int(raw.get("height")),
        fps=_as_float(raw.get("fps")),
        filesize_bytes=size,
        has_video=bool(vcodec and vcodec != "none"),
        has_audio=bool(acodec and acodec != "none"),
        vcodec=vcodec,
        acodec=acodec,
    )


def normalize_metadata(raw: dict[str, Any], expected_video_id: str) -> YouTubeMetadata:
    actual_id = str(raw.get("id") or expected_video_id)
    normalized = normalize_youtube_url(f"https://www.youtube.com/watch?v={actual_id}")

    live_status = raw.get("live_status")
    is_live = bool(raw.get("is_live")) or live_status == "is_live"
    if is_live or live_status in {"is_upcoming", "upcoming"}:
        raise LiveVideoNotSupportedError(
            "Live or upcoming YouTube streams are not supported. Use the finished replay instead."
        )

    upload_date = None
    raw_upload_date = raw.get("upload_date")
    if raw_upload_date:
        try:
            upload_date = datetime.strptime(str(raw_upload_date), "%Y%m%d").date()
        except ValueError:
            upload_date = None

    formats = [_format(item) for item in raw.get("formats", []) if isinstance(item, dict)]
    duration = _as_float(raw.get("duration"))
    if duration is not None and duration < 0:
        duration = None

    return YouTubeMetadata(
        video_id=normalized.video_id,
        canonical_url=normalized.canonical_url,
        title=str(raw.get("title") or "Untitled YouTube video"),
        description=raw.get("description"),
        channel=raw.get("channel"),
        channel_id=raw.get("channel_id"),
        uploader=raw.get("uploader"),
        duration_seconds=duration,
        thumbnail_url=raw.get("thumbnail"),
        upload_date=upload_date,
        availability=raw.get("availability"),
        live_status=live_status,
        is_live=is_live,
        was_live=bool(raw.get("was_live")) or live_status == "was_live",
        age_limit=_as_int(raw.get("age_limit")),
        view_count=_as_int(raw.get("view_count")),
        width=_as_int(raw.get("width")),
        height=_as_int(raw.get("height")),
        fps=_as_float(raw.get("fps")),
        format_count=len(formats),
        subtitles_available=bool(raw.get("subtitles")),
        automatic_captions_available=bool(raw.get("automatic_captions")),
        formats=formats,
    )


class YouTubeMetadataExtractor:
    def __init__(self, backend: MetadataBackend | None = None) -> None:
        self._backend = backend or YtDlpMetadataBackend()

    def extract(self, url: str) -> YouTubeMetadata:
        normalized = normalize_youtube_url(url)
        raw = self._backend.extract(normalized.canonical_url)
        return normalize_metadata(raw, normalized.video_id)

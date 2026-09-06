from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from app.core.exceptions import InvalidYouTubeURLError
from app.ingestion.youtube.models import NormalizedYouTubeURL


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
}


def _validate_video_id(video_id: str | None) -> str:
    candidate = (video_id or "").strip()
    if not _VIDEO_ID_RE.fullmatch(candidate):
        raise InvalidYouTubeURLError("The YouTube URL does not contain a valid video id.")
    return candidate


def normalize_youtube_url(raw_url: str) -> NormalizedYouTubeURL:
    try:
        parsed = urlparse(raw_url.strip())
    except ValueError as exc:
        raise InvalidYouTubeURLError("The supplied YouTube URL is malformed.") from exc

    if parsed.scheme not in {"http", "https"}:
        raise InvalidYouTubeURLError("Only HTTP(S) YouTube URLs are supported.")

    host = (parsed.hostname or "").lower()
    video_id: str | None = None

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0] if parsed.path else None
    elif host in _YOUTUBE_HOSTS:
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path.rstrip("/") == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in {"shorts", "live", "embed"}:
            video_id = parts[1]
    else:
        raise InvalidYouTubeURLError("Only YouTube video URLs are supported.")

    video_id = _validate_video_id(video_id)
    return NormalizedYouTubeURL(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
    )

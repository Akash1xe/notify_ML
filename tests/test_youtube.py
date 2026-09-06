from __future__ import annotations

import pytest

from app.core.exceptions import InvalidYouTubeURLError, LiveVideoNotSupportedError
from app.ingestion.youtube.downloader import build_format_selector, estimate_download_bytes
from app.ingestion.youtube.metadata import YouTubeMetadataExtractor
from app.ingestion.youtube.models import YouTubeFormat, YouTubeMetadata
from app.ingestion.youtube.validator import normalize_youtube_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=abc",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://youtube.com/live/dQw4w9WgXcQ?feature=share",
    ],
)
def test_youtube_urls_normalize(url: str):
    normalized = normalize_youtube_url(url)
    assert normalized.video_id == "dQw4w9WgXcQ"
    assert normalized.canonical_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.mark.parametrize("url", ["https://google.com", "ftp://youtu.be/dQw4w9WgXcQ", "https://youtube.com/watch?v=bad!"])
def test_invalid_youtube_urls_rejected(url: str):
    with pytest.raises(InvalidYouTubeURLError):
        normalize_youtube_url(url)


class MetadataBackend:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def extract(self, url: str):
        self.calls += 1
        return self.payload


def sample_raw(**updates):
    payload = {
        "id": "dQw4w9WgXcQ",
        "title": "Kafka Tutorial",
        "channel": "Example",
        "duration": 3600.5,
        "upload_date": "20260820",
        "live_status": "not_live",
        "formats": [
            {"format_id": "22", "ext": "mp4", "width": 1280, "height": 720, "fps": 30, "filesize": 1000, "vcodec": "h264", "acodec": "aac"},
        ],
    }
    payload.update(updates)
    return payload


def test_metadata_is_normalized_without_network():
    backend = MetadataBackend(sample_raw())
    metadata = YouTubeMetadataExtractor(backend).extract("https://youtu.be/dQw4w9WgXcQ")
    assert metadata.title == "Kafka Tutorial"
    assert metadata.duration_seconds == 3600.5
    assert metadata.format_count == 1
    assert metadata.formats[0].has_audio is True


def test_current_live_stream_is_rejected():
    backend = MetadataBackend(sample_raw(is_live=True, live_status="is_live"))
    with pytest.raises(LiveVideoNotSupportedError):
        YouTubeMetadataExtractor(backend).extract("https://youtu.be/dQw4w9WgXcQ")


def test_format_selector_caps_height():
    selector = build_format_selector(1080, "mp4")
    assert "height<=1080" in selector
    assert "ext=mp4" in selector


def test_download_estimate_uses_progressive_or_video_plus_audio():
    metadata = YouTubeMetadata(
        video_id="dQw4w9WgXcQ",
        canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="x",
        formats=[
            YouTubeFormat(format_id="v", height=1080, filesize_bytes=1000, has_video=True),
            YouTubeFormat(format_id="a", filesize_bytes=200, has_audio=True),
        ],
    )
    assert estimate_download_bytes(metadata, 1080) == 1200

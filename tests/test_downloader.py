from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import AppSettings
from app.core.exceptions import DownloadCancelledError, DownloadOutputMissingError
from app.ingestion.youtube.downloader import YouTubeDownloadManager
from app.ingestion.youtube.models import YouTubeMetadata


class Backend:
    def __init__(self, create_output: bool = True):
        self.create_output = create_output
        self.options = None

    def download(self, url, *, options, progress_hook):
        self.options = options
        progress_hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100, "speed": 10, "eta": 5})
        if self.create_output:
            target = Path(options["outtmpl"].replace("%(ext)s", "mp4"))
            target.write_bytes(b"video")
        return {"format_id": "22", "width": 1280, "height": 720, "fps": 30}


def metadata():
    return YouTubeMetadata(
        video_id="dQw4w9WgXcQ",
        canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Lecture",
    )


def test_download_manager_writes_manifest_and_reports_progress(tmp_path: Path):
    backend = Backend()
    settings = AppSettings(storage_root=tmp_path / "jobs")
    manager = YouTubeDownloadManager(settings, backend)
    values = []
    output = tmp_path / "source"
    result = manager.download(
        metadata=metadata(),
        output_dir=output,
        relative_path=lambda path: f"source/{path.name}",
        progress_callback=values.append,
        cancel_check=lambda: False,
        manifest_path=output / "download.json",
    )
    assert result.path == "source/video.mp4"
    assert result.filesize_bytes == 5
    assert values[0].download_percent == 50
    assert (output / "download.json").exists()
    assert "height<=1080" in backend.options["format"]


def test_download_manager_cancellation_from_progress_hook(tmp_path: Path):
    backend = Backend()
    manager = YouTubeDownloadManager(AppSettings(storage_root=tmp_path / "jobs"), backend)
    with pytest.raises(DownloadCancelledError):
        manager.download(
            metadata=metadata(),
            output_dir=tmp_path / "source",
            relative_path=lambda path: path.name,
            progress_callback=lambda _: None,
            cancel_check=lambda: True,
            manifest_path=tmp_path / "source" / "download.json",
        )


def test_downloader_does_not_accept_missing_output(tmp_path: Path):
    backend = Backend(create_output=False)
    manager = YouTubeDownloadManager(AppSettings(storage_root=tmp_path / "jobs"), backend)
    with pytest.raises(DownloadOutputMissingError):
        manager.download(
            metadata=metadata(),
            output_dir=tmp_path / "source",
            relative_path=lambda path: path.name,
            progress_callback=lambda _: None,
            cancel_check=lambda: False,
            manifest_path=tmp_path / "source" / "download.json",
        )

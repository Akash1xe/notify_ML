from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Protocol

from app.core.config import AppSettings
from app.core.exceptions import (
    DownloadCancelledError,
    DownloadOutputMissingError,
    InsufficientDiskSpaceError,
    VideoDownloadError,
)
from app.ingestion.youtube.models import DownloadProgress, DownloadResult, YouTubeMetadata
from app.storage.workspace import atomic_write_json


ProgressCallback = Callable[[DownloadProgress], None]
CancelCheck = Callable[[], bool]


class DownloadBackend(Protocol):
    def download(
        self,
        url: str,
        *,
        options: dict[str, Any],
        progress_hook: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]: ...


class YtDlpDownloadBackend:
    def download(
        self,
        url: str,
        *,
        options: dict[str, Any],
        progress_hook: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        try:
            import yt_dlp  # type: ignore
        except ImportError as exc:
            raise VideoDownloadError(
                "yt-dlp is not installed. Install project dependencies before downloading videos."
            ) from exc

        merged = dict(options)
        merged["progress_hooks"] = [progress_hook]
        try:
            with yt_dlp.YoutubeDL(merged) as ydl:
                info = ydl.extract_info(url, download=True)
        except DownloadCancelledError:
            raise
        except Exception as exc:
            cause = exc
            while cause is not None:
                if isinstance(cause, DownloadCancelledError):
                    raise cause
                cause = cause.__cause__ or cause.__context__
            raise VideoDownloadError("The lecture video could not be downloaded.") from exc
        if not isinstance(info, dict):
            raise VideoDownloadError("yt-dlp returned an invalid download result.")
        return info


def build_format_selector(max_height: int, preferred_container: str) -> str:
    if preferred_container == "mp4":
        return (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best"
        )
    return (
        f"bestvideo[height<={max_height}][ext={preferred_container}]+bestaudio/"
        f"best[height<={max_height}][ext={preferred_container}]/best[height<={max_height}]/best"
    )


def estimate_download_bytes(metadata: YouTubeMetadata, max_height: int) -> int | None:
    eligible = [
        fmt for fmt in metadata.formats
        if fmt.has_video and (fmt.height is None or fmt.height <= max_height) and fmt.filesize_bytes
    ]
    video_only = [fmt for fmt in eligible if not fmt.has_audio]
    audio_only = [
        fmt for fmt in metadata.formats
        if fmt.has_audio and not fmt.has_video and fmt.filesize_bytes
    ]
    # The selector prefers a separate high-quality video stream when available.
    # Estimate that path first and add the largest known audio-only stream.
    if video_only:
        best_height = max((fmt.height or 0) for fmt in video_only)
        candidates = [fmt.filesize_bytes for fmt in video_only if (fmt.height or 0) == best_height and fmt.filesize_bytes]
        return max(candidates) + (max(fmt.filesize_bytes for fmt in audio_only) if audio_only else 0)
    progressive = [fmt.filesize_bytes for fmt in eligible if fmt.has_audio and fmt.filesize_bytes]
    return max(progressive) if progressive else None


class YouTubeDownloadManager:
    def __init__(self, settings: AppSettings, backend: DownloadBackend | None = None) -> None:
        self._settings = settings
        self._backend = backend or YtDlpDownloadBackend()

    def _check_disk(self, output_dir: Path, estimate: int | None) -> None:
        if estimate is None:
            return
        max_bytes = int(self._settings.max_video_download_gb * 1024**3)
        if estimate > max_bytes:
            raise VideoDownloadError(
                f"Estimated video download exceeds the configured {self._settings.max_video_download_gb:g} GB limit."
            )
        free = shutil.disk_usage(output_dir).free
        margin = self._settings.download_disk_safety_margin_mb * 1024**2
        if free < estimate + margin:
            raise InsufficientDiskSpaceError("There is not enough free disk space to download this lecture.")

    def download(
        self,
        *,
        metadata: YouTubeMetadata,
        output_dir: Path,
        relative_path: Callable[[Path], str],
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
        manifest_path: Path,
    ) -> DownloadResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        estimate = estimate_download_bytes(metadata, self._settings.video_max_height)
        self._check_disk(output_dir, estimate)

        def hook(payload: dict[str, Any]) -> None:
            if cancel_check():
                raise DownloadCancelledError("Video download was cancelled.")
            if payload.get("status") != "downloading":
                return
            downloaded = payload.get("downloaded_bytes")
            total = payload.get("total_bytes") or payload.get("total_bytes_estimate")
            percent = None
            if downloaded is not None and total:
                try:
                    percent = min(100.0, max(0.0, float(downloaded) * 100.0 / float(total)))
                except (TypeError, ValueError, ZeroDivisionError):
                    percent = None
            progress_callback(
                DownloadProgress(
                    bytes_downloaded=int(downloaded) if downloaded is not None else None,
                    total_bytes=int(payload["total_bytes"]) if payload.get("total_bytes") else None,
                    estimated_total_bytes=int(payload["total_bytes_estimate"])
                    if payload.get("total_bytes_estimate")
                    else None,
                    download_percent=percent,
                    download_speed=float(payload["speed"]) if payload.get("speed") else None,
                    eta_seconds=float(payload["eta"]) if payload.get("eta") else None,
                )
            )

        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": build_format_selector(
                self._settings.video_max_height,
                self._settings.video_preferred_container,
            ),
            "merge_output_format": self._settings.video_preferred_container,
            "outtmpl": str(output_dir / "video.%(ext)s"),
            "retries": self._settings.video_download_retries,
            "fragment_retries": self._settings.video_fragment_retries,
            "continuedl": True,
        }
        info = self._backend.download(
            metadata.canonical_url,
            options=options,
            progress_hook=hook,
        )
        if cancel_check():
            raise DownloadCancelledError("Video download was cancelled.")

        candidates = [
            path
            for path in output_dir.glob("video.*")
            if path.is_file()
            and not path.name.endswith((".part", ".ytdl", ".tmp"))
            and path.name not in {"download.json", "metadata.json", "media.json"}
        ]
        if not candidates:
            raise DownloadOutputMissingError("The downloader finished without producing a video file.")
        video_path = max(candidates, key=lambda item: item.stat().st_size)
        size = video_path.stat().st_size
        max_bytes = int(self._settings.max_video_download_gb * 1024**3)
        if size <= 0:
            raise DownloadOutputMissingError("The downloaded video file is empty.")
        if size > max_bytes:
            video_path.unlink(missing_ok=True)
            raise VideoDownloadError("Downloaded video exceeds the configured file-size limit.")

        width = info.get("width")
        height = info.get("height")
        fps = info.get("fps")
        result = DownloadResult(
            video_id=metadata.video_id,
            path=relative_path(video_path),
            container=video_path.suffix.lstrip(".").lower(),
            format_id=str(info.get("format_id")) if info.get("format_id") is not None else None,
            filesize_bytes=size,
            resolution=f"{width}x{height}" if width and height else None,
            width=int(width) if width else metadata.width,
            height=int(height) if height else metadata.height,
            fps=float(fps) if fps else metadata.fps,
        )
        atomic_write_json(manifest_path, result.model_dump(mode="json"))
        return result

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from app.core.config import AppSettings
from app.core.exceptions import FFmpegNotFoundError, FFprobeNotFoundError
from app.media.models import MediaToolInfo, MediaToolsCapabilities


_VERSION_RE = re.compile(r"version\s+([^\s]+)", re.IGNORECASE)


class MediaToolsService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @staticmethod
    def _resolve(configured: Path | None, name: str) -> str | None:
        if configured is not None:
            path = configured.expanduser().resolve()
            return str(path) if path.is_file() else None
        return shutil.which(name)

    @staticmethod
    def _tool_info(path: str | None) -> MediaToolInfo:
        if not path:
            return MediaToolInfo(available=False)
        version = None
        try:
            result = subprocess.run(
                [path, "-version"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            first = (result.stdout or result.stderr or "").splitlines()
            match = _VERSION_RE.search(first[0] if first else "")
            version = match.group(1) if match else None
            available = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            available = False
        return MediaToolInfo(available=available, path=path if available else None, version=version)

    def capabilities(self) -> MediaToolsCapabilities:
        return MediaToolsCapabilities(
            ffmpeg=self._tool_info(self._resolve(self._settings.ffmpeg_path, "ffmpeg")),
            ffprobe=self._tool_info(self._resolve(self._settings.ffprobe_path, "ffprobe")),
        )

    def require_ffmpeg(self) -> str:
        path = self._resolve(self._settings.ffmpeg_path, "ffmpeg")
        if not path:
            raise FFmpegNotFoundError("FFmpeg was not found. Install FFmpeg or configure FFMPEG_PATH.")
        return path

    def require_ffprobe(self) -> str:
        path = self._resolve(self._settings.ffprobe_path, "ffprobe")
        if not path:
            raise FFprobeNotFoundError("ffprobe was not found. Install FFmpeg or configure FFPROBE_PATH.")
        return path

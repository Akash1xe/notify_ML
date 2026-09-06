from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from app.core.config import AppSettings
from app.core.exceptions import (
    CorruptMediaError,
    MediaInspectionError,
    MediaProbeTimeoutError,
    VideoStreamMissingError,
)
from app.media.models import AudioStreamInfo, FileFingerprint, MediaInspection, VideoStreamInfo
from app.media.tools import MediaToolsService


def fingerprint(path: Path) -> FileFingerprint:
    stat = path.stat()
    return FileFingerprint(file_size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def parse_fps(value: Any) -> float | None:
    if value in (None, "", "0/0"):
        return None
    try:
        if isinstance(value, (int, float)):
            result = float(value)
        else:
            result = float(Fraction(str(value)))
        return result if result > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _select_stream(streams: list[dict[str, Any]], codec_type: str) -> dict[str, Any] | None:
    candidates = [stream for stream in streams if stream.get("codec_type") == codec_type]
    if not candidates:
        return None
    defaults = [stream for stream in candidates if (stream.get("disposition") or {}).get("default") == 1]
    return (defaults or candidates)[0]


def parse_probe_payload(
    payload: dict[str, Any],
    *,
    file_path: str,
    source_fingerprint: FileFingerprint,
    require_video: bool = True,
) -> MediaInspection:
    streams = [item for item in payload.get("streams", []) if isinstance(item, dict)]
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    video_raw = _select_stream(streams, "video")
    audio_raw = _select_stream(streams, "audio")
    if require_video and video_raw is None:
        raise VideoStreamMissingError("Downloaded media does not contain a video stream.")

    format_duration = _float(fmt.get("duration"))
    stream_durations = [
        value
        for value in (_float((video_raw or {}).get("duration")), _float((audio_raw or {}).get("duration")))
        if value is not None
    ]
    duration = format_duration if format_duration is not None else (max(stream_durations) if stream_durations else None)

    rotation = None
    if video_raw:
        rotation = _int((video_raw.get("tags") or {}).get("rotate"))
        if rotation is None:
            for item in video_raw.get("side_data_list", []) or []:
                if isinstance(item, dict) and item.get("rotation") is not None:
                    rotation = _int(item.get("rotation"))
                    break

    video = None
    if video_raw:
        video = VideoStreamInfo(
            codec=video_raw.get("codec_name"),
            codec_profile=video_raw.get("profile"),
            width=_int(video_raw.get("width")),
            height=_int(video_raw.get("height")),
            fps=parse_fps(video_raw.get("avg_frame_rate") or video_raw.get("r_frame_rate")),
            pixel_format=video_raw.get("pix_fmt"),
            bitrate=_int(video_raw.get("bit_rate")),
            duration_seconds=_float(video_raw.get("duration")),
            rotation=rotation,
        )

    audio = None
    if audio_raw:
        audio = AudioStreamInfo(
            codec=audio_raw.get("codec_name"),
            sample_rate=_int(audio_raw.get("sample_rate")),
            channels=_int(audio_raw.get("channels")),
            channel_layout=audio_raw.get("channel_layout"),
            bitrate=_int(audio_raw.get("bit_rate")),
            duration_seconds=_float(audio_raw.get("duration")),
        )

    return MediaInspection(
        file_path=file_path,
        container=fmt.get("format_name"),
        duration_seconds=duration,
        file_size_bytes=source_fingerprint.file_size_bytes,
        overall_bitrate=_int(fmt.get("bit_rate")),
        video=video,
        audio=audio,
        source_fingerprint=source_fingerprint,
    )


class MediaInspector:
    def __init__(self, settings: AppSettings, tools: MediaToolsService) -> None:
        self._settings = settings
        self._tools = tools

    def inspect(self, path: Path, *, logical_path: str, require_video: bool = True) -> MediaInspection:
        if not path.exists() or path.stat().st_size <= 0:
            raise CorruptMediaError("Media file is missing or empty.")
        ffprobe = self._tools.require_ffprobe()
        command = [
            ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._settings.media_probe_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaProbeTimeoutError("Media inspection timed out.") from exc
        except OSError as exc:
            raise MediaInspectionError("Unable to execute ffprobe.") from exc
        if result.returncode != 0:
            raise CorruptMediaError("ffprobe could not read the media file.")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MediaInspectionError("ffprobe returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise MediaInspectionError("ffprobe returned an invalid payload.")
        return parse_probe_payload(
            payload,
            file_path=logical_path,
            source_fingerprint=fingerprint(path),
            require_video=require_video,
        )

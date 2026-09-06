from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Callable

from app.core.config import AppSettings
from app.core.exceptions import (
    AudioExtractionError,
    AudioExtractionTimeoutError,
    AudioOutputMissingError,
    AudioStreamMissingError,
    AudioValidationError,
    InsufficientDiskSpaceError,
    JobCancelledError,
)
from app.media.models import AudioResult, FileFingerprint, MediaInspection
from app.media.probe import MediaInspector, fingerprint
from app.media.tools import MediaToolsService
from app.storage.workspace import atomic_write_json


ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def build_audio_command(ffmpeg: str, source: Path, target: Path) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(target),
    ]


def _parse_hms(value: str) -> float | None:
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, TypeError):
        return None


def parse_ffmpeg_progress_value(key: str, value: str) -> float | None:
    try:
        if key in {"out_time_us", "out_time_ms"}:
            return max(0.0, float(value) / 1_000_000.0)
        if key == "out_time":
            return _parse_hms(value)
    except ValueError:
        return None
    return None


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def _drain_stderr(stream: asyncio.StreamReader | None, max_lines: int = 80) -> deque[str]:
    lines: deque[str] = deque(maxlen=max_lines)
    if stream is None:
        return lines
    while True:
        raw = await stream.readline()
        if not raw:
            break
        lines.append(raw.decode(errors="replace").strip())
    return lines


class AudioExtractor:
    def __init__(
        self,
        settings: AppSettings,
        tools: MediaToolsService,
        inspector: MediaInspector,
    ) -> None:
        self._settings = settings
        self._tools = tools
        self._inspector = inspector

    def _check_disk(self, directory: Path, duration: float) -> None:
        expected = int(duration * 32_000) + 4096
        margin = self._settings.audio_disk_safety_margin_mb * 1024**2
        if shutil.disk_usage(directory).free < expected + margin:
            raise InsufficientDiskSpaceError("There is not enough disk space to create transcription audio.")

    def _validate_audio(self, media: MediaInspection, source_duration: float) -> None:
        if media.audio is None:
            raise AudioValidationError("Extracted WAV does not contain an audio stream.")
        if media.audio.codec != "pcm_s16le":
            raise AudioValidationError("Extracted audio codec is not PCM signed 16-bit.")
        if media.audio.sample_rate != 16000:
            raise AudioValidationError("Extracted audio sample rate is not 16 kHz.")
        if media.audio.channels != 1:
            raise AudioValidationError("Extracted audio is not mono.")
        if media.duration_seconds is None:
            raise AudioValidationError("Extracted audio duration is unavailable.")
        tolerance = max(2.0, source_duration * self._settings.audio_duration_tolerance_ratio)
        if abs(media.duration_seconds - source_duration) > tolerance:
            raise AudioValidationError("Extracted audio duration does not match the source lecture.")

    async def extract(
        self,
        *,
        source_path: Path,
        source_media: MediaInspection,
        video_id: str,
        source_fingerprint: FileFingerprint,
        temp_path: Path,
        final_path: Path,
        manifest_path: Path,
        logical_final_path: str,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> AudioResult:
        if source_media.audio is None:
            raise AudioStreamMissingError("The lecture does not contain an audio stream for transcription.")
        duration = source_media.duration_seconds
        if duration is None or duration <= 0:
            raise AudioExtractionError("The lecture duration is unavailable for audio extraction.")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_disk(final_path.parent, duration)
        temp_path.unlink(missing_ok=True)
        ffmpeg = self._tools.require_ffmpeg()
        command = build_audio_command(ffmpeg, source_path, temp_path)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioExtractionError("Unable to start FFmpeg audio extraction.") from exc

        stderr_task = asyncio.create_task(_drain_stderr(process.stderr))
        timeout = max(300.0, duration * self._settings.audio_extraction_timeout_multiplier)
        started = time.monotonic()
        try:
            while True:
                if cancel_check():
                    await _terminate_process(process)
                    raise JobCancelledError("Audio extraction was cancelled.")
                if time.monotonic() - started > timeout:
                    await _terminate_process(process)
                    raise AudioExtractionTimeoutError("Audio extraction exceeded its allowed runtime.")
                try:
                    raw = await asyncio.wait_for(process.stdout.readline(), timeout=0.5)  # type: ignore[union-attr]
                except asyncio.TimeoutError:
                    if process.returncode is not None:
                        break
                    continue
                if not raw:
                    break
                line = raw.decode(errors="replace").strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                current = parse_ffmpeg_progress_value(key, value)
                if current is not None and duration > 0:
                    progress_callback(min(100.0, max(0.0, current * 100.0 / duration)))
            return_code = await process.wait()
            stderr_lines = await stderr_task
            if return_code != 0:
                detail = " | ".join(list(stderr_lines)[-5:])
                raise AudioExtractionError(
                    "FFmpeg could not extract lecture audio." + (f" ({detail[:500]})" if detail else "")
                )
            if not temp_path.exists() or temp_path.stat().st_size <= 0:
                raise AudioOutputMissingError("FFmpeg completed without producing an audio file.")

            inspected = self._inspector.inspect(
                temp_path,
                logical_path="audio/audio.tmp.wav",
                require_video=False,
            )
            self._validate_audio(inspected, duration)
            os.replace(temp_path, final_path)
            audio_fp = fingerprint(final_path)
            result = AudioResult(
                path=logical_final_path,
                duration_seconds=inspected.duration_seconds or duration,
                file_size_bytes=audio_fp.file_size_bytes,
                source_video_id=video_id,
                source_fingerprint=source_fingerprint,
                audio_fingerprint=audio_fp,
            )
            atomic_write_json(manifest_path, result.model_dump(mode="json"))
            progress_callback(100.0)
            return result
        except BaseException:
            if process.returncode is None:
                await _terminate_process(process)
            if not stderr_task.done():
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
            temp_path.unlink(missing_ok=True)
            raise

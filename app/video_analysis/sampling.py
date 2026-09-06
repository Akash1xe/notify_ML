from __future__ import annotations

import asyncio
import math
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Callable

from app.core.config import AppSettings
from app.core.exceptions import (
    FrameOutputMissingError,
    FrameSamplingError,
    FrameSamplingTimeoutError,
    InsufficientDiskSpaceError,
    JobCancelledError,
)
from app.media.models import MediaInspection
from app.media.probe import fingerprint
from app.media.tools import MediaToolsService
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import SampledFrame, SamplingManifest


SAMPLING_ALGORITHM_VERSION = "1"
ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def sampling_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": SAMPLING_ALGORITHM_VERSION,
        "sample_fps": settings.frame_sample_fps,
        "image_format": "jpg",
        "jpeg_quality": settings.frame_jpeg_quality,
    }


def sampling_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(sampling_config_payload(settings))


def expected_frame_count(duration_seconds: float, sample_fps: float) -> int:
    if duration_seconds <= 0 or sample_fps <= 0:
        raise ValueError("Duration and sample FPS must be positive")
    return max(1, int(math.ceil(duration_seconds * sample_fps - 1e-9)))


def _jpeg_qscale(quality: int) -> int:
    # FFmpeg q:v uses 2 (high quality) through 31 (low quality).
    quality = min(100, max(1, quality))
    return min(31, max(2, int(round(31 - (quality / 100.0) * 29))))


def build_frame_sampling_command(
    ffmpeg: str,
    source: Path,
    target_pattern: Path,
    sample_fps: float,
    jpeg_quality: int,
) -> list[str]:
    fps_expr = f"fps=fps={sample_fps:g}:start_time=0:round=near"
    return [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        fps_expr,
        "-q:v",
        str(_jpeg_qscale(jpeg_quality)),
        "-start_number",
        "1",
        "-progress",
        "pipe:1",
        "-nostats",
        str(target_pattern),
    ]


def sampling_artifact_fingerprint(manifest: SamplingManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "source_video": manifest.source_video,
            "source_fingerprint": manifest.source_fingerprint.model_dump(mode="json"),
            "sample_fps": manifest.sample_fps,
            "interval_seconds": manifest.interval_seconds,
            "video_duration_seconds": manifest.video_duration_seconds,
            "actual_frame_count": manifest.actual_frame_count,
            "image_format": manifest.image_format,
            "jpeg_quality": manifest.jpeg_quality,
            "config_fingerprint": manifest.config_fingerprint,
            "frames": [
                {
                    "index": frame.index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "relative_path": frame.relative_path,
                    "file_size_bytes": frame.file_size_bytes,
                }
                for frame in manifest.frames
            ],
        }
    )


def _parse_time_progress(line: str) -> float | None:
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    try:
        if key in {"out_time_us", "out_time_ms"}:
            return max(0.0, float(value) / 1_000_000.0)
        if key == "out_time":
            hours, minutes, seconds = value.split(":")
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, TypeError):
        return None
    return None


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def _drain_stderr(stream: asyncio.StreamReader | None, max_lines: int = 60) -> deque[str]:
    lines: deque[str] = deque(maxlen=max_lines)
    if stream is None:
        return lines
    while True:
        raw = await stream.readline()
        if not raw:
            break
        lines.append(raw.decode(errors="replace").strip())
    return lines


class FrameSampler:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        tools: MediaToolsService,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._tools = tools

    def _validate_plan(self, duration: float, destination: Path) -> int:
        if self._settings.frame_sample_fps > self._settings.frame_sample_max_fps:
            raise FrameSamplingError("Configured frame sampling FPS exceeds the allowed maximum.")
        count = expected_frame_count(duration, self._settings.frame_sample_fps)
        if count > self._settings.max_sampled_frames:
            raise FrameSamplingError(
                f"Frame sampling would create {count} frames, above MAX_SAMPLED_FRAMES={self._settings.max_sampled_frames}."
            )
        estimate = count * self._settings.frame_estimated_size_kb * 1024
        margin = self._settings.frame_disk_safety_margin_mb * 1024**2
        destination.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination).free < estimate + margin:
            raise InsufficientDiskSpaceError("There is not enough disk space to sample lecture frames.")
        return count

    async def sample(
        self,
        *,
        job_id: str,
        source_path: Path,
        source_media: MediaInspection,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> SamplingManifest:
        duration = source_media.duration_seconds
        if duration is None or duration <= 0:
            raise FrameSamplingError("Lecture duration is unavailable for frame sampling.")
        if source_media.video is None:
            raise FrameSamplingError("Lecture video stream is unavailable for frame sampling.")

        frames_root = self._workspace.frames_dir(job_id)
        expected = self._validate_plan(duration, frames_root)
        temp_dir = self._workspace.sampled_frames_temp_dir(job_id)
        final_dir = self._workspace.sampled_frames_dir(job_id)
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        target = temp_dir / "frame_%08d.jpg"
        ffmpeg = self._tools.require_ffmpeg()
        command = build_frame_sampling_command(
            ffmpeg,
            source_path,
            target,
            self._settings.frame_sample_fps,
            self._settings.frame_jpeg_quality,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise FrameSamplingError("Unable to start FFmpeg frame sampling.") from exc

        stderr_task = asyncio.create_task(_drain_stderr(process.stderr))
        timeout = max(
            float(self._settings.frame_sampling_base_timeout_seconds),
            duration * self._settings.frame_sampling_timeout_multiplier,
        )
        started = time.monotonic()
        try:
            while True:
                if cancel_check():
                    await _terminate(process)
                    raise JobCancelledError("Frame sampling was cancelled.")
                if time.monotonic() - started > timeout:
                    await _terminate(process)
                    raise FrameSamplingTimeoutError("Frame sampling exceeded its allowed runtime.")
                try:
                    raw = await asyncio.wait_for(process.stdout.readline(), timeout=0.5)  # type: ignore[union-attr]
                except asyncio.TimeoutError:
                    if process.returncode is not None:
                        break
                    continue
                if not raw:
                    break
                current = _parse_time_progress(raw.decode(errors="replace").strip())
                if current is not None:
                    progress_callback(min(100.0, current * 100.0 / duration))

            return_code = await process.wait()
            stderr_lines = await stderr_task
            if return_code != 0:
                detail = " | ".join(list(stderr_lines)[-4:])
                raise FrameSamplingError(
                    "FFmpeg could not sample lecture frames."
                    + (f" ({detail[:500]})" if detail else "")
                )

            files = sorted(temp_dir.glob("frame_*.jpg"))
            if not files or any(path.stat().st_size <= 0 for path in files):
                raise FrameOutputMissingError("FFmpeg completed without a valid sampled frame set.")
            # Small deviations are legitimate at fractional durations, but a huge
            # mismatch means the extraction or duration metadata is inconsistent.
            tolerance = max(2, int(math.ceil(expected * 0.02)))
            if abs(len(files) - expected) > tolerance:
                raise FrameOutputMissingError(
                    f"Sampled frame count {len(files)} differs unexpectedly from estimate {expected}."
                )

            records = [
                SampledFrame(
                    index=index,
                    timestamp_seconds=round((index - 1) / self._settings.frame_sample_fps, 6),
                    relative_path=f"frames/sampled/{path.name}",
                    file_size_bytes=path.stat().st_size,
                )
                for index, path in enumerate(files, start=1)
            ]
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temp_dir, final_dir)
            source_fp = fingerprint(source_path)
            manifest = SamplingManifest(
                algorithm_version=SAMPLING_ALGORITHM_VERSION,
                source_video=self._workspace.relative_to_workspace(job_id, source_path),
                source_fingerprint=source_fp,
                sample_fps=self._settings.frame_sample_fps,
                interval_seconds=1.0 / self._settings.frame_sample_fps,
                video_duration_seconds=duration,
                expected_frame_count=expected,
                actual_frame_count=len(records),
                jpeg_quality=self._settings.frame_jpeg_quality,
                config_fingerprint=sampling_config_fingerprint(self._settings),
                artifact_fingerprint="pending",
                frames=records,
                sampling_seconds=round(time.monotonic() - started, 3),
            )
            manifest.artifact_fingerprint = sampling_artifact_fingerprint(manifest)
            atomic_write_json(
                self._workspace.frame_manifest_path(job_id),
                manifest.model_dump(mode="json"),
            )
            progress_callback(100.0)
            return manifest
        except BaseException:
            if process.returncode is None:
                await _terminate(process)
            if not stderr_task.done():
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import AppSettings
from app.video_analysis.sampling import (
    build_frame_sampling_command,
    expected_frame_count,
    sampling_config_fingerprint,
)


def test_sampling_defaults(tmp_path: Path):
    settings = AppSettings(storage_root=tmp_path / "jobs")
    assert settings.frame_sample_fps == 1.0
    assert settings.frame_sample_max_fps == 5.0
    assert settings.max_sampled_frames == 20000


@pytest.mark.parametrize(
    ("duration", "fps", "expected"),
    [(10, 1, 10), (10, 2, 20), (10, 0.5, 5), (3600, 1, 3600), (10.2, 1, 11)],
)
def test_expected_frame_count(duration, fps, expected):
    assert expected_frame_count(duration, fps) == expected


@pytest.mark.parametrize(("duration", "fps"), [(0, 1), (10, 0), (-1, 1), (1, -2)])
def test_expected_frame_count_rejects_invalid_values(duration, fps):
    with pytest.raises(ValueError):
        expected_frame_count(duration, fps)


def test_sampling_fps_cannot_exceed_max(tmp_path: Path):
    with pytest.raises(ValidationError):
        AppSettings(
            storage_root=tmp_path / "jobs",
            frame_sample_fps=6,
            frame_sample_max_fps=5,
        )


def test_sampling_command_is_deterministic(tmp_path: Path):
    command = build_frame_sampling_command(
        "ffmpeg",
        tmp_path / "video.mp4",
        tmp_path / "frame_%08d.jpg",
        2.0,
        90,
    )
    assert command[0] == "ffmpeg"
    assert "fps=fps=2:start_time=0:round=near" in command
    assert "-start_number" in command
    assert "-progress" in command
    assert command[-1].endswith("frame_%08d.jpg")


def test_sampling_config_fingerprint_changes_with_fps(tmp_path: Path):
    first = AppSettings(storage_root=tmp_path / "a", frame_sample_fps=1)
    second = AppSettings(storage_root=tmp_path / "b", frame_sample_fps=2)
    assert sampling_config_fingerprint(first) != sampling_config_fingerprint(second)

import asyncio
from types import SimpleNamespace

import cv2

from app.core.exceptions import FrameSamplingError, FrameSamplingTimeoutError, InsufficientDiskSpaceError, JobCancelledError
from app.media.models import MediaInspection, VideoStreamInfo
from app.media.probe import fingerprint
from app.video_analysis.sampling import FrameSampler
from tests.phase3_helpers import build_phase3_workspace, synthetic_frame


class _EmptyStream:
    async def readline(self):
        return b""


class _FakeProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = _EmptyStream()
        self.stderr = _EmptyStream()
        self.terminated = False
        self.killed = False

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9


class _FakeTools:
    def require_ffmpeg(self):
        return "ffmpeg"


def _source_media(source: Path, duration=2.0):
    return MediaInspection(
        file_path="source/video.mp4",
        duration_seconds=duration,
        file_size_bytes=source.stat().st_size,
        video=VideoStreamInfo(codec="h264", width=640, height=360, fps=30),
        source_fingerprint=fingerprint(source),
    )


def test_sampler_rejects_excessive_frame_count(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(
        tmp_path, frame_sample_fps=5, max_sampled_frames=10, frame_estimated_size_kb=1
    )
    source = workspace.source_dir(job_id) / "video.mp4"
    source.write_bytes(b"video")
    sampler = FrameSampler(settings, workspace, _FakeTools())
    with pytest.raises(FrameSamplingError):
        asyncio.run(
            sampler.sample(
                job_id=job_id, source_path=source, source_media=_source_media(source, duration=10),
                progress_callback=lambda _: None, cancel_check=lambda: False,
            )
        )


def test_sampler_checks_disk_before_starting_ffmpeg(tmp_path: Path, monkeypatch):
    settings, workspace, job_id = build_phase3_workspace(
        tmp_path, frame_estimated_size_kb=1000, frame_disk_safety_margin_mb=0
    )
    source = workspace.source_dir(job_id) / "video.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("app.video_analysis.sampling.shutil.disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(InsufficientDiskSpaceError):
        asyncio.run(
            FrameSampler(settings, workspace, _FakeTools()).sample(
                job_id=job_id, source_path=source, source_media=_source_media(source),
                progress_callback=lambda _: None, cancel_check=lambda: False,
            )
        )


def test_sampler_success_promotes_temp_directory_atomically(tmp_path: Path, monkeypatch):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, frame_disk_safety_margin_mb=0)
    source = workspace.source_dir(job_id) / "video.mp4"
    source.write_bytes(b"video")

    async def fake_exec(*command, **kwargs):
        pattern = command[-1]
        for index in (1, 2):
            path = Path(pattern % index)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), synthetic_frame("text" if index == 2 else "blank"))
        return _FakeProcess(0)

    monkeypatch.setattr("app.video_analysis.sampling.asyncio.create_subprocess_exec", fake_exec)
    result = asyncio.run(
        FrameSampler(settings, workspace, _FakeTools()).sample(
            job_id=job_id, source_path=source, source_media=_source_media(source),
            progress_callback=lambda _: None, cancel_check=lambda: False,
        )
    )
    assert result.actual_frame_count == 2
    assert workspace.sampled_frames_dir(job_id).exists()
    assert not workspace.sampled_frames_temp_dir(job_id).exists()
    assert workspace.frame_manifest_path(job_id).exists()


def test_sampler_cancellation_removes_partial_directory(tmp_path: Path, monkeypatch):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, frame_disk_safety_margin_mb=0)
    source = workspace.source_dir(job_id) / "video.mp4"
    source.write_bytes(b"video")
    process = _FakeProcess(None)

    async def fake_exec(*args, **kwargs):
        return process

    monkeypatch.setattr("app.video_analysis.sampling.asyncio.create_subprocess_exec", fake_exec)
    with pytest.raises(JobCancelledError):
        asyncio.run(
            FrameSampler(settings, workspace, _FakeTools()).sample(
                job_id=job_id, source_path=source, source_media=_source_media(source),
                progress_callback=lambda _: None, cancel_check=lambda: True,
            )
        )
    assert process.terminated
    assert not workspace.sampled_frames_temp_dir(job_id).exists()


def test_sampler_timeout_terminates_process(tmp_path: Path, monkeypatch):
    settings, workspace, job_id = build_phase3_workspace(
        tmp_path, frame_disk_safety_margin_mb=0, frame_sampling_base_timeout_seconds=1,
        frame_sampling_timeout_multiplier=0.25,
    )
    source = workspace.source_dir(job_id) / "video.mp4"
    source.write_bytes(b"video")
    process = _FakeProcess(None)

    async def fake_exec(*args, **kwargs):
        return process

    class SlowStream:
        async def readline(self):
            await asyncio.sleep(0.6)
            return b""

    process.stdout = SlowStream()
    monkeypatch.setattr("app.video_analysis.sampling.asyncio.create_subprocess_exec", fake_exec)
    with pytest.raises(FrameSamplingTimeoutError):
        asyncio.run(
            FrameSampler(settings, workspace, _FakeTools()).sample(
                job_id=job_id, source_path=source, source_media=_source_media(source),
                progress_callback=lambda _: None, cancel_check=lambda: False,
            )
        )
    assert process.terminated
    assert not workspace.sampled_frames_temp_dir(job_id).exists()

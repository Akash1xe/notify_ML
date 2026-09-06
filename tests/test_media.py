from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import AppSettings
from app.core.exceptions import VideoStreamMissingError
from app.media.audio import build_audio_command, parse_ffmpeg_progress_value
from app.media.models import FileFingerprint
from app.media.probe import parse_fps, parse_probe_payload


def probe_payload():
    return {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "3620.52",
            "bit_rate": "1136000",
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "High",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "pix_fmt": "yuv420p",
                "disposition": {"default": 1},
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "disposition": {"default": 1},
            },
        ],
    }


def test_parse_fps_fraction():
    assert parse_fps("30/1") == 30.0
    assert parse_fps("30000/1001") == pytest.approx(29.970, rel=1e-3)
    assert parse_fps("0/0") is None


def test_probe_payload_normalization():
    result = parse_probe_payload(
        probe_payload(),
        file_path="source/video.mp4",
        source_fingerprint=FileFingerprint(file_size_bytes=123, mtime_ns=9),
    )
    assert result.duration_seconds == 3620.52
    assert result.video is not None and result.video.height == 1080
    assert result.video.fps == pytest.approx(29.970, rel=1e-3)
    assert result.audio is not None and result.audio.sample_rate == 48000


def test_probe_payload_requires_video():
    payload = probe_payload()
    payload["streams"] = [payload["streams"][1]]
    with pytest.raises(VideoStreamMissingError):
        parse_probe_payload(
            payload,
            file_path="source/audio.m4a",
            source_fingerprint=FileFingerprint(file_size_bytes=123, mtime_ns=9),
        )


def test_audio_command_is_normalized_and_shell_safe(tmp_path: Path):
    command = build_audio_command("ffmpeg", tmp_path / "video.mp4", tmp_path / "audio.wav")
    assert "-vn" in command
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"


def test_ffmpeg_progress_parser():
    assert parse_ffmpeg_progress_value("out_time_us", "1500000") == 1.5
    assert parse_ffmpeg_progress_value("out_time", "00:01:02.500") == 62.5
    assert parse_ffmpeg_progress_value("junk", "x") is None

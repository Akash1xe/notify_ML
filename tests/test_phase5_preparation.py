from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import AudioPreparationError
from app.media.models import AudioResult
from app.media.probe import fingerprint
from app.storage.workspace import atomic_write_json
from app.transcription.preparation import CP_AUDIO_READY
from tests.phase5_helpers import build_phase5_context, build_phase5_pipeline, write_wav


def test_phase51_valid_audio_creates_preparation_and_chunks(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    _, _, repository, preparation, *_ = build_phase5_pipeline(ctx, adapter)
    job = ctx[4]
    manifest = preparation.process(job.id)
    assert manifest.is_valid
    assert manifest.audio.sample_rate_hz == 16000
    assert manifest.audio.channels == 1
    assert manifest.audio.bits_per_sample == 16
    assert len(manifest.chunks) == 4
    assert manifest.chunks[1].decode_start_seconds < manifest.chunks[1].logical_start_seconds
    assert repository.load_preparation(job.id).artifact_fingerprint == manifest.artifact_fingerprint


def test_phase51_wrong_sample_rate_is_rejected(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    settings, workspace, *_ = ctx
    job = ctx[4]
    path = workspace.audio_path(job.id)
    write_wav(path, 20.0, sample_rate=44100)
    audio = AudioResult(
        path="audio/audio.wav",
        duration_seconds=20.0,
        file_size_bytes=path.stat().st_size,
        source_video_id="dQw4w9WgXcQ",
        source_fingerprint=fingerprint(workspace.source_dir(job.id) / "video.mp4"),
        audio_fingerprint=fingerprint(path),
    )
    atomic_write_json(workspace.audio_manifest_path(job.id), audio.model_dump(mode="json"))
    _, _, repository, preparation, *_ = build_phase5_pipeline(ctx, adapter)
    with pytest.raises(AudioPreparationError):
        preparation.process(job.id)
    cached = repository.load_preparation(job.id)
    assert "INVALID_SAMPLE_RATE" in cached.rejection_reasons


def test_phase51_effectively_silent_audio_is_rejected(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path, silent=True)
    _, _, repository, preparation, *_ = build_phase5_pipeline(ctx, adapter)
    with pytest.raises(AudioPreparationError):
        preparation.process(ctx[4].id)
    cached = repository.load_preparation(ctx[4].id)
    assert "AUDIO_EFFECTIVELY_SILENT" in cached.rejection_reasons


def test_phase51_chunk_plan_covers_full_audio_without_logical_overlap(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path, settings_overrides={"transcription_chunk_seconds": 7.0, "transcription_chunk_overlap_seconds": 2.0})
    _, _, _, preparation, *_ = build_phase5_pipeline(ctx, adapter)
    manifest = preparation.process(ctx[4].id)
    assert [(c.logical_start_seconds, c.logical_end_seconds) for c in manifest.chunks] == [
        (0.0, 7.0), (7.0, 14.0), (14.0, 20.0)
    ]
    assert manifest.chunks[-1].decode_end_seconds == 20.0


def test_phase51_candidate_annotation_does_not_reduce_coverage(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    _, _, _, preparation, *_ = build_phase5_pipeline(ctx, adapter)
    manifest = preparation.process(ctx[4].id)
    assert sum(c.duration_seconds for c in manifest.chunks) == pytest.approx(manifest.audio.duration_seconds)
    assert any(c.contains_candidate_timestamps for c in manifest.chunks)

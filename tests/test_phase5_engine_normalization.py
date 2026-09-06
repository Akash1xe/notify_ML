from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import AppSettings
from app.transcription.faster_whisper_adapter import FasterWhisperAdapter
from app.transcription.models import RawTranscriptManifest
from tests.phase5_helpers import FakeTranscriptionAdapter, build_phase5_context, build_phase5_pipeline


def test_faster_whisper_adapter_loads_lazily_and_reuses_model(monkeypatch, tmp_path: Path):
    init_calls = []
    class FakeModel:
        def __init__(self, *args, **kwargs):
            init_calls.append((args, kwargs))
        def transcribe(self, path, **kwargs):
            return iter([]), SimpleNamespace(language="en", language_probability=1.0)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    adapter = FasterWhisperAdapter(AppSettings(storage_root=tmp_path / "jobs"))
    assert not adapter.loaded
    list(adapter.transcribe("x.wav")[0])
    list(adapter.transcribe("y.wav")[0])
    assert adapter.loaded
    assert len(init_calls) == 1


def test_phase52_chunk_cache_reuse_skips_adapter(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    _, _, repository, preparation_service, engine, *_ = build_phase5_pipeline(ctx, adapter)
    prep = preparation_service.process(ctx[4].id)
    engine.process(ctx[4].id, preparation=prep)
    first_calls = adapter.calls
    engine.process(ctx[4].id, preparation=prep)
    assert adapter.calls == first_calls
    assert repository.load_raw_transcript(ctx[4].id).stats.completed_chunk_count == len(prep.chunks)


def test_phase52_absolute_timestamps_use_decode_start(tmp_path: Path):
    adapter = FakeTranscriptionAdapter(outputs=[[(1.0, 2.0, "hello")]])
    ctx, _ = build_phase5_context(tmp_path, adapter=adapter)
    _, _, _, preparation_service, engine, *_ = build_phase5_pipeline(ctx, adapter)
    prep = preparation_service.process(ctx[4].id)
    chunk = prep.chunks[1]
    result = engine.transcribe_chunk(job_id=ctx[4].id, preparation=prep, chunk=chunk, cancel_check=lambda: False)
    assert result.segments[0].absolute_start_seconds == pytest.approx(chunk.decode_start_seconds + 1.0)


def test_phase53_deduplicates_overlapping_chunk_text_but_preserves_far_repetition(tmp_path: Path):
    outputs = [
        [(5.2, 6.2, "Kafka sends the message.")],
        [(0.5, 2.0, "Kafka sends the message to the broker.")],
        [(3.0, 4.0, "Okay.")],
        [(0.5, 1.0, "Okay.")],
    ]
    adapter = FakeTranscriptionAdapter(outputs=outputs)
    ctx, _ = build_phase5_context(tmp_path, adapter=adapter)
    _, _, repository, preparation_service, engine, normalization, *_ = build_phase5_pipeline(ctx, adapter)
    prep = preparation_service.process(ctx[4].id)
    raw = engine.process(ctx[4].id, preparation=prep)
    transcript = normalization.process(ctx[4].id, raw=raw)
    texts = [s.text for s in transcript.segments]
    assert transcript.stats.deduplicated_segment_count >= 1
    assert any("broker" in text for text in texts)
    assert sum("Okay" in text for text in texts) >= 1
    assert repository.load_transcript(ctx[4].id).artifact_fingerprint == transcript.artifact_fingerprint


def test_phase53_unicode_is_preserved(tmp_path: Path):
    adapter = FakeTranscriptionAdapter(outputs=[[(1.0, 3.0, "अब Kafka broker message भेजता है")]])
    ctx, _ = build_phase5_context(tmp_path, adapter=adapter)
    _, _, _, prep_service, engine, normalization, *_ = build_phase5_pipeline(ctx, adapter)
    prep = prep_service.process(ctx[4].id)
    raw = engine.process(ctx[4].id, preparation=prep)
    transcript = normalization.process(ctx[4].id, raw=raw)
    assert any("अब Kafka" in s.text for s in transcript.segments)


def test_phase53_normalization_is_deterministic(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    _, _, _, prep_service, engine, normalization, *_ = build_phase5_pipeline(ctx, adapter)
    prep = prep_service.process(ctx[4].id)
    raw = engine.process(ctx[4].id, preparation=prep)
    first = normalization.process(ctx[4].id, raw=raw)
    second = normalization.process(ctx[4].id, raw=raw)
    assert first.artifact_fingerprint == second.artifact_fingerprint
    assert [s.model_dump() for s in first.segments] == [s.model_dump() for s in second.segments]

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError
from app.transcription.cache import TranscriptCacheCoordinator
from app.transcription.evaluation import Phase5Evaluator
from app.transcription.models import TranscriptResumeStage
from tests.phase5_helpers import FakeTranscriptionAdapter, build_phase5_context, build_phase5_pipeline


def test_phase54_alignment_and_phase55_contexts_are_created(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, cache, repository, *_ = build_phase5_pipeline(ctx, adapter)
    summary = asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    alignment = repository.load_alignment(ctx[4].id)
    contexts = repository.load_contexts(ctx[4].id)
    assert alignment.stats.candidate_count >= 1
    assert contexts.stats.context_count == alignment.stats.candidate_count
    assert summary.aligned_candidate_count == alignment.stats.candidate_count
    assert cache.inspect(ctx[4].id).resume_stage is TranscriptResumeStage.PHASE5_READY


def test_phase54_silent_candidate_is_not_rejected(tmp_path: Path):
    # Speech only near the beginning of each chunk leaves some visual candidates between speech.
    adapter = FakeTranscriptionAdapter(outputs=[[(0.1, 0.8, "brief speech")]])
    ctx, _ = build_phase5_context(tmp_path, adapter=adapter)
    pipeline, _, repository, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    alignment = repository.load_alignment(ctx[4].id)
    assert alignment.stats.candidate_count >= 1
    assert all(a.candidate_id > 0 for a in alignment.alignments)


def test_phase55_context_preserves_transcript_wording(tmp_path: Path):
    text = "अब Kafka broker message consumer को भेजता है"
    adapter = FakeTranscriptionAdapter(outputs=[[(1.0, 5.0, text)]])
    ctx, _ = build_phase5_context(tmp_path, adapter=adapter)
    pipeline, _, repository, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    contexts = repository.load_contexts(ctx[4].id)
    combined = " ".join(c.combined_text for c in contexts.contexts)
    assert "Kafka" in combined


def test_phase5_full_cache_reuse_skips_adapter(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, cache, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    calls = adapter.calls
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert adapter.calls == calls
    assert cache.inspect(ctx[4].id).resume_stage is TranscriptResumeStage.PHASE5_READY


def test_missing_raw_assembly_rebuilds_without_retranscribing_chunks(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, cache, repository, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    calls = adapter.calls
    ctx[1].raw_transcript_path(ctx[4].id).unlink()
    ctx[3].invalidate(ctx[4].id, "RAW_TRANSCRIPTION_COMPLETE")
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert adapter.calls == calls
    assert repository.load_raw_transcript(ctx[4].id).stats.raw_segment_count >= 1


def _pipeline_with_new_settings(ctx, adapter, **overrides):
    old = ctx[0]
    values = {
        "storage_root": old.storage_root,
        "processor_mode": "transcription",
        "transcription_chunk_seconds": old.transcription_chunk_seconds,
        "transcription_chunk_overlap_seconds": old.transcription_chunk_overlap_seconds,
        "frame_disk_safety_margin_mb": 0,
        "audio_disk_safety_margin_mb": 0,
        "download_disk_safety_margin_mb": 0,
    }
    values.update(overrides)
    new_settings = AppSettings(**values)
    mutable = list(ctx)
    mutable[0] = new_settings
    return build_phase5_pipeline(tuple(mutable), adapter)


def test_normalization_config_change_resumes_at_phase53(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    calls = adapter.calls
    pipeline2, cache2, *_ = _pipeline_with_new_settings(ctx, adapter, transcript_merge_max_gap_seconds=0.2)
    snapshot = cache2.inspect(ctx[4].id)
    assert snapshot.preparation.valid
    assert snapshot.raw_transcription.valid
    assert snapshot.resume_stage is TranscriptResumeStage.NORMALIZATION
    asyncio.run(pipeline2.process(ctx[4].id, finalize_job=False))
    assert adapter.calls == calls


def test_alignment_config_change_resumes_at_phase54(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    pipeline2, cache2, *_ = _pipeline_with_new_settings(ctx, adapter, alignment_proximity_saturation_seconds=20.0)
    assert cache2.inspect(ctx[4].id).resume_stage is TranscriptResumeStage.CANDIDATE_ALIGNMENT


def test_context_config_change_resumes_at_phase55(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    pipeline2, cache2, *_ = _pipeline_with_new_settings(ctx, adapter, transcript_context_before_seconds=20.0)
    assert cache2.inspect(ctx[4].id).resume_stage is TranscriptResumeStage.CONTEXT_EXTRACTION


def test_one_corrupt_raw_chunk_causes_partial_raw_resume(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, cache, _, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    chunk_path = ctx[1].raw_transcript_chunk_path(ctx[4].id, 2)
    chunk_path.write_text("{bad json", encoding="utf-8")
    snapshot = cache.inspect(ctx[4].id)
    assert snapshot.resume_stage is TranscriptResumeStage.RAW_TRANSCRIPTION
    assert 2 in snapshot.invalid_raw_chunk_ids
    calls = adapter.calls
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert adapter.calls == calls + 1


def test_no_speech_pipeline_completes_with_empty_contexts(tmp_path: Path):
    adapter = FakeTranscriptionAdapter(outputs=[[]])
    ctx, _ = build_phase5_context(tmp_path, adapter=adapter)
    pipeline, cache, repository, *_ = build_phase5_pipeline(ctx, adapter)
    summary = asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert repository.load_transcript(ctx[4].id).stats.normalized_segment_count == 0
    assert repository.load_contexts(ctx[4].id).stats.contexts_with_speech == 0
    assert summary.contexts_without_speech >= 1
    assert cache.inspect(ctx[4].id).resume_stage is TranscriptResumeStage.PHASE5_READY


def test_context_cancellation_preserves_upstream(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, _, repository, _, _, _, _, context = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    transcript = repository.load_transcript(ctx[4].id)
    alignment = repository.load_alignment(ctx[4].id)
    with pytest.raises(JobCancelledError):
        context.process(ctx[4].id, transcript=transcript, alignment=alignment, cancel_check=lambda: True)
    assert ctx[1].normalized_transcript_path(ctx[4].id).exists()
    assert ctx[1].candidate_alignment_path(ctx[4].id).exists()


def test_phase57_evaluator_reads_existing_outputs_without_transcription(tmp_path: Path):
    ctx, adapter = build_phase5_context(tmp_path)
    pipeline, _, repository, *_ = build_phase5_pipeline(ctx, adapter)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    calls = adapter.calls
    report = Phase5Evaluator(ctx[0], ctx[1], repository).evaluate(ctx[4].id, persist=True)
    assert report.chunk_count >= 1
    assert report.candidate_count >= 1
    assert adapter.calls == calls
    assert ctx[1].transcript_evaluation_path(ctx[4].id).exists()

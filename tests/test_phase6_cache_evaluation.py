from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import AppSettings
from app.semantic_analysis.models import Phase6ResumeStage
from app.semantic_analysis.runtime import FakeVisionLanguageModelAdapter
from tests.phase6_helpers import build_phase6_context, build_phase6_pipeline


def test_phase6_all_cache_hit_skips_vlm(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    pipeline, cache, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    calls = fake.generate_calls
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert fake.generate_calls == calls
    assert cache.inspect(ctx[4].id).resume_plan.resume_stage is Phase6ResumeStage.PHASE6_READY


def test_missing_one_candidate_result_runs_one_vlm_call(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    pipeline, _, repository, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    inputs = repository.load_input_manifest(ctx[4].id).inputs
    assert inputs
    before = fake.generate_calls
    ctx[1].semantic_candidate_result_path(ctx[4].id, inputs[0].candidate_id).unlink()
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert fake.generate_calls == before + 1


def test_missing_semantic_aggregate_rebuilds_without_vlm(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    pipeline, _, _, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    before = fake.generate_calls
    ctx[1].semantic_results_path(ctx[4].id).unlink()
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert fake.generate_calls == before


def test_missing_decision_artifact_rebuilds_without_vlm(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    pipeline, _, _, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    before = fake.generate_calls
    ctx[1].semantic_selections_path(ctx[4].id).unlink()
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert fake.generate_calls == before


def test_decision_weight_change_does_not_rerun_vlm(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    pipeline, _, _, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    before = fake.generate_calls
    old_settings = ctx[0]
    settings = AppSettings(
        storage_root=old_settings.storage_root,
        processor_mode="semantic",
        qwen_vl_device="cpu",
        transcription_chunk_seconds=old_settings.transcription_chunk_seconds,
        transcription_chunk_overlap_seconds=old_settings.transcription_chunk_overlap_seconds,
        semantic_decision_completion_weight=0.30,
        semantic_decision_phase4_weight=0.10,
    )
    ctx2 = (settings,) + ctx[1:]
    fake2 = FakeVisionLanguageModelAdapter()
    pipeline2, *_ = build_phase6_pipeline(ctx2, fake2)
    asyncio.run(pipeline2.process(ctx[4].id, finalize_job=False))
    assert fake2.generate_calls == 0
    assert fake.generate_calls == before


def test_phase6_evaluator_reads_existing_artifacts_without_vlm(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    pipeline, _, _, *rest = build_phase6_pipeline(ctx, fake)
    evaluator = rest[-1]
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    before = fake.generate_calls
    report = evaluator.evaluate(ctx[4].id, persist=True)
    assert report.semantic_input_count >= 1
    assert fake.generate_calls == before

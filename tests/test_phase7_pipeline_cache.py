from __future__ import annotations

import asyncio

from app.screenshots.cache import CP_FINAL_SCREENSHOTS_READY
from app.screenshots.models import Phase7ResumeStage
from tests.phase7_helpers import build_phase7_context, build_phase7_pipeline


def test_phase7_pipeline_end_to_end_and_all_cache_hit(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend)
    job=ctx[4].id
    first=asyncio.run(pipeline.process(job,finalize_job=False))
    calls=len(backend.calls)
    assert first.final_screenshot_count==1
    assert ctx[3].is_completed(job,CP_FINAL_SCREENSHOTS_READY)
    second=asyncio.run(pipeline.process(job,finalize_job=False))
    assert len(backend.calls)==calls
    assert second.final_screenshot_count==1
    assert cache.inspect(job).resume_plan.resume_stage is Phase7ResumeStage.PHASE7_READY


def test_missing_final_selection_requires_selection_only_resume(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    asyncio.run(pipeline.process(job,finalize_job=False))
    ctx[1].screenshot_final_selections_path(job).unlink()
    snap=cache.inspect(job)
    assert snap.resume_plan.resume_stage is Phase7ResumeStage.FINAL_SCREENSHOT_SELECTION
    assert not snap.resume_plan.candidate_ids_to_extract
    assert not snap.resume_plan.candidate_ids_to_quality_check
    assert not snap.resume_plan.candidate_ids_to_fingerprint


def test_missing_fingerprint_asset_backtracks_only_to_fingerprint_stage(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    asyncio.run(pipeline.process(job,finalize_job=False))
    f=repo.load_fingerprint_manifest(job).fingerprints[0]
    (ctx[1].workspace(job)/f.thumbnail_relative_path).unlink()
    snap=cache.inspect(job)
    assert snap.resume_plan.resume_stage is Phase7ResumeStage.FINGERPRINT_GENERATION
    assert snap.resume_plan.candidate_ids_to_extract==[]
    assert snap.resume_plan.candidate_ids_to_quality_check==[]
    assert snap.resume_plan.candidate_ids_to_fingerprint==[f.candidate_id]


def test_summary_rebuild_does_not_require_expensive_stage_work(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    asyncio.run(pipeline.process(job,finalize_job=False)); calls=len(backend.calls)
    ctx[1].screenshot_summary_path(job).unlink()
    asyncio.run(pipeline.process(job,finalize_job=False))
    assert len(backend.calls)==calls
    assert ctx[1].screenshot_summary_path(job).exists()


def test_phase7_evaluation_reads_existing_artifacts_only(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*rest=build_phase7_pipeline(ctx,sem,backend); evaluator=rest[-1]; job=ctx[4].id
    asyncio.run(pipeline.process(job,finalize_job=False)); calls=len(backend.calls)
    report=evaluator.evaluate(job,persist=True)
    assert report.final_screenshot_count==1
    assert len(backend.calls)==calls


def test_phase7_zero_semantic_winners_is_valid(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    sem.load_phase7_handoff=lambda job_id: []
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    summary=asyncio.run(pipeline.process(job,finalize_job=False))
    assert summary.final_screenshot_count==0
    assert backend.calls==[]
    assert cache.inspect(job).resume_plan.resume_stage is Phase7ResumeStage.PHASE7_READY


def test_source_video_change_invalidates_extraction_chain(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    asyncio.run(pipeline.process(job,finalize_job=False))
    source=ctx[1].workspace(job)/sem.load_phase7_handoff(job)[0].analysis_frame_path
    # Change the actual Phase-2 source video instead of analysis frame.
    import json
    from app.ingestion.models import IngestionResult
    ingestion=IngestionResult.model_validate(json.loads(ctx[1].ingestion_path(job).read_text()))
    video=ctx[1].workspace(job)/ingestion.source_video
    video.write_bytes(video.read_bytes()+b'changed')
    snap=cache.inspect(job)
    assert snap.resume_plan.resume_stage is Phase7ResumeStage.SOURCE_EXTRACTION


def test_quality_config_change_preserves_exact_extraction(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    asyncio.run(pipeline.process(job,finalize_job=False))
    new_settings=ctx[0].model_copy(update={'screenshot_min_quality_score':0.31})
    ctx2=(new_settings,)+ctx[1:]
    _,cache2,_,*_=build_phase7_pipeline(ctx2,sem,backend)
    snap=cache2.inspect(job)
    assert snap.extraction.valid
    assert snap.resume_plan.resume_stage is Phase7ResumeStage.QUALITY_VALIDATION
    assert snap.resume_plan.candidate_ids_to_extract==[]


def test_dedup_config_change_preserves_fingerprints(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    asyncio.run(pipeline.process(job,finalize_job=False))
    new_settings=ctx[0].model_copy(update={'duplicate_score_threshold':0.90})
    ctx2=(new_settings,)+ctx[1:]
    _,cache2,_,*_=build_phase7_pipeline(ctx2,sem,backend)
    snap=cache2.inspect(job)
    assert snap.extraction.valid and snap.quality.valid and snap.fingerprints.valid
    assert snap.resume_plan.resume_stage is Phase7ResumeStage.DUPLICATE_DETECTION


def test_final_selection_weight_change_preserves_duplicate_groups(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    asyncio.run(pipeline.process(job,finalize_job=False))
    new_settings=ctx[0].model_copy(update={'final_selection_semantic_weight':0.34,'final_selection_quality_weight':0.31})
    ctx2=(new_settings,)+ctx[1:]
    _,cache2,_,*_=build_phase7_pipeline(ctx2,sem,backend)
    snap=cache2.inspect(job)
    assert snap.duplicates.valid
    assert snap.resume_plan.resume_stage is Phase7ResumeStage.FINAL_SCREENSHOT_SELECTION

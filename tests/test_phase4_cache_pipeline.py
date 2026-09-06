from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.candidate_analysis.boundaries import StableBoundaryDetector
from app.candidate_analysis.cache import (
    CP_CANDIDATES_READY,
    CandidateAnalysisCacheManager,
)
from app.candidate_analysis.evaluation import Phase4Evaluator
from app.candidate_analysis.generation import CandidateGenerator
from app.candidate_analysis.heuristics import CandidateHeuristicAnalyzer
from app.candidate_analysis.models import CandidateResumeStage
from app.candidate_analysis.pipeline import CandidateAnalysisPipeline
from app.candidate_analysis.ranking import CandidateRankingService
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.candidate_analysis.stability import StabilityWindowDetector
from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError
from app.core.logging import JobEventLogger
from app.main import create_app
from app.video_analysis.cache import FrameAnalysisCacheManager
from app.video_analysis.repository import FrameAnalysisRepository
from tests.conftest import wait_for_status
from tests.phase4_helpers import build_phase4_context
from tests.test_phase3_pipeline import FakeFrameSampler
from tests.test_pipeline import FakeAudioExtractor, FakeMediaInspector, FakeMediaTools, FakeYouTube


class CountingService:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def process(self, **kwargs):
        self.calls += 1
        return self.inner.process(**kwargs)


def build_pipeline(ctx, *, counting: bool = False):
    settings, workspace, jobs, checkpoints, _, *_ = ctx
    frame_cache = FrameAnalysisCacheManager(settings, workspace, checkpoints)
    frame_repo = FrameAnalysisRepository(workspace)
    candidate_cache = CandidateAnalysisCacheManager(settings, workspace, checkpoints)
    candidate_repo = CandidateAnalysisRepository(workspace)
    services = [
        StabilityWindowDetector(settings, workspace),
        StableBoundaryDetector(settings, workspace),
        CandidateGenerator(settings, workspace),
        CandidateHeuristicAnalyzer(settings, workspace),
        CandidateRankingService(settings, workspace),
    ]
    if counting:
        services = [CountingService(item) for item in services]
    pipeline = CandidateAnalysisPipeline(
        settings=settings,
        jobs=jobs,
        workspace=workspace,
        checkpoints=checkpoints,
        events=JobEventLogger(workspace),
        frame_cache=frame_cache,
        frame_repository=frame_repo,
        stability=services[0],
        boundaries=services[1],
        generator=services[2],
        heuristics=services[3],
        ranking=services[4],
        cache=candidate_cache,
        repository=candidate_repo,
    )
    return pipeline, candidate_cache, candidate_repo, frame_repo, services


def phase3_kwargs(ctx):
    _, _, _, _, _, sampling, preprocessing, differences, major, timeline = ctx
    return {
        "sampling": sampling,
        "preprocessing": preprocessing,
        "differences": differences,
        "major_changes": major,
        "timeline": timeline,
    }


def test_phase47_pipeline_creates_complete_candidate_contract(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, cache, repo, _, _ = build_pipeline(ctx)
    job = ctx[4]
    summary = asyncio.run(pipeline.process(job.id, finalize_job=False))
    workspace = ctx[1]
    checkpoints = ctx[3]
    assert workspace.stability_windows_path(job.id).exists()
    assert workspace.boundaries_path(job.id).exists()
    assert workspace.generated_candidates_path(job.id).exists()
    assert workspace.scored_candidates_path(job.id).exists()
    assert workspace.ranked_candidates_path(job.id).exists()
    assert workspace.candidate_selections_path(job.id).exists()
    assert workspace.candidate_summary_path(job.id).exists()
    assert checkpoints.is_completed(job.id, CP_CANDIDATES_READY)
    assert summary.primary_candidate_count >= 1
    snapshot = cache.inspect(job.id, **phase3_kwargs(ctx))
    assert snapshot.resume_stage is CandidateResumeStage.PHASE4_READY
    assert snapshot.candidates_ready.valid
    assert repo.load_summary(job.id).selections_fingerprint == summary.selections_fingerprint


def test_full_phase4_reuse_skips_all_stage_services(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, _, _, _, services = build_pipeline(ctx, counting=True)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    assert [service.calls for service in services] == [1, 1, 1, 1, 1]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    assert [service.calls for service in services] == [1, 1, 1, 1, 1]


def test_ranking_config_change_invalidates_only_phase45(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, _, _, _, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    new_settings = AppSettings(
        storage_root=ctx[1].root,
        processor_mode="candidates",
        rank_quality_weight=0.40,
        rank_completeness_weight=0.10,
        frame_disk_safety_margin_mb=0,
        audio_disk_safety_margin_mb=0,
        download_disk_safety_margin_mb=0,
    )
    cache = CandidateAnalysisCacheManager(new_settings, ctx[1], ctx[3])
    snapshot = cache.inspect(job.id, **phase3_kwargs(ctx))
    assert snapshot.stability_windows.valid
    assert snapshot.boundaries.valid
    assert snapshot.generated_candidates.valid
    assert snapshot.scored_candidates.valid
    assert snapshot.resume_stage is CandidateResumeStage.CANDIDATE_RANKING


def test_generation_config_change_invalidates_phase43_and_downstream(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, _, _, _, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    new_settings = AppSettings(
        storage_root=ctx[1].root,
        processor_mode="candidates",
        candidate_settle_delay_seconds=1.4,
        frame_disk_safety_margin_mb=0,
        audio_disk_safety_margin_mb=0,
        download_disk_safety_margin_mb=0,
    )
    cache = CandidateAnalysisCacheManager(new_settings, ctx[1], ctx[3])
    snapshot = cache.inspect(job.id, **phase3_kwargs(ctx))
    assert snapshot.stability_windows.valid
    assert snapshot.boundaries.valid
    assert snapshot.resume_stage is CandidateResumeStage.CANDIDATE_GENERATION


def test_stability_config_change_invalidates_all_phase4(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, _, _, _, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    new_settings = AppSettings(
        storage_root=ctx[1].root,
        processor_mode="candidates",
        stability_window_min_duration_seconds=3.0,
        frame_disk_safety_margin_mb=0,
        audio_disk_safety_margin_mb=0,
        download_disk_safety_margin_mb=0,
    )
    cache = CandidateAnalysisCacheManager(new_settings, ctx[1], ctx[3])
    snapshot = cache.inspect(job.id, **phase3_kwargs(ctx))
    assert snapshot.resume_stage is CandidateResumeStage.STABILITY_WINDOWS


def test_phase3_timeline_change_invalidates_phase41(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, cache, _, _, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    kwargs = phase3_kwargs(ctx)
    changed_timeline = kwargs["timeline"].model_copy(deep=True)
    changed_timeline.artifact_fingerprint = "changed-phase3-timeline"
    kwargs["timeline"] = changed_timeline
    snapshot = cache.inspect(job.id, **kwargs)
    assert snapshot.resume_stage is CandidateResumeStage.STABILITY_WINDOWS


def test_corrupt_selection_output_resumes_ranking_only(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, cache, _, _, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    ctx[1].candidate_selections_path(job.id).write_text("{bad json", encoding="utf-8")
    snapshot = cache.inspect(job.id, **phase3_kwargs(ctx))
    assert snapshot.stability_windows.valid
    assert snapshot.scored_candidates.valid
    assert snapshot.resume_stage is CandidateResumeStage.CANDIDATE_RANKING


def test_partial_candidate_files_are_cleaned_without_removing_valid_final(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, cache, _, _, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    final = ctx[1].generated_candidates_path(job.id)
    temp = final.with_suffix(final.suffix + ".tmp")
    temp.write_text("partial", encoding="utf-8")
    removed = cache.cleanup_partial_artifacts(job.id)
    assert not temp.exists()
    assert final.exists()
    assert any("generated_candidates.json.tmp" in item for item in removed)


def test_phase4_progress_never_moves_backward_on_ranking_resume(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, _, _, _, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    ctx[2].update_progress(job.id, 98, "already high")
    ctx[1].ranked_candidates_path(job.id).unlink()
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    assert ctx[2].get_job(job.id).progress >= 98


def test_repository_handoff_contains_primary_and_alternate_metadata(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, _, repo, _, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    handoff = repo.load_handoff(job.id)
    assert handoff
    assert any(item.selection_role.value == "PRIMARY" for item in handoff)
    assert all(item.sampled_frame_path.startswith("frames/sampled/") for item in handoff)
    assert all(item.processed_frame_path.startswith("frames/processed/") for item in handoff)


def test_phase4_evaluator_reports_candidate_diagnostics(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, _, repo, frame_repo, _ = build_pipeline(ctx)
    job = ctx[4]
    asyncio.run(pipeline.process(job.id, finalize_job=False))
    evaluator = Phase4Evaluator(ctx[0], ctx[1], repo, frame_repo)
    report = evaluator.evaluate(job.id, persist=True)
    assert report.stability_windows == 1
    assert report.generated_candidates >= report.primary_candidates
    assert report.primary_density_per_minute >= 0
    assert ctx[1].candidate_evaluation_path(job.id).exists()


def test_pipeline_honors_cancellation_and_preserves_phase3(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    pipeline, _, _, _, _ = build_pipeline(ctx)
    job = ctx[4]
    ctx[2].cancel_job(job.id)
    with pytest.raises(JobCancelledError):
        asyncio.run(pipeline.process(job.id, finalize_job=False))
    assert ctx[1].timeline_path(job.id).exists()
    assert not ctx[1].stability_windows_path(job.id).exists()


def test_candidates_mode_fastapi_runs_phase2_phase3_phase4(tmp_path: Path):
    settings = AppSettings(
        storage_root=tmp_path / "jobs",
        processor_mode="candidates",
        max_concurrent_jobs=1,
        frame_disk_safety_margin_mb=0,
        audio_disk_safety_margin_mb=0,
        download_disk_safety_margin_mb=0,
        log_level="CRITICAL",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.ingestion_pipeline._youtube = FakeYouTube(app.state.workspace_manager)
        app.state.ingestion_pipeline._media_tools = FakeMediaTools()
        app.state.ingestion_pipeline._media_inspector = FakeMediaInspector()
        app.state.ingestion_pipeline._audio_extractor = FakeAudioExtractor()
        fake_sampler = FakeFrameSampler(settings, app.state.workspace_manager)
        app.state.frame_analysis_pipeline._sampler = fake_sampler
        created = client.post(
            "/api/jobs",
            json={"source_url": "https://youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        job_id = created.json()["id"]
        final = wait_for_status(client, job_id, "COMPLETED", timeout=5)
        assert final["message"] == "Visual candidates ready for transcript and semantic analysis"
        summary = client.get(f"/api/jobs/{job_id}/candidates")
        assert summary.status_code == 200
        assert "primary_candidate_count" in summary.json()
        cache = client.get(f"/api/jobs/{job_id}/candidates/cache")
        assert cache.status_code == 200
        assert cache.json()["candidates_ready"]["state"] == "VALID"
        selections = client.get(f"/api/jobs/{job_id}/candidates/selections")
        assert selections.status_code == 200
        assert fake_sampler.calls == 1


def test_candidate_endpoint_not_ready_in_phase3_only_mode(tmp_path: Path):
    settings = AppSettings(storage_root=tmp_path / "jobs", processor_mode="fake", fake_processor_step_delay=0)
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/jobs",
            json={"source_url": "https://youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        job_id = created.json()["id"]
        wait_for_status(client, job_id, "COMPLETED")
        assert client.get(f"/api/jobs/{job_id}/candidates").status_code == 404

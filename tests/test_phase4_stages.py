from __future__ import annotations

from pathlib import Path

import pytest

from app.candidate_analysis.boundaries import StableBoundaryDetector
from app.candidate_analysis.generation import CandidateGenerator
from app.candidate_analysis.heuristics import CandidateHeuristicAnalyzer
from app.candidate_analysis.models import (
    BoundaryRejectionReason,
    BoundaryType,
    CandidateSourceType,
    CandidateType,
    HeuristicRejectionReason,
    StabilityRejectionReason,
)
from app.candidate_analysis.ranking import CandidateRankingService
from app.candidate_analysis.stability import StabilityWindowDetector
from app.video_analysis.models import TimelineState
from tests.phase4_helpers import build_phase4_context


def run_windows(ctx):
    settings, workspace, _, _, job, _, preprocessing, differences, major, timeline = ctx
    return StabilityWindowDetector(settings, workspace).process(
        job_id=job.id,
        timeline=timeline,
        differences=differences,
        preprocessing=preprocessing,
        major_changes=major,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )


def run_boundaries(ctx, windows):
    settings, workspace, _, _, job, _, _, differences, major, timeline = ctx
    return StableBoundaryDetector(settings, workspace).process(
        job_id=job.id,
        windows=windows,
        timeline=timeline,
        differences=differences,
        major_changes=major,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )


def run_generated(ctx, windows, boundaries):
    settings, workspace, _, _, job, sampling, preprocessing, _, _, _ = ctx
    return CandidateGenerator(settings, workspace).process(
        job_id=job.id,
        windows=windows,
        boundaries=boundaries,
        sampling=sampling,
        preprocessing=preprocessing,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )


def run_scored(ctx, windows, boundaries, generated):
    settings, workspace, _, _, job, _, preprocessing, differences, _, timeline = ctx
    return CandidateHeuristicAnalyzer(settings, workspace).process(
        job_id=job.id,
        generated=generated,
        windows=windows,
        boundaries=boundaries,
        preprocessing=preprocessing,
        differences=differences,
        timeline=timeline,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )


def test_phase41_detects_sustained_stable_window(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    result = run_windows(ctx)
    assert result.stats.valid_stability_windows == 1
    window = result.windows[0]
    assert (window.start_timestamp_seconds, window.end_timestamp_seconds) == (8.0, 16.0)
    assert window.previous_segment_state is TimelineState.CHANGING
    assert 0 <= window.stability_score <= 1
    assert 0 <= window.quality_score <= 1


def test_phase41_rejects_short_stable_segment(tmp_path: Path):
    ctx = build_phase4_context(
        tmp_path,
        segments=[
            (TimelineState.CHANGING, 0, 4),
            (TimelineState.STABLE, 4, 5),
            (TimelineState.CHANGING, 5, 8),
        ],
    )
    result = run_windows(ctx)
    assert result.stats.valid_stability_windows == 0
    assert StabilityRejectionReason.TOO_SHORT in result.windows[0].rejection_reasons


def test_phase41_rejects_stable_black_screen(tmp_path: Path):
    kinds = {timestamp: "black" for timestamp in range(8, 17)}
    ctx = build_phase4_context(tmp_path, frame_kinds=kinds)
    result = run_windows(ctx)
    window = result.windows[0]
    assert not window.is_valid
    assert StabilityRejectionReason.TOO_MANY_BLACK_FRAMES in window.rejection_reasons


def test_phase41_longer_stable_window_gets_larger_duration_support(tmp_path: Path):
    ctx = build_phase4_context(
        tmp_path,
        segments=[
            (TimelineState.CHANGING, 0, 3),
            (TimelineState.STABLE, 3, 6),
            (TimelineState.CHANGING, 6, 8),
            (TimelineState.STABLE, 8, 16),
            (TimelineState.CHANGING, 16, 20),
        ],
    )
    result = run_windows(ctx)
    valid = [item for item in result.windows if item.is_valid]
    assert len(valid) == 2
    assert valid[1].duration_seconds > valid[0].duration_seconds
    assert valid[1].stability_score >= valid[0].stability_score


def test_phase42_detects_changing_to_stable_boundary(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    item = boundaries.boundaries[0]
    assert item.boundary_type is BoundaryType.CHANGING_TO_STABLE
    assert item.timestamp_seconds == 8.0
    assert item.is_valid
    assert item.activity_drop_score > 0.1


def test_phase42_detects_major_transition_to_stable(tmp_path: Path):
    ctx = build_phase4_context(
        tmp_path,
        segments=[
            (TimelineState.CHANGING, 0, 4),
            (TimelineState.MAJOR_TRANSITION, 4, 5),
            (TimelineState.STABLE, 5, 12),
            (TimelineState.CHANGING, 12, 16),
        ],
        major_event_times=[4.5],
    )
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    assert boundaries.boundaries[0].boundary_type is BoundaryType.MAJOR_TRANSITION_TO_STABLE


def test_phase42_invalid_to_stable_is_untrusted(tmp_path: Path):
    ctx = build_phase4_context(
        tmp_path,
        segments=[
            (TimelineState.INVALID, 0, 4),
            (TimelineState.STABLE, 4, 10),
        ],
    )
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    item = boundaries.boundaries[0]
    assert item.boundary_type is BoundaryType.INVALID_TO_STABLE
    assert not item.is_valid
    assert BoundaryRejectionReason.UNTRUSTED_PREVIOUS_STATE in item.rejection_reasons


def test_phase42_weak_activity_drop_is_rejected(tmp_path: Path):
    scores = {timestamp: (0.06 if timestamp <= 8 else 0.05) for timestamp in range(1, 21)}
    ctx = build_phase4_context(tmp_path, score_overrides=scores)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    item = boundaries.boundaries[0]
    assert not item.is_valid
    assert BoundaryRejectionReason.NO_MEANINGFUL_ACTIVITY_DROP in item.rejection_reasons


def test_phase43_generates_bounded_strategic_candidates(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    valid = [item for item in generated.candidates if item.is_valid]
    assert 1 <= len(valid) <= ctx[0].max_candidates_per_window
    assert {item.candidate_type for item in valid} == {
        CandidateType.SETTLED_START,
        CandidateType.MID_STABLE,
        CandidateType.PRE_EXIT,
    }
    assert all(8 <= item.frame_timestamp_seconds <= 16 for item in valid)


def test_phase43_respects_max_candidates_per_window(tmp_path: Path):
    ctx = build_phase4_context(
        tmp_path,
        settings_overrides={"max_candidates_per_window": 2},
        segments=[
            (TimelineState.CHANGING, 0, 4),
            (TimelineState.STABLE, 4, 20),
            (TimelineState.CHANGING, 20, 24),
        ],
    )
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    assert len([item for item in generated.candidates if item.is_valid]) <= 2


def test_phase43_avoids_black_nearest_frame(tmp_path: Path):
    ctx = build_phase4_context(tmp_path, frame_kinds={9: "black"})
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    settled = next(item for item in generated.candidates if item.candidate_type is CandidateType.SETTLED_START)
    assert settled.is_valid
    assert settled.frame_timestamp_seconds != 9.0
    assert not settled.is_black


def test_phase43_window_only_fallback_for_rejected_boundary(tmp_path: Path):
    scores = {timestamp: (0.06 if timestamp <= 8 else 0.05) for timestamp in range(1, 21)}
    ctx = build_phase4_context(tmp_path, score_overrides=scores)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    assert not boundaries.boundaries[0].is_valid
    generated = run_generated(ctx, windows, boundaries)
    valid = [item for item in generated.candidates if item.is_valid]
    assert valid
    assert all(item.source_type is CandidateSourceType.WINDOW_ONLY for item in valid)


def test_phase43_window_only_fallback_can_be_disabled(tmp_path: Path):
    scores = {timestamp: (0.06 if timestamp <= 8 else 0.05) for timestamp in range(1, 21)}
    ctx = build_phase4_context(
        tmp_path,
        settings_overrides={"allow_window_only_candidates": False},
        score_overrides=scores,
    )
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    assert generated.stats.valid_candidates == 0


def test_phase44_dense_candidate_has_more_content_than_blank_candidate(tmp_path: Path):
    ctx = build_phase4_context(
        tmp_path,
        frame_kinds={9: "blank", 12: "dense", 15: "dense"},
    )
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    scored = run_scored(ctx, windows, boundaries, generated)
    by_id = {item.candidate_id: item for item in scored.candidates}
    valid_generated = [item for item in generated.candidates if item.is_valid]
    settled = next(item for item in valid_generated if item.candidate_type is CandidateType.SETTLED_START)
    middle = next(item for item in valid_generated if item.candidate_type is CandidateType.MID_STABLE)
    assert by_id[middle.candidate_id].content_density_score > by_id[settled.candidate_id].content_density_score


def test_phase44_mid_candidate_is_safer_than_immediate_settle_candidate(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    scored = run_scored(ctx, windows, boundaries, generated)
    by_type = {item.candidate_type: item for item in scored.candidates if item.is_valid}
    assert by_type[CandidateType.MID_STABLE].transition_safety_score > by_type[CandidateType.SETTLED_START].transition_safety_score
    assert by_type[CandidateType.MID_STABLE].local_stability_score > by_type[CandidateType.SETTLED_START].local_stability_score


def test_phase44_later_added_content_gets_accumulation_signal(tmp_path: Path):
    ctx = build_phase4_context(tmp_path, frame_kinds={9: "text", 12: "dense", 15: "dense"})
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    scored = run_scored(ctx, windows, boundaries, generated)
    by_type = {item.candidate_type: item for item in scored.candidates if item.is_valid}
    assert by_type[CandidateType.MID_STABLE].content_accumulation_score >= by_type[CandidateType.SETTLED_START].content_accumulation_score


def test_phase44_black_candidate_is_rejected_if_present(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    candidate = next(item for item in generated.candidates if item.is_valid)
    candidate.is_black = True
    scored = run_scored(ctx, windows, boundaries, generated)
    item = next(value for value in scored.candidates if value.candidate_id == candidate.candidate_id)
    assert not item.is_valid
    assert HeuristicRejectionReason.BLACK_FRAME in item.rejection_reasons


def test_phase44_all_normalized_scores_stay_bounded(tmp_path: Path):
    ctx = build_phase4_context(tmp_path, frame_kinds={12: "dense"})
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    scored = run_scored(ctx, windows, boundaries, generated)
    for item in scored.candidates:
        for value in (
            item.visual_quality_score,
            item.content_density_score,
            item.local_stability_score,
            item.transition_risk_score,
            item.transition_safety_score,
            item.preceding_activity_strength,
            item.content_accumulation_score,
            item.completeness_heuristic_score,
        ):
            assert 0 <= value <= 1


def test_phase45_ranking_selects_primary_and_alternate(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    scored = run_scored(ctx, windows, boundaries, generated)
    settings, workspace, _, _, job, *_ = ctx
    ranked, selections = CandidateRankingService(settings, workspace).process(
        job_id=job.id,
        scored=scored,
        windows=windows,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    selection = selections.windows[0]
    assert selection.primary_candidate_id is not None
    assert len(selection.alternate_candidate_ids) <= settings.top_candidates_per_window - 1
    primary = next(item for item in ranked.candidates if item.selection_role.value == "PRIMARY")
    assert primary.rank_within_window == 1


def test_phase45_close_scores_are_marked_ambiguous(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    scored = run_scored(ctx, windows, boundaries, generated)
    # Equalize the two best candidates to force the deterministic ambiguity path.
    valid = [item for item in scored.candidates if item.is_valid]
    for item in valid[:2]:
        item.visual_quality_score = 0.8
        item.completeness_heuristic_score = 0.8
        item.transition_safety_score = 0.8
        item.local_stability_score = 0.8
        item.boundary_score = 0.8
        item.content_density_score = 0.5
        item.content_accumulation_score = 0.2
    settings, workspace, _, _, job, *_ = ctx
    _, selections = CandidateRankingService(settings, workspace).process(
        job_id=job.id,
        scored=scored,
        windows=windows,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    assert selections.windows[0].is_ambiguous


def test_phase45_ranking_is_deterministic(tmp_path: Path):
    ctx = build_phase4_context(tmp_path)
    windows = run_windows(ctx)
    boundaries = run_boundaries(ctx, windows)
    generated = run_generated(ctx, windows, boundaries)
    scored = run_scored(ctx, windows, boundaries, generated)
    settings, workspace, _, _, job, *_ = ctx
    service = CandidateRankingService(settings, workspace)
    first, first_sel = service.process(job_id=job.id, scored=scored, windows=windows, progress_callback=lambda _: None, cancel_check=lambda: False)
    second, second_sel = service.process(job_id=job.id, scored=scored, windows=windows, progress_callback=lambda _: None, cancel_check=lambda: False)
    assert [(x.candidate_id, x.rank_within_window, x.selection_role) for x in first.candidates] == [(x.candidate_id, x.rank_within_window, x.selection_role) for x in second.candidates]
    assert first_sel.windows == second_sel.windows


def test_phase4_weight_groups_reject_all_zero(tmp_path: Path):
    with pytest.raises(ValueError):
        build_phase4_context(
            tmp_path,
            settings_overrides={
                "rank_quality_weight": 0,
                "rank_completeness_weight": 0,
                "rank_safety_weight": 0,
                "rank_local_stability_weight": 0,
                "rank_boundary_weight": 0,
                "rank_density_weight": 0,
                "rank_accumulation_weight": 0,
            },
        )

from pathlib import Path

import pytest

from app.video_analysis.changes import MajorChangeDetector, derive_major_thresholds
from app.video_analysis.models import (
    DifferenceManifest,
    DifferenceRecord,
    DifferenceStats,
    MajorEventType,
)
from tests.phase3_helpers import build_phase3_workspace


def make_differences(scores, *, black_indices=None, invalid_indices=None, metric_mode="balanced"):
    black_indices = set(black_indices or [])
    invalid_indices = set(invalid_indices or [])
    records = []
    valid_scores = []
    for i, score in enumerate(scores, start=1):
        valid = i not in invalid_indices
        if valid:
            valid_scores.append(score)
        if metric_mode == "pixel_only":
            metrics = dict(pixel_difference=score, ssim_difference=0.02, phash_difference=0.02, edge_difference=0.02)
        else:
            metrics = dict(pixel_difference=score, ssim_difference=score, phash_difference=score, edge_difference=score)
        records.append(
            DifferenceRecord(
                previous_frame_index=i,
                current_frame_index=i + 1,
                previous_timestamp_seconds=float(i - 1),
                current_timestamp_seconds=float(i),
                delta_seconds=1.0,
                difference_score=score if valid else None,
                ssim_similarity=(1 - score) if valid else None,
                valid=valid,
                reason=None if valid else "invalid",
                previous_is_black=(i - 1) in black_indices,
                current_is_black=i in black_indices,
                **({k: v for k, v in metrics.items()} if valid else {}),
            )
        )
    ordered = sorted(valid_scores)
    def percentile(p):
        if not ordered:
            return 0.0
        import numpy as np
        return float(np.percentile(ordered, p))
    stats = DifferenceStats(
        total_comparisons=len(records),
        valid_comparisons=len(valid_scores),
        invalid_comparisons=len(records)-len(valid_scores),
        mean_difference_score=sum(valid_scores)/len(valid_scores) if valid_scores else 0,
        median_difference_score=percentile(50),
        max_difference_score=max(valid_scores, default=0),
        p50=percentile(50), p75=percentile(75), p90=percentile(90), p95=percentile(95), p99=percentile(99),
    )
    return DifferenceManifest(
        algorithm_version="1",
        preprocessing_fingerprint="pre",
        config_fingerprint="cfg",
        artifact_fingerprint="diff-fp",
        stats=stats,
        comparisons=records,
    )


def test_static_scores_create_no_major_events(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path)
    diffs = make_differences([0.01, 0.02, 0.01, 0.03, 0.02] * 10)
    result = MajorChangeDetector(settings, workspace).process(
        job_id=job_id, differences=diffs, progress_callback=lambda _: None, cancel_check=lambda: False
    )
    assert result.stats.major_events == 0
    assert result.stats.very_major_events == 0


def test_single_outlier_creates_major_event(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path)
    diffs = make_differences([0.03] * 30 + [0.82] + [0.04] * 20)
    result = MajorChangeDetector(settings, workspace).process(
        job_id=job_id, differences=diffs, progress_callback=lambda _: None, cancel_check=lambda: False
    )
    visual = [e for e in result.events if e.event_type in {MajorEventType.MAJOR_VISUAL_CHANGE, MajorEventType.VERY_MAJOR_VISUAL_CHANGE}]
    assert len(visual) == 1
    assert visual[0].difference_score == pytest.approx(0.82)


def test_clustered_high_scores_merge_and_choose_strongest(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, major_change_merge_window_seconds=2)
    diffs = make_differences([0.03] * 20 + [0.72, 0.88, 0.69] + [0.03] * 20)
    result = MajorChangeDetector(settings, workspace).process(
        job_id=job_id, differences=diffs, progress_callback=lambda _: None, cancel_check=lambda: False
    )
    visual = [e for e in result.events if e.event_type in {MajorEventType.MAJOR_VISUAL_CHANGE, MajorEventType.VERY_MAJOR_VISUAL_CHANGE}]
    assert len(visual) == 1
    assert visual[0].difference_score == pytest.approx(0.88)


def test_distant_major_events_remain_separate(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, major_change_merge_window_seconds=2)
    scores = [0.03] * 50
    scores[10] = 0.9
    scores[40] = 0.95
    result = MajorChangeDetector(settings, workspace).process(
        job_id=job_id, differences=make_differences(scores), progress_callback=lambda _: None, cancel_check=lambda: False
    )
    visual = [e for e in result.events if e.event_type in {MajorEventType.MAJOR_VISUAL_CHANGE, MajorEventType.VERY_MAJOR_VISUAL_CHANGE}]
    assert len(visual) == 2


def test_black_fade_sequence_collapses_to_one_transition(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path)
    diffs = make_differences([0.1, 0.7, 0.8, 0.6, 0.1], black_indices={2, 3})
    result = MajorChangeDetector(settings, workspace).process(
        job_id=job_id, differences=diffs, progress_callback=lambda _: None, cancel_check=lambda: False
    )
    black = [e for e in result.events if e.event_type is MajorEventType.BLACK_TRANSITION]
    assert len(black) == 1


def test_invalid_gap_is_preserved(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path)
    result = MajorChangeDetector(settings, workspace).process(
        job_id=job_id,
        differences=make_differences([0.02, 0.02, 0.02], invalid_indices={2}),
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    assert any(e.event_type is MajorEventType.INVALID_GAP for e in result.events)


def test_adaptive_threshold_changes_for_noisier_distribution(tmp_path: Path):
    settings, _, _ = build_phase3_workspace(tmp_path)
    quiet = derive_major_thresholds(make_differences([0.01, 0.02, 0.03] * 20), settings)
    noisy = derive_major_thresholds(make_differences([0.15, 0.22, 0.31, 0.42, 0.48] * 20), settings)
    assert noisy.major_threshold > quiet.major_threshold


def test_absolute_floor_prevents_tiny_outlier_from_becoming_major(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, major_change_min_score=0.3)
    result = MajorChangeDetector(settings, workspace).process(
        job_id=job_id,
        differences=make_differences([0.001, 0.002, 0.003, 0.01] * 10),
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    assert result.stats.major_events == 0


def test_metric_agreement_increases_confidence(tmp_path: Path):
    settings_a, workspace_a, job_a = build_phase3_workspace(tmp_path / "a", major_change_min_score=0.3, very_major_change_min_score=0.95)
    pixel_result = MajorChangeDetector(settings_a, workspace_a).process(
        job_id=job_a,
        differences=make_differences([0.8], metric_mode="pixel_only"),
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    settings_b, workspace_b, job_b = build_phase3_workspace(tmp_path / "b", major_change_min_score=0.3, very_major_change_min_score=0.95)
    structural_result = MajorChangeDetector(settings_b, workspace_b).process(
        job_id=job_b,
        differences=make_differences([0.8], metric_mode="balanced"),
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    assert structural_result.events[0].confidence > pixel_result.events[0].confidence

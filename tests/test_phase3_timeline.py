from pathlib import Path

import numpy as np

from app.video_analysis.models import (
    DifferenceManifest,
    DifferenceRecord,
    DifferenceStats,
    MajorChangeEvent,
    MajorChangeStats,
    MajorChangeThresholds,
    MajorChangesManifest,
    MajorEventType,
    TimelineState,
)
from app.video_analysis.timeline import TemporalTimelineService, derive_timeline_thresholds
from tests.phase3_helpers import build_phase3_workspace


def make_diffs(scores, *, delta=1.0, invalid=None):
    invalid = set(invalid or [])
    records = []
    valid_scores = []
    for i, score in enumerate(scores, start=1):
        ok = i not in invalid
        if ok:
            valid_scores.append(score)
        records.append(
            DifferenceRecord(
                previous_frame_index=i,
                current_frame_index=i + 1,
                previous_timestamp_seconds=(i - 1) * delta,
                current_timestamp_seconds=i * delta,
                delta_seconds=delta,
                pixel_difference=score if ok else None,
                ssim_similarity=1-score if ok else None,
                ssim_difference=score if ok else None,
                phash_difference=score if ok else None,
                edge_difference=score if ok else None,
                difference_score=score if ok else None,
                valid=ok,
                reason=None if ok else "invalid",
            )
        )
    arr = np.asarray(valid_scores or [0.0])
    stats = DifferenceStats(
        total_comparisons=len(records),
        valid_comparisons=len(valid_scores),
        invalid_comparisons=len(records)-len(valid_scores),
        mean_difference_score=float(np.mean(arr)),
        median_difference_score=float(np.median(arr)),
        max_difference_score=float(np.max(arr)),
        p50=float(np.percentile(arr,50)), p75=float(np.percentile(arr,75)),
        p90=float(np.percentile(arr,90)), p95=float(np.percentile(arr,95)), p99=float(np.percentile(arr,99)),
    )
    return DifferenceManifest(
        algorithm_version="1", preprocessing_fingerprint="pre", config_fingerprint="cfg",
        artifact_fingerprint="diff", stats=stats, comparisons=records
    )


def make_major(events=None, major_threshold=0.5):
    events = events or []
    return MajorChangesManifest(
        algorithm_version="1", differences_fingerprint="diff", config_fingerprint="cfg",
        artifact_fingerprint="major",
        thresholds=MajorChangeThresholds(
            median_score=0.03, mad=0.01, p90=0.1, p95=0.2, p99=0.4,
            major_threshold=major_threshold, very_major_threshold=max(0.7, major_threshold),
        ),
        stats=MajorChangeStats(
            total_comparisons=10, major_events=sum(e.event_type is MajorEventType.MAJOR_VISUAL_CHANGE for e in events),
            very_major_events=sum(e.event_type is MajorEventType.VERY_MAJOR_VISUAL_CHANGE for e in events),
            black_transitions=sum(e.event_type is MajorEventType.BLACK_TRANSITION for e in events),
            invalid_gaps=sum(e.event_type is MajorEventType.INVALID_GAP for e in events),
        ),
        events=events,
    )


def run_timeline(tmp_path: Path, scores, *, settings_overrides=None, events=None, delta=1.0, invalid=None):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, **(settings_overrides or {}))
    result = TemporalTimelineService(settings, workspace).process(
        job_id=job_id,
        differences=make_diffs(scores, delta=delta, invalid=invalid),
        major_changes=make_major(events),
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    return result


def test_sustained_low_scores_form_stable_segment(tmp_path: Path):
    result = run_timeline(tmp_path, [0.01] * 10)
    assert len(result.segments) == 1
    assert result.segments[0].state is TimelineState.STABLE
    assert result.segments[0].duration_seconds == 10


def test_sustained_moderate_scores_form_changing_segment(tmp_path: Path):
    result = run_timeline(tmp_path, [0.15] * 10)
    assert len(result.segments) == 1
    assert result.segments[0].state is TimelineState.CHANGING


def test_flicker_is_smoothed_between_stable_runs(tmp_path: Path):
    result = run_timeline(
        tmp_path,
        [0.01, 0.01, 0.12, 0.01, 0.01],
        settings_overrides={"max_stable_gap_seconds": 1.1},
    )
    assert len(result.segments) == 1
    assert result.segments[0].state is TimelineState.STABLE


def test_strong_short_change_is_not_smoothed_away(tmp_path: Path):
    result = run_timeline(
        tmp_path,
        [0.01, 0.01, 0.45, 0.01, 0.01],
        settings_overrides={"max_stable_gap_seconds": 2},
    )
    assert any(segment.state is TimelineState.CHANGING for segment in result.segments)


def test_minimum_stable_duration_is_enforced(tmp_path: Path):
    result = run_timeline(tmp_path, [0.01, 0.2, 0.2])
    assert result.segments[0].state is TimelineState.CHANGING


def test_major_transition_splits_stable_regions(tmp_path: Path):
    event = MajorChangeEvent(
        event_id=1, event_type=MajorEventType.MAJOR_VISUAL_CHANGE,
        timestamp_seconds=3, start_timestamp_seconds=2, end_timestamp_seconds=3,
        previous_frame_index=3, current_frame_index=4, difference_score=0.8,
        threshold=0.5, confidence=0.9,
    )
    result = run_timeline(tmp_path, [0.01] * 5, events=[event])
    assert [segment.state for segment in result.segments] == [
        TimelineState.STABLE, TimelineState.MAJOR_TRANSITION, TimelineState.STABLE
    ]


def test_black_transition_remains_boundary(tmp_path: Path):
    event = MajorChangeEvent(
        event_id=1, event_type=MajorEventType.BLACK_TRANSITION,
        timestamp_seconds=3, start_timestamp_seconds=2, end_timestamp_seconds=3,
        previous_frame_index=3, current_frame_index=4, confidence=1,
    )
    result = run_timeline(tmp_path, [0.01] * 5, events=[event])
    assert any(segment.state is TimelineState.BLACK_TRANSITION for segment in result.segments)


def test_invalid_comparison_becomes_invalid_segment(tmp_path: Path):
    result = run_timeline(tmp_path, [0.01] * 5, invalid={3})
    assert any(segment.state is TimelineState.INVALID for segment in result.segments)


def test_duration_uses_timestamps_not_frame_count(tmp_path: Path):
    result = run_timeline(tmp_path, [0.01] * 4, delta=0.5)
    assert result.stats.total_duration_seconds == 2.0


def test_timeline_statistics_match_segments(tmp_path: Path):
    result = run_timeline(tmp_path, [0.01] * 4 + [0.2] * 4)
    assert result.stats.stable_segment_count >= 1
    assert result.stats.changing_segment_count >= 1
    assert 0 <= result.stats.stable_ratio <= 1
    assert 0 <= result.stats.changing_ratio <= 1
    assert result.stats.total_duration_seconds == sum(s.duration_seconds for s in result.segments)


def test_stable_threshold_stays_below_major_threshold(tmp_path: Path):
    settings, _, _ = build_phase3_workspace(tmp_path)
    thresholds = derive_timeline_thresholds(make_diffs([0.01,0.02,0.1,0.15]), make_major(), settings)
    assert thresholds.enter_stable_threshold < thresholds.major_threshold
    assert thresholds.exit_stable_threshold < thresholds.major_threshold
    assert thresholds.enter_stable_threshold <= thresholds.exit_stable_threshold

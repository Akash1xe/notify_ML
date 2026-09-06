from app.video_analysis.evaluation import evaluate_phase3
from app.video_analysis.models import (
    DifferenceManifest,
    DifferenceStats,
    FrameAnalysisSummary,
    MajorChangeStats,
    MajorChangeThresholds,
    MajorChangesManifest,
    Phase3DiskUsage,
    Phase3Timings,
    PreprocessingManifest,
    PreprocessingStats,
    TimelineManifest,
    TimelineSegment,
    TimelineState,
    TimelineStats,
    TimelineThresholds,
)


def make_inputs(*, duration=600, invalid=0, black=0, major_events=5, stable_ratio=0.5, changing_ratio=0.45, segments=10):
    summary = FrameAnalysisSummary(
        sampled_frame_count=600,
        processed_frame_count=600-invalid,
        invalid_frame_count=invalid,
        difference_count=599,
        major_event_count=major_events,
        stable_segment_count=segments//2,
        changing_segment_count=segments//2,
        analysis_start_seconds=0,
        analysis_end_seconds=duration,
        timings=Phase3Timings(),
        disk_usage=Phase3DiskUsage(),
        sampling_fingerprint="s", preprocessing_fingerprint="p", differences_fingerprint="d",
        major_changes_fingerprint="m", timeline_fingerprint="t",
    )
    preprocessing = PreprocessingManifest(
        algorithm_version="1", sampling_fingerprint="s", config_fingerprint="c", artifact_fingerprint="p",
        analysis_width=640,
        stats=PreprocessingStats(
            total_frames=600, valid_frames=600-invalid, corrupt_frames=invalid,
            black_frames=black, blurry_frames=10, low_information_frames=20,
        ),
        frames=[],
    )
    differences = DifferenceManifest(
        algorithm_version="1", preprocessing_fingerprint="p", config_fingerprint="c", artifact_fingerprint="d",
        stats=DifferenceStats(
            total_comparisons=599, valid_comparisons=599, invalid_comparisons=0,
            mean_difference_score=.1, median_difference_score=.05, max_difference_score=.9,
            p50=.05,p75=.1,p90=.2,p95=.4,p99=.8,
        ),
        comparisons=[],
    )
    major = MajorChangesManifest(
        algorithm_version="1", differences_fingerprint="d", config_fingerprint="c", artifact_fingerprint="m",
        thresholds=MajorChangeThresholds(
            median_score=.05,mad=.02,p90=.2,p95=.4,p99=.8,major_threshold=.4,very_major_threshold=.8
        ),
        stats=MajorChangeStats(
            total_comparisons=599, major_events=major_events, very_major_events=0,
            black_transitions=0, invalid_gaps=0,
        ),
        events=[],
    )
    stable_duration = duration*stable_ratio
    changing_duration = duration*changing_ratio
    segment_list = []
    if stable_duration:
        segment_list.append(TimelineSegment(
            segment_id=1,state=TimelineState.STABLE,start_timestamp_seconds=0,end_timestamp_seconds=stable_duration,
            duration_seconds=stable_duration,comparison_count=1
        ))
    if changing_duration:
        segment_list.append(TimelineSegment(
            segment_id=2,state=TimelineState.CHANGING,start_timestamp_seconds=stable_duration,
            end_timestamp_seconds=stable_duration+changing_duration,duration_seconds=changing_duration,comparison_count=1
        ))
    # Inflate fragmentation count without changing the timing semantics used by the warning test.
    while len(segment_list) < segments:
        last = segment_list[-1]
        segment_list.append(TimelineSegment(
            segment_id=len(segment_list)+1,state=TimelineState.INVALID,
            start_timestamp_seconds=last.end_timestamp_seconds,end_timestamp_seconds=last.end_timestamp_seconds,
            duration_seconds=0,comparison_count=0
        ))
    timeline = TimelineManifest(
        algorithm_version="1",differences_fingerprint="d",major_changes_fingerprint="m",config_fingerprint="c",artifact_fingerprint="t",
        thresholds=TimelineThresholds(enter_stable_threshold=.05,exit_stable_threshold=.08,major_threshold=.4),
        stats=TimelineStats(
            total_duration_seconds=duration,stable_duration_seconds=stable_duration,changing_duration_seconds=changing_duration,
            stable_segment_count=1 if stable_duration else 0,changing_segment_count=1 if changing_duration else 0,
            stable_ratio=stable_ratio,changing_ratio=changing_ratio,
        ),segments=segment_list,
    )
    return summary, preprocessing, differences, major, timeline


def test_evaluation_calculates_core_metrics():
    report = evaluate_phase3(
        summary=make_inputs()[0], preprocessing=make_inputs()[1], differences=make_inputs()[2],
        major_changes=make_inputs()[3], timeline=make_inputs()[4]
    )
    assert report.lecture_duration_seconds == 600
    assert report.events_per_minute == 0.5
    assert report.score_distribution["p95"] == 0.4
    assert report.stable_ratio == 0.5


def test_evaluation_warns_about_invalid_frames():
    args = make_inputs(invalid=60)
    report = evaluate_phase3(summary=args[0], preprocessing=args[1], differences=args[2], major_changes=args[3], timeline=args[4])
    assert "HIGH_INVALID_FRAME_RATIO" in {warning.code for warning in report.warnings}


def test_evaluation_warns_about_mostly_changing_timeline():
    args = make_inputs(stable_ratio=.05, changing_ratio=.94)
    report = evaluate_phase3(summary=args[0], preprocessing=args[1], differences=args[2], major_changes=args[3], timeline=args[4])
    assert "MOSTLY_CHANGING" in {warning.code for warning in report.warnings}


def test_evaluation_warns_about_high_event_density():
    args = make_inputs(major_events=50)
    report = evaluate_phase3(summary=args[0], preprocessing=args[1], differences=args[2], major_changes=args[3], timeline=args[4])
    assert "HIGH_MAJOR_EVENT_DENSITY" in {warning.code for warning in report.warnings}


def test_evaluation_warnings_are_diagnostic_not_failures():
    args = make_inputs(invalid=100, black=100, major_events=100, stable_ratio=.01, changing_ratio=.98, segments=100)
    report = evaluate_phase3(summary=args[0], preprocessing=args[1], differences=args[2], major_changes=args[3], timeline=args[4])
    assert len(report.warnings) >= 3

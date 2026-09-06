from __future__ import annotations

import statistics

from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.models import (
    DifferenceManifest,
    EvaluationWarning,
    FrameAnalysisSummary,
    MajorChangesManifest,
    Phase3EvaluationReport,
    PreprocessingManifest,
    TimelineManifest,
    TimelineState,
)


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def evaluate_phase3(
    *,
    summary: FrameAnalysisSummary,
    preprocessing: PreprocessingManifest,
    differences: DifferenceManifest,
    major_changes: MajorChangesManifest,
    timeline: TimelineManifest,
) -> Phase3EvaluationReport:
    duration = max(0.0, summary.analysis_end_seconds - summary.analysis_start_seconds)
    minutes = duration / 60.0 if duration > 0 else 0.0
    stable_durations = [
        segment.duration_seconds
        for segment in timeline.segments
        if segment.state is TimelineState.STABLE
    ]
    changing_durations = [
        segment.duration_seconds
        for segment in timeline.segments
        if segment.state is TimelineState.CHANGING
    ]
    total_frames = max(1, preprocessing.stats.total_frames)
    invalid_ratio = preprocessing.stats.corrupt_frames / total_frames
    black_ratio = preprocessing.stats.black_frames / total_frames
    events_per_minute = (
        (major_changes.stats.major_events + major_changes.stats.very_major_events) / minutes
        if minutes > 0
        else 0.0
    )
    segments_per_minute = len(timeline.segments) / minutes if minutes > 0 else 0.0
    warnings: list[EvaluationWarning] = []
    if invalid_ratio > 0.05:
        warnings.append(EvaluationWarning(code="HIGH_INVALID_FRAME_RATIO", message="More than 5% of sampled frames are invalid."))
    if black_ratio > 0.10:
        warnings.append(EvaluationWarning(code="HIGH_BLACK_FRAME_RATIO", message="More than 10% of sampled frames are black transition-like frames."))
    if events_per_minute > 3.0:
        warnings.append(EvaluationWarning(code="HIGH_MAJOR_EVENT_DENSITY", message="Major visual events are unusually dense; thresholds may be too sensitive."))
    if minutes >= 10 and major_changes.stats.major_events + major_changes.stats.very_major_events == 0:
        warnings.append(EvaluationWarning(code="NO_MAJOR_EVENTS", message="No major visual events were detected in a long lecture."))
    if timeline.stats.changing_ratio > 0.90:
        warnings.append(EvaluationWarning(code="MOSTLY_CHANGING", message="More than 90% of analyzed time is classified as changing."))
    if timeline.stats.stable_ratio > 0.95 and duration > 60:
        warnings.append(EvaluationWarning(code="MOSTLY_STABLE", message="More than 95% of analyzed time is classified as stable."))
    if segments_per_minute > 5.0:
        warnings.append(EvaluationWarning(code="HIGH_FRAGMENTATION", message="The activity timeline contains many short segments."))

    return Phase3EvaluationReport(
        lecture_duration_seconds=duration,
        sampled_frames=summary.sampled_frame_count,
        valid_frames=preprocessing.stats.valid_frames,
        invalid_frames=preprocessing.stats.corrupt_frames,
        black_frames=preprocessing.stats.black_frames,
        blurry_frames=preprocessing.stats.blurry_frames,
        low_information_frames=preprocessing.stats.low_information_frames,
        score_distribution={
            "p50": differences.stats.p50,
            "p75": differences.stats.p75,
            "p90": differences.stats.p90,
            "p95": differences.stats.p95,
            "p99": differences.stats.p99,
            "max": differences.stats.max_difference_score,
        },
        major_events=major_changes.stats.major_events + major_changes.stats.very_major_events,
        events_per_minute=events_per_minute,
        stable_ratio=timeline.stats.stable_ratio,
        changing_ratio=timeline.stats.changing_ratio,
        segments_per_minute=segments_per_minute,
        mean_stable_segment_seconds=_mean(stable_durations),
        median_stable_segment_seconds=_median(stable_durations),
        mean_changing_segment_seconds=_mean(changing_durations),
        median_changing_segment_seconds=_median(changing_durations),
        example_event_timestamps=[event.timestamp_seconds for event in major_changes.events[:8]],
        warnings=warnings,
    )


def persist_evaluation(
    workspace: WorkspaceManager,
    job_id: str,
    report: Phase3EvaluationReport,
) -> None:
    atomic_write_json(workspace.evaluation_path(job_id), report.model_dump(mode="json"))

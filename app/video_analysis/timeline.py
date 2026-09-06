from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, TimelineGenerationError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import (
    DifferenceManifest,
    DifferenceRecord,
    MajorChangesManifest,
    MajorEventType,
    TimelineManifest,
    TimelineSegment,
    TimelineState,
    TimelineStats,
    TimelineThresholds,
)


TIMELINE_ALGORITHM_VERSION = "1"
CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[float], None]


def timeline_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": TIMELINE_ALGORITHM_VERSION,
        "stable_min_score": settings.timeline_stable_min_score,
        "stable_max_score": settings.timeline_stable_max_score,
        "stable_robust_multiplier": settings.timeline_stable_robust_multiplier,
        "exit_stable_factor": settings.timeline_exit_stable_factor,
        "min_stable_duration_seconds": settings.min_stable_duration_seconds,
        "min_changing_duration_seconds": settings.min_changing_duration_seconds,
        "max_stable_gap_seconds": settings.max_stable_gap_seconds,
    }


def timeline_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(timeline_config_payload(settings))


def timeline_artifact_fingerprint(manifest: TimelineManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "differences_fingerprint": manifest.differences_fingerprint,
            "major_changes_fingerprint": manifest.major_changes_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "thresholds": manifest.thresholds.model_dump(mode="json"),
            "stats": manifest.stats.model_dump(mode="json"),
            "segments": [segment.model_dump(mode="json") for segment in manifest.segments],
        }
    )


def derive_timeline_thresholds(
    differences: DifferenceManifest,
    major_changes: MajorChangesManifest,
    settings: AppSettings,
) -> TimelineThresholds:
    scores = [
        item.difference_score
        for item in differences.comparisons
        if item.valid and item.difference_score is not None
    ]
    major = major_changes.thresholds.major_threshold
    if not scores:
        enter = min(major * 0.5, settings.timeline_stable_min_score)
        exit_threshold = min(major * 0.85, max(enter, enter * settings.timeline_exit_stable_factor))
        return TimelineThresholds(
            enter_stable_threshold=max(0.0, enter),
            exit_stable_threshold=max(0.0, exit_threshold),
            major_threshold=major,
        )
    arr = np.asarray(scores, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust = median + settings.timeline_stable_robust_multiplier * 1.4826 * mad
    enter = max(settings.timeline_stable_min_score, robust)
    enter = min(enter, settings.timeline_stable_max_score, major * 0.35)
    exit_threshold = max(enter, enter * settings.timeline_exit_stable_factor)
    exit_threshold = min(exit_threshold, major * 0.65)
    return TimelineThresholds(
        enter_stable_threshold=float(max(0.0, min(1.0, enter))),
        exit_stable_threshold=float(max(0.0, min(1.0, exit_threshold))),
        major_threshold=major,
    )


@dataclass
class _Segment:
    state: TimelineState
    start: float
    end: float
    records: list[DifferenceRecord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def scores(self) -> list[float]:
        return [
            record.difference_score
            for record in self.records
            if record.valid and record.difference_score is not None
        ]

    @property
    def max_score(self) -> float:
        return max(self.scores, default=0.0)


def _event_state(record: DifferenceRecord, major_changes: MajorChangesManifest) -> TimelineState | None:
    start = record.previous_timestamp_seconds
    end = record.current_timestamp_seconds
    overlapping = [
        event
        for event in major_changes.events
        if record.current_timestamp_seconds > event.start_timestamp_seconds + 1e-6
        and record.current_timestamp_seconds <= event.end_timestamp_seconds + 1e-6
    ]
    if any(event.event_type is MajorEventType.INVALID_GAP for event in overlapping):
        return TimelineState.INVALID
    if any(event.event_type is MajorEventType.BLACK_TRANSITION for event in overlapping):
        return TimelineState.BLACK_TRANSITION
    if any(
        event.event_type
        in {MajorEventType.MAJOR_VISUAL_CHANGE, MajorEventType.VERY_MAJOR_VISUAL_CHANGE}
        for event in overlapping
    ):
        return TimelineState.MAJOR_TRANSITION
    return None


def _append_segment(segments: list[_Segment], state: TimelineState, record: DifferenceRecord) -> None:
    start = record.previous_timestamp_seconds
    end = record.current_timestamp_seconds
    if segments and segments[-1].state is state and start <= segments[-1].end + 1e-6:
        segments[-1].end = max(segments[-1].end, end)
        segments[-1].records.append(record)
    else:
        if segments and start > segments[-1].end + 1e-6:
            segments.append(_Segment(TimelineState.INVALID, segments[-1].end, start, []))
        segments.append(_Segment(state, start, end, [record]))


def _merge_adjacent(segments: list[_Segment]) -> list[_Segment]:
    merged: list[_Segment] = []
    for segment in segments:
        if merged and merged[-1].state is segment.state and segment.start <= merged[-1].end + 1e-6:
            merged[-1].end = max(merged[-1].end, segment.end)
            merged[-1].records.extend(segment.records)
        else:
            merged.append(segment)
    return merged


def _smooth_segments(segments: list[_Segment], settings: AppSettings, major_threshold: float) -> list[_Segment]:
    if not segments:
        return segments
    # A short low-change run is not enough to establish stability.
    for segment in segments:
        if (
            segment.state is TimelineState.STABLE
            and segment.duration < settings.min_stable_duration_seconds
        ):
            segment.state = TimelineState.CHANGING
    segments = _merge_adjacent(segments)

    # Bridge only weak, short changing gaps between stable regions. Strong short
    # changes remain explicit boundaries.
    for index in range(1, len(segments) - 1):
        segment = segments[index]
        if segment.state is not TimelineState.CHANGING:
            continue
        left, right = segments[index - 1], segments[index + 1]
        if left.state is not TimelineState.STABLE or right.state is not TimelineState.STABLE:
            continue
        weak = segment.max_score < major_threshold * 0.80
        short_gap = segment.duration <= settings.max_stable_gap_seconds
        below_min_run = segment.duration < settings.min_changing_duration_seconds
        if weak and (short_gap or below_min_run):
            segment.state = TimelineState.STABLE
    return _merge_adjacent(segments)


def _to_public_segments(segments: list[_Segment], major_threshold: float) -> list[TimelineSegment]:
    result: list[TimelineSegment] = []
    for index, segment in enumerate(segments, start=1):
        scores = segment.scores
        mean_score = float(statistics.mean(scores)) if scores else None
        median_score = float(statistics.median(scores)) if scores else None
        min_score = min(scores) if scores else None
        max_score = max(scores) if scores else None
        start_frame = segment.records[0].previous_frame_index if segment.records else None
        end_frame = segment.records[-1].current_frame_index if segment.records else None
        intensity = None
        if scores and segment.state is TimelineState.CHANGING:
            intensity = float(min(1.0, (mean_score or 0.0) / max(1e-6, major_threshold)))
        result.append(
            TimelineSegment(
                segment_id=index,
                state=segment.state,
                start_timestamp_seconds=segment.start,
                end_timestamp_seconds=segment.end,
                duration_seconds=max(0.0, segment.end - segment.start),
                start_frame_index=start_frame,
                end_frame_index=end_frame,
                mean_difference_score=mean_score,
                median_difference_score=median_score,
                min_difference_score=min_score,
                max_difference_score=max_score,
                change_intensity=intensity,
                comparison_count=len(segment.records),
            )
        )
    return result


def _timeline_stats(segments: list[TimelineSegment]) -> TimelineStats:
    total = sum(segment.duration_seconds for segment in segments)
    stable = sum(segment.duration_seconds for segment in segments if segment.state is TimelineState.STABLE)
    changing = sum(segment.duration_seconds for segment in segments if segment.state is TimelineState.CHANGING)
    transitions = sum(
        segment.duration_seconds
        for segment in segments
        if segment.state in {TimelineState.MAJOR_TRANSITION, TimelineState.BLACK_TRANSITION}
    )
    invalid = sum(segment.duration_seconds for segment in segments if segment.state is TimelineState.INVALID)
    return TimelineStats(
        total_duration_seconds=total,
        stable_duration_seconds=stable,
        changing_duration_seconds=changing,
        transition_duration_seconds=transitions,
        invalid_duration_seconds=invalid,
        stable_segment_count=sum(segment.state is TimelineState.STABLE for segment in segments),
        changing_segment_count=sum(segment.state is TimelineState.CHANGING for segment in segments),
        transition_segment_count=sum(
            segment.state in {TimelineState.MAJOR_TRANSITION, TimelineState.BLACK_TRANSITION}
            for segment in segments
        ),
        invalid_segment_count=sum(segment.state is TimelineState.INVALID for segment in segments),
        stable_ratio=stable / total if total > 0 else 0.0,
        changing_ratio=changing / total if total > 0 else 0.0,
        invalid_ratio=invalid / total if total > 0 else 0.0,
    )


class TemporalTimelineService:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def process(
        self,
        *,
        job_id: str,
        differences: DifferenceManifest,
        major_changes: MajorChangesManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> TimelineManifest:
        started = time.monotonic()
        thresholds = derive_timeline_thresholds(differences, major_changes, self._settings)
        comparisons = differences.comparisons
        positive_deltas = [record.delta_seconds for record in comparisons if record.delta_seconds > 0]
        expected_delta = float(statistics.median(positive_deltas)) if positive_deltas else 0.0
        raw: list[_Segment] = []
        stable_mode = False
        total = len(comparisons)

        for index, record in enumerate(comparisons, start=1):
            if cancel_check():
                raise JobCancelledError("Temporal timeline generation was cancelled.")
            special = _event_state(record, major_changes)
            if not record.valid:
                state = TimelineState.INVALID
            elif expected_delta > 0 and record.delta_seconds > expected_delta * 2.5:
                state = TimelineState.INVALID
            elif special is not None:
                state = special
            else:
                score = record.difference_score or 0.0
                if stable_mode:
                    stable_mode = score <= thresholds.exit_stable_threshold
                else:
                    stable_mode = score <= thresholds.enter_stable_threshold
                state = TimelineState.STABLE if stable_mode else TimelineState.CHANGING
            _append_segment(raw, state, record)
            if total:
                progress_callback(min(80.0, index * 80.0 / total))

        smoothed = _smooth_segments(raw, self._settings, thresholds.major_threshold)
        public_segments = _to_public_segments(smoothed, thresholds.major_threshold)
        stats = _timeline_stats(public_segments)
        manifest = TimelineManifest(
            algorithm_version=TIMELINE_ALGORITHM_VERSION,
            differences_fingerprint=differences.artifact_fingerprint,
            major_changes_fingerprint=major_changes.artifact_fingerprint,
            config_fingerprint=timeline_config_fingerprint(self._settings),
            artifact_fingerprint="pending",
            thresholds=thresholds,
            stats=stats,
            segments=public_segments,
            timeline_seconds=round(time.monotonic() - started, 3),
        )
        manifest.artifact_fingerprint = timeline_artifact_fingerprint(manifest)
        atomic_write_json(self._workspace.timeline_path(job_id), manifest.model_dump(mode="json"))
        progress_callback(100.0)
        return manifest

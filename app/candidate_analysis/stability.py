from __future__ import annotations

import bisect
import math
import statistics
import time
from typing import Callable

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, StabilityWindowDetectionError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import (
    DifferenceManifest,
    MajorChangesManifest,
    PreprocessingManifest,
    TimelineManifest,
    TimelineState,
)
from app.candidate_analysis.models import (
    StabilityRejectionReason,
    StabilityWindow,
    StabilityWindowStats,
    StabilityWindowsManifest,
)
from app.candidate_analysis.utils import (
    clamp01,
    duration_saturation,
    mean,
    median,
    normalized_weighted,
    percentile,
    ratio,
)

STABILITY_WINDOW_ALGORITHM_VERSION = "1"
ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def stability_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": STABILITY_WINDOW_ALGORITHM_VERSION,
        "min_duration_seconds": settings.stability_window_min_duration_seconds,
        "max_invalid_frame_ratio": settings.stability_max_invalid_frame_ratio,
        "max_black_frame_ratio": settings.stability_max_black_frame_ratio,
        "duration_saturation_seconds": settings.stability_duration_saturation_seconds,
        "difference_weight": settings.stability_difference_weight,
        "variance_weight": settings.stability_variance_weight,
        "max_spike_weight": settings.stability_max_spike_weight,
        "duration_weight": settings.stability_duration_weight,
    }


def stability_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(stability_config_payload(settings))


def stability_artifact_fingerprint(manifest: StabilityWindowsManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "timeline_fingerprint": manifest.timeline_fingerprint,
            "differences_fingerprint": manifest.differences_fingerprint,
            "preprocessing_fingerprint": manifest.preprocessing_fingerprint,
            "major_changes_fingerprint": manifest.major_changes_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "stats": manifest.stats.model_dump(mode="json"),
            "windows": [window.model_dump(mode="json") for window in manifest.windows],
        }
    )


def _avg_optional(values) -> float:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(numeric)


class StabilityWindowDetector:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def process(
        self,
        *,
        job_id: str,
        timeline: TimelineManifest,
        differences: DifferenceManifest,
        preprocessing: PreprocessingManifest,
        major_changes: MajorChangesManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> StabilityWindowsManifest:
        started = time.monotonic()
        stable_segments = [segment for segment in timeline.segments if segment.state is TimelineState.STABLE]
        frame_times = [frame.timestamp_seconds for frame in preprocessing.frames]
        comparison_times = [record.current_timestamp_seconds for record in differences.comparisons]
        event_times = sorted(event.timestamp_seconds for event in major_changes.events)
        segment_index = {segment.segment_id: idx for idx, segment in enumerate(timeline.segments)}
        windows: list[StabilityWindow] = []

        for ordinal, segment in enumerate(stable_segments, start=1):
            if cancel_check():
                raise JobCancelledError("Stability-window detection was cancelled.")
            if segment.end_timestamp_seconds < segment.start_timestamp_seconds:
                raise StabilityWindowDetectionError("Timeline contains an invalid stable segment interval.")

            comp_left = bisect.bisect_left(comparison_times, segment.start_timestamp_seconds - 1e-9)
            comp_right = bisect.bisect_right(comparison_times, segment.end_timestamp_seconds + 1e-9)
            comparison_records = [
                record
                for record in differences.comparisons[comp_left:comp_right]
                if record.valid and record.difference_score is not None
            ]
            scores = [float(record.difference_score) for record in comparison_records]

            frame_left = bisect.bisect_left(frame_times, segment.start_timestamp_seconds - 1e-9)
            frame_right = bisect.bisect_right(frame_times, segment.end_timestamp_seconds + 1e-9)
            frames = preprocessing.frames[frame_left:frame_right]
            total_frames = len(frames)
            valid_frames = [frame for frame in frames if frame.is_valid]
            invalid_ratio = ratio(total_frames - len(valid_frames), total_frames)
            black_ratio = ratio(sum(frame.is_black for frame in frames), total_frames)
            blurry_ratio = ratio(sum(frame.is_blurry for frame in frames), total_frames)
            low_info_ratio = ratio(sum(frame.is_low_information for frame in frames), total_frames)
            valid_ratio = ratio(len(valid_frames), total_frames)

            avg_quality = _avg_optional(frame.quality_score for frame in valid_frames)
            quality_score = normalized_weighted(
                [
                    (avg_quality, 0.55),
                    (valid_ratio, 0.20),
                    (1.0 - blurry_ratio, 0.10),
                    (1.0 - low_info_ratio, 0.10),
                    (1.0 - black_ratio, 0.05),
                ]
            )
            if black_ratio > self._settings.stability_max_black_frame_ratio:
                quality_score *= 0.25

            mean_score = mean(scores)
            median_score = median(scores)
            max_score = max(scores) if scores else 0.0
            stddev = float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0
            stable_threshold = max(1e-6, timeline.thresholds.enter_stable_threshold)
            exit_threshold = max(stable_threshold, timeline.thresholds.exit_stable_threshold)
            activity_quiet = clamp01(1.0 - mean_score / max(exit_threshold, stable_threshold))
            consistency = clamp01(1.0 - stddev / max(stable_threshold * 2.0, 1e-6))
            spike_safety = clamp01(1.0 - max_score / max(exit_threshold * 2.0, 1e-6))
            duration_component = duration_saturation(
                segment.duration_seconds, self._settings.stability_duration_saturation_seconds
            )
            stability_score = normalized_weighted(
                [
                    (activity_quiet, self._settings.stability_difference_weight),
                    (consistency, self._settings.stability_variance_weight),
                    (spike_safety, self._settings.stability_max_spike_weight),
                    (duration_component, self._settings.stability_duration_weight),
                ]
            )
            confidence = normalized_weighted(
                [(stability_score, 0.50), (quality_score, 0.25), (valid_ratio, 0.15), (duration_component, 0.10)]
            )

            timeline_idx = segment_index[segment.segment_id]
            previous = timeline.segments[timeline_idx - 1] if timeline_idx > 0 else None
            next_segment = timeline.segments[timeline_idx + 1] if timeline_idx + 1 < len(timeline.segments) else None
            major_before_pos = bisect.bisect_right(event_times, segment.start_timestamp_seconds) - 1
            major_after_pos = bisect.bisect_left(event_times, segment.end_timestamp_seconds)
            seconds_since = None
            seconds_until = None
            if major_before_pos >= 0:
                seconds_since = max(0.0, segment.start_timestamp_seconds - event_times[major_before_pos])
            if major_after_pos < len(event_times):
                seconds_until = max(0.0, event_times[major_after_pos] - segment.end_timestamp_seconds)

            reasons: list[StabilityRejectionReason] = []
            if segment.duration_seconds < self._settings.stability_window_min_duration_seconds:
                reasons.append(StabilityRejectionReason.TOO_SHORT)
            if invalid_ratio > self._settings.stability_max_invalid_frame_ratio:
                reasons.append(StabilityRejectionReason.TOO_MANY_INVALID_FRAMES)
            if black_ratio > self._settings.stability_max_black_frame_ratio:
                reasons.append(StabilityRejectionReason.TOO_MANY_BLACK_FRAMES)
            if not scores:
                reasons.append(StabilityRejectionReason.INSUFFICIENT_VALID_COMPARISONS)

            windows.append(
                StabilityWindow(
                    window_id=ordinal,
                    source_segment_id=segment.segment_id,
                    start_timestamp_seconds=segment.start_timestamp_seconds,
                    end_timestamp_seconds=segment.end_timestamp_seconds,
                    duration_seconds=segment.duration_seconds,
                    start_frame_index=segment.start_frame_index,
                    end_frame_index=segment.end_frame_index,
                    comparison_count=len(scores),
                    mean_difference_score=mean_score,
                    median_difference_score=median_score,
                    max_difference_score=max_score,
                    difference_stddev=stddev,
                    stability_score=stability_score,
                    quality_score=quality_score,
                    window_confidence=confidence,
                    frame_count=total_frames,
                    valid_frame_ratio=valid_ratio,
                    black_frame_ratio=black_ratio,
                    blurry_frame_ratio=blurry_ratio,
                    low_information_ratio=low_info_ratio,
                    average_sharpness=_avg_optional(frame.sharpness_score for frame in valid_frames),
                    average_brightness=_avg_optional(frame.brightness for frame in valid_frames),
                    average_contrast=_avg_optional(frame.contrast for frame in valid_frames),
                    average_edge_density=_avg_optional(frame.edge_density for frame in valid_frames),
                    previous_segment_state=previous.state if previous else None,
                    previous_segment_start_seconds=previous.start_timestamp_seconds if previous else None,
                    previous_segment_end_seconds=previous.end_timestamp_seconds if previous else None,
                    previous_segment_duration_seconds=previous.duration_seconds if previous else None,
                    next_segment_state=next_segment.state if next_segment else None,
                    next_segment_start_seconds=next_segment.start_timestamp_seconds if next_segment else None,
                    next_segment_end_seconds=next_segment.end_timestamp_seconds if next_segment else None,
                    seconds_since_previous_major_transition=seconds_since,
                    seconds_until_next_major_transition=seconds_until,
                    is_valid=not reasons,
                    rejection_reasons=reasons,
                )
            )
            if stable_segments:
                progress_callback(100.0 * ordinal / len(stable_segments))

        valid = [window for window in windows if window.is_valid]
        durations = [window.duration_seconds for window in valid]
        stability_scores = [window.stability_score for window in valid]
        stats = StabilityWindowStats(
            total_stable_segments=len(windows),
            valid_stability_windows=len(valid),
            rejected_stability_windows=len(windows) - len(valid),
            total_stable_duration_seconds=sum(window.duration_seconds for window in windows),
            valid_stability_duration_seconds=sum(durations),
            mean_window_duration_seconds=mean(durations),
            median_window_duration_seconds=median(durations),
            p75_window_duration_seconds=percentile(durations, 75),
            p90_window_duration_seconds=percentile(durations, 90),
            mean_stability_score=mean(stability_scores),
            median_stability_score=median(stability_scores),
            p90_stability_score=percentile(stability_scores, 90),
            rejected_too_short=sum(StabilityRejectionReason.TOO_SHORT in window.rejection_reasons for window in windows),
            rejected_black=sum(StabilityRejectionReason.TOO_MANY_BLACK_FRAMES in window.rejection_reasons for window in windows),
            rejected_invalid=sum(StabilityRejectionReason.TOO_MANY_INVALID_FRAMES in window.rejection_reasons for window in windows),
        )
        manifest = StabilityWindowsManifest(
            algorithm_version=STABILITY_WINDOW_ALGORITHM_VERSION,
            timeline_fingerprint=timeline.artifact_fingerprint,
            differences_fingerprint=differences.artifact_fingerprint,
            preprocessing_fingerprint=preprocessing.artifact_fingerprint,
            major_changes_fingerprint=major_changes.artifact_fingerprint,
            config_fingerprint=stability_config_fingerprint(self._settings),
            artifact_fingerprint="pending",
            stats=stats,
            windows=windows,
            detection_seconds=round(time.monotonic() - started, 3),
        )
        manifest.artifact_fingerprint = stability_artifact_fingerprint(manifest)
        atomic_write_json(self._workspace.stability_windows_path(job_id), manifest.model_dump(mode="json"))
        progress_callback(100.0)
        return manifest

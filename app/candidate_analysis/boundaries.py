from __future__ import annotations

import bisect
import time
from typing import Callable

from app.candidate_analysis.models import (
    BoundariesManifest,
    BoundaryRejectionReason,
    BoundaryStats,
    BoundaryType,
    StabilityWindowsManifest,
    StableBoundary,
)
from app.candidate_analysis.utils import clamp01, duration_saturation, mean, median, normalized_weighted, percentile
from app.core.config import AppSettings
from app.core.exceptions import BoundaryDetectionError, JobCancelledError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import DifferenceManifest, MajorChangesManifest, TimelineManifest, TimelineState

BOUNDARY_DETECTION_ALGORITHM_VERSION = "1"
ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def boundary_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": BOUNDARY_DETECTION_ALGORITHM_VERSION,
        "context_seconds": settings.boundary_context_seconds,
        "min_activity_drop": settings.boundary_min_activity_drop,
        "min_score": settings.boundary_min_score,
        "duration_saturation_seconds": settings.boundary_duration_saturation_seconds,
        "drop_weight": settings.boundary_drop_weight,
        "stability_weight": settings.boundary_stability_weight,
        "quality_weight": settings.boundary_quality_weight,
        "duration_weight": settings.boundary_duration_weight,
        "type_weight": settings.boundary_type_weight,
    }


def boundary_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(boundary_config_payload(settings))


def boundary_artifact_fingerprint(manifest: BoundariesManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "stability_windows_fingerprint": manifest.stability_windows_fingerprint,
            "timeline_fingerprint": manifest.timeline_fingerprint,
            "differences_fingerprint": manifest.differences_fingerprint,
            "major_changes_fingerprint": manifest.major_changes_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "stats": manifest.stats.model_dump(mode="json"),
            "boundaries": [item.model_dump(mode="json") for item in manifest.boundaries],
        }
    )


def _boundary_type(previous: TimelineState | None, start: float) -> BoundaryType:
    if previous is TimelineState.CHANGING:
        return BoundaryType.CHANGING_TO_STABLE
    if previous is TimelineState.MAJOR_TRANSITION:
        return BoundaryType.MAJOR_TRANSITION_TO_STABLE
    if previous is TimelineState.BLACK_TRANSITION:
        return BoundaryType.BLACK_TRANSITION_TO_STABLE
    if previous is TimelineState.INVALID:
        return BoundaryType.INVALID_TO_STABLE
    if previous is None and start <= 1e-6:
        return BoundaryType.STABLE_AT_VIDEO_START
    return BoundaryType.OTHER_TO_STABLE


def _type_prior(boundary_type: BoundaryType) -> float:
    return {
        BoundaryType.CHANGING_TO_STABLE: 1.0,
        BoundaryType.MAJOR_TRANSITION_TO_STABLE: 0.95,
        BoundaryType.BLACK_TRANSITION_TO_STABLE: 0.85,
        BoundaryType.STABLE_AT_VIDEO_START: 0.45,
        BoundaryType.INVALID_TO_STABLE: 0.0,
        BoundaryType.OTHER_TO_STABLE: 0.20,
    }[boundary_type]


class StableBoundaryDetector:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def process(
        self,
        *,
        job_id: str,
        windows: StabilityWindowsManifest,
        timeline: TimelineManifest,
        differences: DifferenceManifest,
        major_changes: MajorChangesManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> BoundariesManifest:
        started = time.monotonic()
        valid_windows = [window for window in windows.windows if window.is_valid]
        comparison_times = [record.current_timestamp_seconds for record in differences.comparisons]
        segment_position = {segment.segment_id: idx for idx, segment in enumerate(timeline.segments)}
        results: list[StableBoundary] = []
        stable_threshold = max(1e-6, timeline.thresholds.enter_stable_threshold)
        activity_scale = max(stable_threshold, timeline.thresholds.major_threshold, 1e-6)

        for ordinal, window in enumerate(valid_windows, start=1):
            if cancel_check():
                raise JobCancelledError("Stable-boundary detection was cancelled.")
            boundary_time = window.start_timestamp_seconds
            btype = _boundary_type(window.previous_segment_state, boundary_time)
            pos = segment_position.get(window.source_segment_id)
            if pos is None:
                raise BoundaryDetectionError("Stability window references a missing timeline segment.")
            previous = timeline.segments[pos - 1] if pos > 0 else None

            pre_start = max(0.0, boundary_time - self._settings.boundary_context_seconds)
            post_end = min(window.end_timestamp_seconds, boundary_time + self._settings.boundary_context_seconds)
            pre_left = bisect.bisect_left(comparison_times, pre_start - 1e-9)
            pre_right = bisect.bisect_right(comparison_times, boundary_time + 1e-9)
            post_left = bisect.bisect_left(comparison_times, boundary_time - 1e-9)
            post_right = bisect.bisect_right(comparison_times, post_end + 1e-9)
            pre_scores = [
                float(record.difference_score)
                for record in differences.comparisons[pre_left:pre_right]
                if record.valid and record.difference_score is not None
            ]
            post_scores = [
                float(record.difference_score)
                for record in differences.comparisons[post_left:post_right]
                if record.valid and record.difference_score is not None
            ]
            pre_mean = mean(pre_scores)
            post_mean = mean(post_scores)
            pre_max = max(pre_scores) if pre_scores else 0.0
            post_max = max(post_scores) if post_scores else 0.0
            raw_drop = max(0.0, pre_mean - post_mean)
            activity_drop = clamp01(raw_drop / activity_scale)
            preceding_activity = clamp01(max(pre_mean, pre_max * 0.65) / activity_scale)
            preceding_duration = (
                previous.duration_seconds if previous and previous.state is TimelineState.CHANGING else 0.0
            )
            duration_component = duration_saturation(
                window.duration_seconds, self._settings.boundary_duration_saturation_seconds
            )
            score = normalized_weighted(
                [
                    (activity_drop, self._settings.boundary_drop_weight),
                    (window.stability_score, self._settings.boundary_stability_weight),
                    (window.quality_score, self._settings.boundary_quality_weight),
                    (duration_component, self._settings.boundary_duration_weight),
                    (_type_prior(btype), self._settings.boundary_type_weight),
                ]
            )
            # Low-information stable states are often blank boards/loading states.
            score *= 1.0 - 0.30 * window.low_information_ratio
            score = clamp01(score)

            reasons: list[BoundaryRejectionReason] = []
            if btype is BoundaryType.INVALID_TO_STABLE:
                reasons.append(BoundaryRejectionReason.UNTRUSTED_PREVIOUS_STATE)
            elif btype is BoundaryType.OTHER_TO_STABLE:
                reasons.append(BoundaryRejectionReason.UNTRUSTED_PREVIOUS_STATE)
            if btype not in {BoundaryType.STABLE_AT_VIDEO_START, BoundaryType.INVALID_TO_STABLE}:
                if not pre_scores:
                    reasons.append(BoundaryRejectionReason.INSUFFICIENT_PRE_BOUNDARY_DATA)
                if activity_drop < self._settings.boundary_min_activity_drop:
                    reasons.append(BoundaryRejectionReason.NO_MEANINGFUL_ACTIVITY_DROP)
            if not post_scores:
                reasons.append(BoundaryRejectionReason.INSUFFICIENT_POST_BOUNDARY_DATA)
            if window.quality_score <= 0.05:
                reasons.append(BoundaryRejectionReason.STABLE_WINDOW_LOW_QUALITY)
            if window.stability_score <= 0.05:
                reasons.append(BoundaryRejectionReason.STABLE_WINDOW_TOO_WEAK)
            if score < self._settings.boundary_min_score and btype is not BoundaryType.STABLE_AT_VIDEO_START:
                reasons.append(BoundaryRejectionReason.BOUNDARY_SCORE_TOO_LOW)

            results.append(
                StableBoundary(
                    boundary_id=ordinal,
                    boundary_type=btype,
                    timestamp_seconds=boundary_time,
                    previous_segment_start_seconds=previous.start_timestamp_seconds if previous else None,
                    previous_segment_end_seconds=previous.end_timestamp_seconds if previous else None,
                    previous_segment_duration_seconds=previous.duration_seconds if previous else None,
                    stable_window_id=window.window_id,
                    stable_window_start_seconds=window.start_timestamp_seconds,
                    stable_window_end_seconds=window.end_timestamp_seconds,
                    last_changing_frame_index=(
                        previous.end_frame_index if previous and previous.state is TimelineState.CHANGING else None
                    ),
                    first_stable_frame_index=window.start_frame_index,
                    pre_boundary_mean_difference=pre_mean,
                    pre_boundary_max_difference=pre_max,
                    post_boundary_mean_difference=post_mean,
                    post_boundary_max_difference=post_max,
                    activity_drop_score=activity_drop,
                    preceding_activity_score=preceding_activity,
                    preceding_changing_duration_seconds=preceding_duration,
                    stability_score=window.stability_score,
                    quality_score=window.quality_score,
                    boundary_score=score,
                    opportunity_duration_seconds=window.duration_seconds,
                    stable_exit_timestamp_seconds=window.end_timestamp_seconds,
                    is_valid=not reasons,
                    rejection_reasons=reasons,
                )
            )
            if valid_windows:
                progress_callback(100.0 * ordinal / len(valid_windows))

        valid = [item for item in results if item.is_valid]
        scores = [item.boundary_score for item in valid]
        drops = [item.activity_drop_score for item in valid]
        stats = BoundaryStats(
            total_boundaries=len(results),
            valid_boundaries=len(valid),
            rejected_boundaries=len(results) - len(valid),
            changing_to_stable_count=sum(item.boundary_type is BoundaryType.CHANGING_TO_STABLE for item in valid),
            major_transition_to_stable_count=sum(
                item.boundary_type is BoundaryType.MAJOR_TRANSITION_TO_STABLE for item in valid
            ),
            black_transition_to_stable_count=sum(
                item.boundary_type is BoundaryType.BLACK_TRANSITION_TO_STABLE for item in valid
            ),
            stable_at_start_count=sum(item.boundary_type is BoundaryType.STABLE_AT_VIDEO_START for item in valid),
            mean_boundary_score=mean(scores),
            median_boundary_score=median(scores),
            p90_boundary_score=percentile(scores, 90),
            p95_boundary_score=percentile(scores, 95),
            mean_activity_drop_score=mean(drops),
        )
        manifest = BoundariesManifest(
            algorithm_version=BOUNDARY_DETECTION_ALGORITHM_VERSION,
            stability_windows_fingerprint=windows.artifact_fingerprint,
            timeline_fingerprint=timeline.artifact_fingerprint,
            differences_fingerprint=differences.artifact_fingerprint,
            major_changes_fingerprint=major_changes.artifact_fingerprint,
            config_fingerprint=boundary_config_fingerprint(self._settings),
            artifact_fingerprint="pending",
            stats=stats,
            boundaries=results,
            detection_seconds=round(time.monotonic() - started, 3),
        )
        manifest.artifact_fingerprint = boundary_artifact_fingerprint(manifest)
        atomic_write_json(self._workspace.boundaries_path(job_id), manifest.model_dump(mode="json"))
        progress_callback(100.0)
        return manifest

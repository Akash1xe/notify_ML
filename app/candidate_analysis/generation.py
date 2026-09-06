from __future__ import annotations

import bisect
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from app.candidate_analysis.models import (
    BoundariesManifest,
    CandidateGenerationStats,
    CandidateRejectionReason,
    CandidateSourceType,
    CandidateType,
    GeneratedCandidate,
    GeneratedCandidatesManifest,
    StabilityWindow,
    StabilityWindowsManifest,
)
from app.candidate_analysis.utils import clamp01, mean
from app.core.config import AppSettings
from app.core.exceptions import CandidateGenerationError, JobCancelledError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import FrameQuality, PreprocessingManifest, SamplingManifest

CANDIDATE_GENERATION_ALGORITHM_VERSION = "1"
ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]

_TYPE_PRIORITY = {
    CandidateType.SETTLED_START: 0,
    CandidateType.MID_STABLE: 1,
    CandidateType.LATE_STABLE: 2,
    CandidateType.PRE_EXIT: 3,
    CandidateType.EARLY_STABLE: 4,
}


def generation_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": CANDIDATE_GENERATION_ALGORITHM_VERSION,
        "settle_delay_seconds": settings.candidate_settle_delay_seconds,
        "exit_margin_seconds": settings.candidate_exit_margin_seconds,
        "min_spacing_seconds": settings.min_candidate_spacing_seconds,
        "max_candidates_per_window": settings.max_candidates_per_window,
        "frame_search_radius_seconds": settings.candidate_frame_search_radius_seconds,
        "allow_window_only_candidates": settings.allow_window_only_candidates,
        "max_total_candidates": settings.max_total_candidates,
        "tie_break": "distance_then_quality_then_later_frame",
    }


def generation_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(generation_config_payload(settings))


def generation_artifact_fingerprint(manifest: GeneratedCandidatesManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "stability_windows_fingerprint": manifest.stability_windows_fingerprint,
            "boundaries_fingerprint": manifest.boundaries_fingerprint,
            "sampling_fingerprint": manifest.sampling_fingerprint,
            "preprocessing_fingerprint": manifest.preprocessing_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "stats": manifest.stats.model_dump(mode="json"),
            "candidates": [candidate.model_dump(mode="json") for candidate in manifest.candidates],
        }
    )


@dataclass(frozen=True)
class _Requested:
    candidate_type: CandidateType
    timestamp: float


def _requested_positions(window: StabilityWindow, settings: AppSettings) -> list[_Requested]:
    duration = window.duration_seconds
    if duration < 3.0:
        desired = 1
    elif duration < 6.0:
        desired = 2
    elif duration < 15.0:
        desired = 3
    else:
        desired = 5
    desired = min(desired, settings.max_candidates_per_window)
    usable_start = window.start_timestamp_seconds + settings.candidate_settle_delay_seconds
    usable_end = window.end_timestamp_seconds - settings.candidate_exit_margin_seconds
    if usable_start > usable_end:
        midpoint = (window.start_timestamp_seconds + window.end_timestamp_seconds) / 2.0
        return [_Requested(CandidateType.MID_STABLE, midpoint)]
    span = usable_end - usable_start
    if desired <= 1:
        return [_Requested(CandidateType.MID_STABLE, usable_start + span * 0.5)]
    if desired == 2:
        return [
            _Requested(CandidateType.SETTLED_START, usable_start),
            _Requested(CandidateType.MID_STABLE, usable_start + span * 0.5),
        ]
    if desired == 3:
        return [
            _Requested(CandidateType.SETTLED_START, usable_start),
            _Requested(CandidateType.MID_STABLE, usable_start + span * 0.5),
            _Requested(CandidateType.PRE_EXIT, usable_end),
        ]
    if desired == 4:
        return [
            _Requested(CandidateType.SETTLED_START, usable_start),
            _Requested(CandidateType.EARLY_STABLE, usable_start + span * 0.33),
            _Requested(CandidateType.LATE_STABLE, usable_start + span * 0.72),
            _Requested(CandidateType.PRE_EXIT, usable_end),
        ]
    return [
        _Requested(CandidateType.SETTLED_START, usable_start),
        _Requested(CandidateType.EARLY_STABLE, usable_start + span * 0.25),
        _Requested(CandidateType.MID_STABLE, usable_start + span * 0.50),
        _Requested(CandidateType.LATE_STABLE, usable_start + span * 0.78),
        _Requested(CandidateType.PRE_EXIT, usable_end),
    ]


def _nearest_valid_frame(
    frames: list[FrameQuality],
    timestamps: list[float],
    requested: float,
    window: StabilityWindow,
    radius: float,
) -> FrameQuality | None:
    left = bisect.bisect_left(timestamps, max(window.start_timestamp_seconds, requested - radius) - 1e-9)
    right = bisect.bisect_right(timestamps, min(window.end_timestamp_seconds, requested + radius) + 1e-9)
    choices = [
        frame
        for frame in frames[left:right]
        if frame.is_valid and not frame.is_black and frame.processed_path and frame.source_path
    ]
    if not choices:
        return None
    # Primary criterion is temporal proximity. For equal/near-equal proximity,
    # prefer the better proxy frame and finally the later frame after a settle boundary.
    return min(
        choices,
        key=lambda frame: (
            round(abs(frame.timestamp_seconds - requested), 9),
            -(frame.quality_score or 0.0),
            -frame.timestamp_seconds,
            frame.index,
        ),
    )


class CandidateGenerator:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def process(
        self,
        *,
        job_id: str,
        windows: StabilityWindowsManifest,
        boundaries: BoundariesManifest,
        sampling: SamplingManifest,
        preprocessing: PreprocessingManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> GeneratedCandidatesManifest:
        started = time.monotonic()
        valid_windows = sorted(
            (window for window in windows.windows if window.is_valid),
            key=lambda window: (window.start_timestamp_seconds, window.window_id),
        )
        window_by_id = {window.window_id: window for window in valid_windows}
        valid_boundary_by_window = {
            item.stable_window_id: item
            for item in boundaries.boundaries
            if item.is_valid and item.stable_window_id in window_by_id
        }
        frames = sorted(preprocessing.frames, key=lambda frame: (frame.timestamp_seconds, frame.index))
        timestamps = [frame.timestamp_seconds for frame in frames]
        records: list[GeneratedCandidate] = []
        temp_id = 1
        shifted = 0

        for offset, window in enumerate(valid_windows, start=1):
            if cancel_check():
                raise JobCancelledError("Candidate generation was cancelled.")
            boundary = valid_boundary_by_window.get(window.window_id)
            if boundary is None and not self._settings.allow_window_only_candidates:
                if valid_windows:
                    progress_callback(100.0 * offset / len(valid_windows))
                continue
            source_type = CandidateSourceType.BOUNDARY if boundary else CandidateSourceType.WINDOW_ONLY
            requested = _requested_positions(window, self._settings)
            # Remove generated requests that are too close before frame mapping.
            filtered: list[_Requested] = []
            for item in requested:
                if not filtered or abs(item.timestamp - filtered[-1].timestamp) >= self._settings.min_candidate_spacing_seconds - 1e-9:
                    filtered.append(item)
            per_window: list[GeneratedCandidate] = []
            for item in filtered:
                frame = _nearest_valid_frame(
                    frames,
                    timestamps,
                    item.timestamp,
                    window,
                    self._settings.candidate_frame_search_radius_seconds,
                )
                reasons: list[CandidateRejectionReason] = []
                if item.timestamp < window.start_timestamp_seconds - 1e-9 or item.timestamp > window.end_timestamp_seconds + 1e-9:
                    reasons.append(CandidateRejectionReason.OUTSIDE_WINDOW)
                if frame is None:
                    reasons.append(CandidateRejectionReason.NO_VALID_FRAME_NEAR_TIMESTAMP)
                    frame_timestamp = None
                    frame_index = None
                    sampled_path = None
                    processed_path = None
                    relative_position = clamp01(
                        (item.timestamp - window.start_timestamp_seconds) / max(window.duration_seconds, 1e-9)
                    )
                else:
                    frame_timestamp = frame.timestamp_seconds
                    frame_index = frame.index
                    sampled_path = frame.source_path
                    processed_path = frame.processed_path
                    relative_position = clamp01(
                        (frame_timestamp - window.start_timestamp_seconds) / max(window.duration_seconds, 1e-9)
                    )
                    if abs(frame_timestamp - item.timestamp) > 1e-6:
                        shifted += 1
                after_boundary = (
                    max(0.0, (frame_timestamp if frame_timestamp is not None else item.timestamp) - boundary.timestamp_seconds)
                    if boundary
                    else None
                )
                before_exit = max(
                    0.0,
                    window.end_timestamp_seconds - (frame_timestamp if frame_timestamp is not None else item.timestamp),
                )
                candidate = GeneratedCandidate(
                    candidate_id=temp_id,
                    stable_window_id=window.window_id,
                    boundary_id=boundary.boundary_id if boundary else None,
                    candidate_type=item.candidate_type,
                    source_type=source_type,
                    boundary_type=boundary.boundary_type if boundary else None,
                    requested_timestamp_seconds=item.timestamp,
                    frame_timestamp_seconds=frame_timestamp,
                    frame_index=frame_index,
                    sampled_frame_path=sampled_path,
                    processed_frame_path=processed_path,
                    stable_window_start_seconds=window.start_timestamp_seconds,
                    stable_window_end_seconds=window.end_timestamp_seconds,
                    stable_window_duration_seconds=window.duration_seconds,
                    relative_position=relative_position,
                    seconds_after_boundary=after_boundary,
                    seconds_before_stable_exit=before_exit,
                    boundary_score=boundary.boundary_score if boundary else 0.0,
                    activity_drop_score=boundary.activity_drop_score if boundary else 0.0,
                    preceding_activity_score=boundary.preceding_activity_score if boundary else 0.0,
                    preceding_changing_duration_seconds=(
                        boundary.preceding_changing_duration_seconds if boundary else 0.0
                    ),
                    stability_score=window.stability_score,
                    window_quality_score=window.quality_score,
                    brightness=frame.brightness if frame else None,
                    contrast=frame.contrast if frame else None,
                    sharpness_score=frame.sharpness_score if frame else None,
                    edge_density=frame.edge_density if frame else None,
                    frame_quality_score=frame.quality_score if frame else None,
                    is_blurry=frame.is_blurry if frame else False,
                    is_black=frame.is_black if frame else False,
                    is_low_information=frame.is_low_information if frame else False,
                    is_valid=not reasons,
                    rejection_reasons=reasons,
                )
                temp_id += 1
                per_window.append(candidate)

            # Multiple strategic requests can map to the same sampled frame. Keep
            # the deterministic type winner and retain losers as rejected diagnostics.
            by_frame: dict[int, list[GeneratedCandidate]] = {}
            for candidate in per_window:
                if candidate.is_valid and candidate.frame_index is not None:
                    by_frame.setdefault(candidate.frame_index, []).append(candidate)
            for collision in by_frame.values():
                if len(collision) <= 1:
                    continue
                collision.sort(key=lambda candidate: (_TYPE_PRIORITY[candidate.candidate_type], candidate.candidate_id))
                for loser in collision[1:]:
                    loser.is_valid = False
                    loser.rejection_reasons.append(CandidateRejectionReason.DUPLICATE_TIMESTAMP)
            records.extend(per_window)
            if valid_windows:
                progress_callback(100.0 * offset / len(valid_windows))

        # Global safety cap applies only to valid candidates and deterministically
        # preserves the strongest temporal opportunities first.
        valid_records = [candidate for candidate in records if candidate.is_valid]
        if len(valid_records) > self._settings.max_total_candidates:
            keep = sorted(
                valid_records,
                key=lambda candidate: (
                    -candidate.boundary_score,
                    -candidate.stability_score,
                    -candidate.window_quality_score,
                    candidate.frame_timestamp_seconds or candidate.requested_timestamp_seconds,
                    candidate.candidate_id,
                ),
            )[: self._settings.max_total_candidates]
            keep_ids = {candidate.candidate_id for candidate in keep}
            for candidate in records:
                if candidate.is_valid and candidate.candidate_id not in keep_ids:
                    candidate.is_valid = False
                    candidate.rejection_reasons.append(CandidateRejectionReason.GLOBAL_CANDIDATE_CAP)

        # Reassign IDs after chronological deterministic ordering so reruns produce
        # stable identifiers independent of implementation iteration order.
        records.sort(
            key=lambda candidate: (
                candidate.frame_timestamp_seconds
                if candidate.frame_timestamp_seconds is not None
                else candidate.requested_timestamp_seconds,
                candidate.stable_window_id,
                _TYPE_PRIORITY[candidate.candidate_type],
                candidate.candidate_id,
            )
        )
        for new_id, candidate in enumerate(records, start=1):
            candidate.candidate_id = new_id

        valid_final = [candidate for candidate in records if candidate.is_valid]
        type_counts = Counter(candidate.candidate_type.value for candidate in valid_final)
        duration_seconds = sampling.video_duration_seconds
        stats = CandidateGenerationStats(
            valid_boundaries=boundaries.stats.valid_boundaries,
            windows_considered=len(valid_windows),
            candidates_generated=len(records),
            valid_candidates=len(valid_final),
            rejected_candidates=len(records) - len(valid_final),
            mean_candidates_per_window=(len(valid_final) / len(valid_windows) if valid_windows else 0.0),
            candidates_per_minute=(len(valid_final) * 60.0 / duration_seconds if duration_seconds > 0 else 0.0),
            shifted_to_nearby_frame_count=shifted,
            candidate_type_distribution=dict(sorted(type_counts.items())),
        )
        manifest = GeneratedCandidatesManifest(
            algorithm_version=CANDIDATE_GENERATION_ALGORITHM_VERSION,
            stability_windows_fingerprint=windows.artifact_fingerprint,
            boundaries_fingerprint=boundaries.artifact_fingerprint,
            sampling_fingerprint=sampling.artifact_fingerprint,
            preprocessing_fingerprint=preprocessing.artifact_fingerprint,
            config_fingerprint=generation_config_fingerprint(self._settings),
            artifact_fingerprint="pending",
            stats=stats,
            candidates=records,
            generation_seconds=round(time.monotonic() - started, 3),
        )
        manifest.artifact_fingerprint = generation_artifact_fingerprint(manifest)
        atomic_write_json(self._workspace.generated_candidates_path(job_id), manifest.model_dump(mode="json"))
        progress_callback(100.0)
        return manifest

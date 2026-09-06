from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, MajorChangeAnalysisError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import (
    DifferenceManifest,
    DifferenceRecord,
    MajorChangeEvent,
    MajorChangeStats,
    MajorChangeThresholds,
    MajorChangesManifest,
    MajorEventType,
)


MAJOR_CHANGE_ALGORITHM_VERSION = "1"
CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[float], None]


def major_change_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": MAJOR_CHANGE_ALGORITHM_VERSION,
        "minimum_major_score": settings.major_change_min_score,
        "minimum_very_major_score": settings.very_major_change_min_score,
        "robust_multiplier": settings.major_change_robust_multiplier,
        "merge_window_seconds": settings.major_change_merge_window_seconds,
    }


def major_change_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(major_change_config_payload(settings))


def major_change_artifact_fingerprint(manifest: MajorChangesManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "differences_fingerprint": manifest.differences_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "thresholds": manifest.thresholds.model_dump(mode="json"),
            "stats": manifest.stats.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in manifest.events],
        }
    )


def derive_major_thresholds(differences: DifferenceManifest, settings: AppSettings) -> MajorChangeThresholds:
    scores = [
        item.difference_score
        for item in differences.comparisons
        if item.valid and item.difference_score is not None
    ]
    if not scores:
        return MajorChangeThresholds(
            median_score=0.0,
            mad=0.0,
            p90=0.0,
            p95=0.0,
            p99=0.0,
            major_threshold=settings.major_change_min_score,
            very_major_threshold=settings.very_major_change_min_score,
        )
    arr = np.asarray(scores, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust = median + settings.major_change_robust_multiplier * 1.4826 * mad
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    major = min(1.0, max(settings.major_change_min_score, p95, robust))
    very = min(1.0, max(settings.very_major_change_min_score, p99, major * 1.15))
    very = max(major, very)
    return MajorChangeThresholds(
        median_score=median,
        mad=mad,
        p90=p90,
        p95=p95,
        p99=p99,
        major_threshold=major,
        very_major_threshold=very,
    )


def _metric_agreement(record: DifferenceRecord, threshold: float) -> float:
    metrics = [
        record.pixel_difference,
        record.ssim_difference,
        record.phash_difference,
        record.edge_difference,
    ]
    valid = [value for value in metrics if value is not None]
    if not valid:
        return 0.0
    support_floor = max(0.10, threshold * 0.45)
    return sum(value >= support_floor for value in valid) / len(valid)


@dataclass
class _Candidate:
    kind: str
    record: DifferenceRecord
    score: float
    threshold: float | None = None


def _cluster_candidates(candidates: list[_Candidate], merge_window: float) -> list[list[_Candidate]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item.record.current_timestamp_seconds)
    clusters: list[list[_Candidate]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        previous = clusters[-1][-1]
        if candidate.record.current_timestamp_seconds - previous.record.current_timestamp_seconds <= merge_window:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def _event_from_cluster(
    cluster: list[_Candidate],
    event_id: int,
    thresholds: MajorChangeThresholds,
) -> MajorChangeEvent:
    strongest = max(cluster, key=lambda item: item.score)
    record = strongest.record
    start = min(item.record.previous_timestamp_seconds for item in cluster)
    end = max(item.record.current_timestamp_seconds for item in cluster)

    if strongest.kind == "black":
        event_type = MajorEventType.BLACK_TRANSITION
        confidence = 1.0
        threshold = None
        robust_score = None
        agreement = None
    elif strongest.kind == "invalid":
        event_type = MajorEventType.INVALID_GAP
        confidence = 0.0
        threshold = None
        robust_score = None
        agreement = None
    else:
        event_type = (
            MajorEventType.VERY_MAJOR_VISUAL_CHANGE
            if strongest.score >= thresholds.very_major_threshold
            else MajorEventType.MAJOR_VISUAL_CHANGE
        )
        threshold = thresholds.major_threshold
        scaled_mad = 1.4826 * thresholds.mad
        robust_score = (strongest.score - thresholds.median_score) / (scaled_mad + 1e-6)
        agreement = _metric_agreement(record, thresholds.major_threshold)
        excess = max(0.0, strongest.score - thresholds.major_threshold)
        score_confidence = min(1.0, excess / max(1e-6, 1.0 - thresholds.major_threshold))
        confidence = min(1.0, 0.35 + 0.45 * score_confidence + 0.20 * agreement)

    return MajorChangeEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp_seconds=record.current_timestamp_seconds,
        start_timestamp_seconds=start,
        end_timestamp_seconds=end,
        previous_frame_index=record.previous_frame_index,
        current_frame_index=record.current_frame_index,
        difference_score=record.difference_score,
        threshold=threshold,
        robust_score=robust_score,
        metric_agreement=agreement,
        confidence=confidence,
    )


class MajorChangeDetector:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def process(
        self,
        *,
        job_id: str,
        differences: DifferenceManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> MajorChangesManifest:
        started = time.monotonic()
        thresholds = derive_major_thresholds(differences, self._settings)
        major_candidates: list[_Candidate] = []
        black_candidates: list[_Candidate] = []
        invalid_candidates: list[_Candidate] = []
        total = len(differences.comparisons)

        for index, record in enumerate(differences.comparisons, start=1):
            if cancel_check():
                raise JobCancelledError("Major visual change detection was cancelled.")
            if not record.valid:
                invalid_candidates.append(_Candidate("invalid", record, 0.0))
            elif record.previous_is_black or record.current_is_black:
                black_candidates.append(
                    _Candidate("black", record, record.difference_score or 0.0)
                )
            elif (
                record.difference_score is not None
                and record.difference_score >= thresholds.major_threshold
            ):
                major_candidates.append(
                    _Candidate(
                        "major",
                        record,
                        record.difference_score,
                        thresholds.major_threshold,
                    )
                )
            if total:
                progress_callback(min(90.0, index * 90.0 / total))

        events: list[MajorChangeEvent] = []
        event_id = 1
        for candidates in (black_candidates, invalid_candidates, major_candidates):
            for cluster in _cluster_candidates(
                candidates, self._settings.major_change_merge_window_seconds
            ):
                events.append(_event_from_cluster(cluster, event_id, thresholds))
                event_id += 1
        events.sort(key=lambda event: (event.timestamp_seconds, event.event_id))
        for index, event in enumerate(events, start=1):
            event.event_id = index

        major_like = [
            event
            for event in events
            if event.event_type
            in {MajorEventType.MAJOR_VISUAL_CHANGE, MajorEventType.VERY_MAJOR_VISUAL_CHANGE}
        ]
        spacings = [
            major_like[index].timestamp_seconds - major_like[index - 1].timestamp_seconds
            for index in range(1, len(major_like))
        ]
        stats = MajorChangeStats(
            total_comparisons=total,
            major_events=sum(event.event_type is MajorEventType.MAJOR_VISUAL_CHANGE for event in events),
            very_major_events=sum(event.event_type is MajorEventType.VERY_MAJOR_VISUAL_CHANGE for event in events),
            black_transitions=sum(event.event_type is MajorEventType.BLACK_TRANSITION for event in events),
            invalid_gaps=sum(event.event_type is MajorEventType.INVALID_GAP for event in events),
            average_seconds_between_major_changes=(statistics.mean(spacings) if spacings else None),
            median_seconds_between_major_changes=(statistics.median(spacings) if spacings else None),
            event_ratio=(len(major_like) / max(1, differences.stats.valid_comparisons)),
        )
        manifest = MajorChangesManifest(
            algorithm_version=MAJOR_CHANGE_ALGORITHM_VERSION,
            differences_fingerprint=differences.artifact_fingerprint,
            config_fingerprint=major_change_config_fingerprint(self._settings),
            artifact_fingerprint="pending",
            thresholds=thresholds,
            stats=stats,
            events=events,
            detection_seconds=round(time.monotonic() - started, 3),
        )
        manifest.artifact_fingerprint = major_change_artifact_fingerprint(manifest)
        atomic_write_json(self._workspace.major_changes_path(job_id), manifest.model_dump(mode="json"))
        progress_callback(100.0)
        return manifest

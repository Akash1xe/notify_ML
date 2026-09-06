from __future__ import annotations

import bisect
import math
import time
from collections import defaultdict
from typing import Callable

import cv2
import numpy as np

from app.candidate_analysis.models import (
    CandidateHeuristicStats,
    CandidateSourceType,
    GeneratedCandidatesManifest,
    HeuristicRejectionReason,
    ScoredCandidate,
    ScoredCandidatesManifest,
    StabilityWindowsManifest,
    BoundariesManifest,
)
from app.candidate_analysis.utils import clamp01, duration_saturation, mean, normalized_weighted, percentile, safe_workspace_path
from app.core.config import AppSettings
from app.core.exceptions import CandidateHeuristicsError, JobCancelledError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import DifferenceManifest, PreprocessingManifest, TimelineManifest

CANDIDATE_HEURISTICS_ALGORITHM_VERSION = "1"
ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def heuristics_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": CANDIDATE_HEURISTICS_ALGORITHM_VERSION,
        "sharpness_saturation": settings.candidate_sharpness_saturation,
        "local_context_seconds": settings.candidate_local_context_seconds,
        "transition_safe_margin_seconds": settings.candidate_transition_safe_margin_seconds,
        "weights": {
            "boundary": settings.completeness_boundary_weight,
            "stability": settings.completeness_stability_weight,
            "activity": settings.completeness_activity_weight,
            "accumulation": settings.completeness_accumulation_weight,
            "density": settings.completeness_density_weight,
            "position": settings.completeness_position_weight,
            "quality": settings.completeness_quality_weight,
            "safety": settings.completeness_safety_weight,
        },
    }


def heuristics_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(heuristics_config_payload(settings))


def heuristics_artifact_fingerprint(manifest: ScoredCandidatesManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "generated_candidates_fingerprint": manifest.generated_candidates_fingerprint,
            "boundaries_fingerprint": manifest.boundaries_fingerprint,
            "stability_windows_fingerprint": manifest.stability_windows_fingerprint,
            "preprocessing_fingerprint": manifest.preprocessing_fingerprint,
            "differences_fingerprint": manifest.differences_fingerprint,
            "timeline_fingerprint": manifest.timeline_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "stats": manifest.stats.model_dump(mode="json"),
            "candidates": [candidate.model_dump(mode="json") for candidate in manifest.candidates],
        }
    )


def _entropy_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist[hist > 0] / total
    entropy = float(-(probs * np.log2(probs)).sum())
    return clamp01(entropy / 8.0)


def _brightness_quality(value: float | None) -> float:
    if value is None:
        return 0.0
    # Broad plateau keeps dark-mode editors, dark slides and blackboards usable.
    if 0.12 <= value <= 0.88:
        return 1.0
    if value < 0.12:
        return clamp01(value / 0.12)
    return clamp01((1.0 - value) / 0.12)


def _position_score(relative_position: float) -> float:
    target = 0.65
    return clamp01(1.0 - abs(relative_position - target) / 0.70)


class CandidateHeuristicAnalyzer:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def process(
        self,
        *,
        job_id: str,
        generated: GeneratedCandidatesManifest,
        windows: StabilityWindowsManifest,
        boundaries: BoundariesManifest,
        preprocessing: PreprocessingManifest,
        differences: DifferenceManifest,
        timeline: TimelineManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> ScoredCandidatesManifest:
        started = time.monotonic()
        boundary_by_id = {item.boundary_id: item for item in boundaries.boundaries}
        comparison_times = [item.current_timestamp_seconds for item in differences.comparisons]
        stable_threshold = max(1e-6, timeline.thresholds.exit_stable_threshold)
        major_threshold = max(stable_threshold, timeline.thresholds.major_threshold, 1e-6)
        prelim: dict[int, dict[str, float | bool | list[HeuristicRejectionReason]]] = {}
        valid_generated = [item for item in generated.candidates if item.is_valid]

        for offset, candidate in enumerate(generated.candidates, start=1):
            if cancel_check():
                raise JobCancelledError("Candidate heuristic analysis was cancelled.")
            reasons: list[HeuristicRejectionReason] = []
            if not candidate.is_valid or candidate.frame_timestamp_seconds is None or not candidate.processed_frame_path:
                reasons.append(HeuristicRejectionReason.INVALID_FRAME)
                prelim[candidate.candidate_id] = {
                    "visual_quality": 0.0,
                    "content_density": 0.0,
                    "local_stability": 0.0,
                    "transition_risk": 1.0,
                    "safety": 0.0,
                    "preceding_activity": 0.0,
                    "position": _position_score(candidate.relative_position),
                    "entropy": 0.0,
                    "reasons": reasons,
                }
                continue
            if candidate.is_black:
                reasons.append(HeuristicRejectionReason.BLACK_FRAME)
            try:
                image_path = safe_workspace_path(self._workspace, job_id, candidate.processed_frame_path)
            except Exception as exc:
                raise CandidateHeuristicsError("Candidate frame path is invalid.") from exc
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path.exists() else None
            if image is None or image.size == 0:
                reasons.append(HeuristicRejectionReason.IMAGE_DECODE_FAILED)
                entropy = 0.0
            else:
                entropy = _entropy_score(image)

            sharpness_quality = clamp01((candidate.sharpness_score or 0.0) / self._settings.candidate_sharpness_saturation)
            brightness_quality = _brightness_quality(candidate.brightness)
            contrast_quality = clamp01((candidate.contrast or 0.0) * 2.0)
            existing_quality = candidate.frame_quality_score or 0.0
            visual_quality = normalized_weighted(
                [
                    (sharpness_quality, 0.35),
                    (brightness_quality, 0.15),
                    (contrast_quality, 0.15),
                    (existing_quality, 0.35),
                ]
            )
            if candidate.is_blurry:
                visual_quality *= 0.75
            if candidate.is_black or HeuristicRejectionReason.IMAGE_DECODE_FAILED in reasons:
                visual_quality = 0.0

            edge_signal = clamp01((candidate.edge_density or 0.0) * 8.0)
            content_density = normalized_weighted(
                [(edge_signal, 0.55), (contrast_quality, 0.20), (entropy, 0.25)]
            )
            if candidate.is_low_information:
                content_density *= 0.60

            context = self._settings.candidate_local_context_seconds
            left = bisect.bisect_left(comparison_times, candidate.frame_timestamp_seconds - context - 1e-9)
            right = bisect.bisect_right(comparison_times, candidate.frame_timestamp_seconds + context + 1e-9)
            local_scores = [
                float(item.difference_score)
                for item in differences.comparisons[left:right]
                if item.valid and item.difference_score is not None
            ]
            local_mean = mean(local_scores)
            local_max = max(local_scores) if local_scores else 0.0
            local_stability = clamp01(1.0 - local_mean / max(stable_threshold, 1e-6))

            safe_margin = self._settings.candidate_transition_safe_margin_seconds
            if candidate.seconds_after_boundary is None:
                start_risk = 0.20
            else:
                start_risk = clamp01(1.0 - candidate.seconds_after_boundary / safe_margin)
            exit_risk = clamp01(1.0 - candidate.seconds_before_stable_exit / safe_margin)
            spike_risk = clamp01(local_max / major_threshold)
            transition_risk = normalized_weighted(
                [(start_risk, 0.35), (exit_risk, 0.35), (spike_risk, 0.30)]
            )
            safety = clamp01(1.0 - transition_risk)

            boundary = boundary_by_id.get(candidate.boundary_id) if candidate.boundary_id else None
            preceding_duration = boundary.preceding_changing_duration_seconds if boundary else 0.0
            preceding_activity = normalized_weighted(
                [
                    (candidate.preceding_activity_score, 0.75),
                    (
                        duration_saturation(
                            preceding_duration, self._settings.boundary_duration_saturation_seconds
                        ),
                        0.25,
                    ),
                ]
            )
            prelim[candidate.candidate_id] = {
                "visual_quality": clamp01(visual_quality),
                "content_density": clamp01(content_density),
                "local_stability": local_stability,
                "transition_risk": transition_risk,
                "safety": safety,
                "preceding_activity": preceding_activity,
                "position": _position_score(candidate.relative_position),
                "entropy": entropy,
                "reasons": reasons,
            }
            if generated.candidates:
                progress_callback(55.0 * offset / len(generated.candidates))

        # Relative content accumulation uses only the small candidate set and is
        # computed against the earliest valid candidate in the same stable window.
        grouped: dict[int, list] = defaultdict(list)
        for candidate in valid_generated:
            grouped[candidate.stable_window_id].append(candidate)
        for values in grouped.values():
            values.sort(key=lambda item: (item.frame_timestamp_seconds or item.requested_timestamp_seconds, item.candidate_id))

        scored: list[ScoredCandidate] = []
        total = len(generated.candidates)
        for offset, candidate in enumerate(generated.candidates, start=1):
            if cancel_check():
                raise JobCancelledError("Candidate heuristic analysis was cancelled.")
            base = prelim[candidate.candidate_id]
            reasons = list(base["reasons"])
            density = float(base["content_density"])
            accumulation = 0.0
            relative_gain = 0.0
            if candidate.is_valid and grouped.get(candidate.stable_window_id):
                earliest = grouped[candidate.stable_window_id][0]
                early_base = prelim[earliest.candidate_id]
                early_density = float(early_base["content_density"])
                relative_gain = clamp01((density - early_density) / max(0.10, 1.0 - early_density))
                early_edge = earliest.edge_density or 0.0
                edge_gain = clamp01(((candidate.edge_density or 0.0) - early_edge) / max(0.02, 1.0 - early_edge))
                accumulation = normalized_weighted([(relative_gain, 0.65), (edge_gain, 0.35)])

            boundary_feature = candidate.boundary_score if candidate.source_type is CandidateSourceType.BOUNDARY else 0.15
            completeness = normalized_weighted(
                [
                    (boundary_feature, self._settings.completeness_boundary_weight),
                    (float(base["local_stability"]), self._settings.completeness_stability_weight),
                    (float(base["preceding_activity"]), self._settings.completeness_activity_weight),
                    (accumulation, self._settings.completeness_accumulation_weight),
                    (density, self._settings.completeness_density_weight),
                    (float(base["position"]), self._settings.completeness_position_weight),
                    (float(base["visual_quality"]), self._settings.completeness_quality_weight),
                    (float(base["safety"]), self._settings.completeness_safety_weight),
                ]
            )
            scored.append(
                ScoredCandidate(
                    candidate_id=candidate.candidate_id,
                    stable_window_id=candidate.stable_window_id,
                    boundary_id=candidate.boundary_id,
                    candidate_type=candidate.candidate_type,
                    source_type=candidate.source_type,
                    frame_timestamp_seconds=candidate.frame_timestamp_seconds,
                    relative_position=candidate.relative_position,
                    boundary_score=candidate.boundary_score,
                    activity_drop_score=candidate.activity_drop_score,
                    stability_score=candidate.stability_score,
                    window_quality_score=candidate.window_quality_score,
                    visual_quality_score=float(base["visual_quality"]),
                    content_density_score=density,
                    local_stability_score=float(base["local_stability"]),
                    transition_risk_score=float(base["transition_risk"]),
                    transition_safety_score=float(base["safety"]),
                    preceding_activity_strength=float(base["preceding_activity"]),
                    content_accumulation_score=accumulation,
                    relative_content_gain=relative_gain,
                    structural_progress_score=0.0,
                    positional_completeness_score=float(base["position"]),
                    completeness_heuristic_score=completeness,
                    is_valid=candidate.is_valid and not reasons,
                    rejection_reasons=reasons,
                )
            )
            if total:
                progress_callback(55.0 + 45.0 * offset / total)

        valid = [item for item in scored if item.is_valid]
        qualities = [item.visual_quality_score for item in valid]
        densities = [item.content_density_score for item in valid]
        completeness_values = [item.completeness_heuristic_score for item in valid]
        risks = [item.transition_risk_score for item in valid]
        by_type: dict[str, list[float]] = defaultdict(list)
        for item in valid:
            by_type[item.candidate_type.value].append(item.completeness_heuristic_score)
        warnings: list[str] = []
        if completeness_values and sum(value >= 0.95 for value in completeness_values) / len(completeness_values) > 0.8:
            warnings.append("Completeness scores are heavily saturated near 1.0")
        if completeness_values and sum(value <= 0.05 for value in completeness_values) / len(completeness_values) > 0.8:
            warnings.append("Completeness scores are heavily saturated near 0.0")
        stats = CandidateHeuristicStats(
            total_candidates=len(scored),
            valid_candidates=len(valid),
            rejected_candidates=len(scored) - len(valid),
            mean_visual_quality=mean(qualities),
            mean_content_density=mean(densities),
            mean_completeness=mean(completeness_values),
            mean_transition_risk=mean(risks),
            p50_completeness=percentile(completeness_values, 50),
            p75_completeness=percentile(completeness_values, 75),
            p90_completeness=percentile(completeness_values, 90),
            p95_completeness=percentile(completeness_values, 95),
            candidate_type_mean_completeness={key: mean(values) for key, values in sorted(by_type.items())},
            warnings=warnings,
        )
        manifest = ScoredCandidatesManifest(
            algorithm_version=CANDIDATE_HEURISTICS_ALGORITHM_VERSION,
            generated_candidates_fingerprint=generated.artifact_fingerprint,
            boundaries_fingerprint=boundaries.artifact_fingerprint,
            stability_windows_fingerprint=windows.artifact_fingerprint,
            preprocessing_fingerprint=preprocessing.artifact_fingerprint,
            differences_fingerprint=differences.artifact_fingerprint,
            timeline_fingerprint=timeline.artifact_fingerprint,
            config_fingerprint=heuristics_config_fingerprint(self._settings),
            artifact_fingerprint="pending",
            stats=stats,
            candidates=scored,
            scoring_seconds=round(time.monotonic() - started, 3),
        )
        manifest.artifact_fingerprint = heuristics_artifact_fingerprint(manifest)
        atomic_write_json(self._workspace.scored_candidates_path(job_id), manifest.model_dump(mode="json"))
        progress_callback(100.0)
        return manifest

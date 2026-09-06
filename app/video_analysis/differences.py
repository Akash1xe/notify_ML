from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from app.core.config import AppSettings
from app.core.exceptions import DifferenceAnalysisError, JobCancelledError, TooManyInvalidComparisonsError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import (
    DifferenceManifest,
    DifferenceRecord,
    DifferenceStats,
    FrameQuality,
    PreprocessingManifest,
)


DIFFERENCE_ALGORITHM_VERSION = "1"
ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def difference_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": DIFFERENCE_ALGORITHM_VERSION,
        "pixel_weight": settings.diff_pixel_weight,
        "ssim_weight": settings.diff_ssim_weight,
        "phash_weight": settings.diff_phash_weight,
        "edge_weight": settings.diff_edge_weight,
        "phash_size": settings.diff_phash_size,
        "canny_low": settings.diff_canny_low,
        "canny_high": settings.diff_canny_high,
        "max_invalid_comparison_ratio": settings.max_invalid_comparison_ratio,
    }


def difference_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(difference_config_payload(settings))


def difference_artifact_fingerprint(manifest: DifferenceManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "preprocessing_fingerprint": manifest.preprocessing_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "stats": manifest.stats.model_dump(mode="json"),
            "comparisons": [item.model_dump(mode="json") for item in manifest.comparisons],
        }
    )


def normalize_weights(settings: AppSettings) -> tuple[float, float, float, float]:
    raw = (
        settings.diff_pixel_weight,
        settings.diff_ssim_weight,
        settings.diff_phash_weight,
        settings.diff_edge_weight,
    )
    if any(value < 0 for value in raw) or sum(raw) <= 0:
        raise DifferenceAnalysisError("Visual difference metric weights are invalid.")
    total = sum(raw)
    return tuple(value / total for value in raw)  # type: ignore[return-value]


def mean_pixel_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(a, b)) / 255.0)


def ssim_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    similarity = float(structural_similarity(a, b, data_range=255))
    similarity = float(max(0.0, min(1.0, similarity)))
    return similarity, 1.0 - similarity


def perceptual_hash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    size = max(hash_size * 4, 32)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:hash_size, :hash_size]
    flattened = low.flatten()
    median = float(np.median(flattened[1:])) if flattened.size > 1 else float(flattened[0])
    return low > median


def phash_difference(a: np.ndarray, b: np.ndarray, hash_size: int = 8) -> float:
    ha = perceptual_hash(a, hash_size)
    hb = perceptual_hash(b, hash_size)
    return float(np.count_nonzero(ha != hb) / ha.size)


def edge_difference(a: np.ndarray, b: np.ndarray, low: int, high: int) -> float:
    edge_a = cv2.Canny(a, low, high)
    edge_b = cv2.Canny(b, low, high)
    changed = cv2.bitwise_xor(edge_a, edge_b)
    return float(np.count_nonzero(changed) / changed.size) if changed.size else 0.0


def combine_metrics(
    pixel: float,
    ssim_difference: float,
    phash: float,
    edge: float,
    weights: tuple[float, float, float, float],
) -> float:
    score = pixel * weights[0] + ssim_difference * weights[1] + phash * weights[2] + edge * weights[3]
    return float(max(0.0, min(1.0, score)))


def _distribution(scores: list[float]) -> DifferenceStats:
    if not scores:
        return DifferenceStats(total_comparisons=0, valid_comparisons=0, invalid_comparisons=0)
    arr = np.asarray(scores, dtype=np.float64)
    return DifferenceStats(
        total_comparisons=len(scores),
        valid_comparisons=len(scores),
        invalid_comparisons=0,
        mean_difference_score=float(np.mean(arr)),
        median_difference_score=float(np.median(arr)),
        max_difference_score=float(np.max(arr)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


class VisualDifferenceService:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def _safe_path(self, job_id: str, relative: str | None) -> Path | None:
        if not relative:
            return None
        root = self._workspace.workspace(job_id)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    @staticmethod
    def _invalid_record(previous: FrameQuality, current: FrameQuality, reason: str) -> DifferenceRecord:
        return DifferenceRecord(
            previous_frame_index=previous.index,
            current_frame_index=current.index,
            previous_timestamp_seconds=previous.timestamp_seconds,
            current_timestamp_seconds=current.timestamp_seconds,
            delta_seconds=max(0.0, current.timestamp_seconds - previous.timestamp_seconds),
            previous_is_black=previous.is_black,
            current_is_black=current.is_black,
            valid=False,
            reason=reason,
        )

    def process(
        self,
        *,
        job_id: str,
        preprocessing: PreprocessingManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> DifferenceManifest:
        started = time.monotonic()
        frames = preprocessing.frames
        weights = normalize_weights(self._settings)
        comparisons: list[DifferenceRecord] = []
        scores: list[float] = []
        total_pairs = max(0, len(frames) - 1)
        invalid_count = 0
        last_reported = -1

        for pair_index in range(1, len(frames)):
            if cancel_check():
                raise JobCancelledError("Visual difference scoring was cancelled.")
            previous = frames[pair_index - 1]
            current = frames[pair_index]
            if not previous.is_valid or not current.is_valid:
                invalid_count += 1
                comparisons.append(
                    self._invalid_record(
                        previous,
                        current,
                        "previous_frame_invalid" if not previous.is_valid else "current_frame_invalid",
                    )
                )
            else:
                previous_path = self._safe_path(job_id, previous.processed_path)
                current_path = self._safe_path(job_id, current.processed_path)
                prev_img = cv2.imread(str(previous_path), cv2.IMREAD_GRAYSCALE) if previous_path and previous_path.exists() else None
                curr_img = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE) if current_path and current_path.exists() else None
                if prev_img is None or curr_img is None or prev_img.size == 0 or curr_img.size == 0:
                    invalid_count += 1
                    comparisons.append(self._invalid_record(previous, current, "processed_frame_decode_failed"))
                else:
                    if prev_img.shape != curr_img.shape:
                        curr_img = cv2.resize(
                            curr_img,
                            (prev_img.shape[1], prev_img.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    try:
                        pixel = mean_pixel_difference(prev_img, curr_img)
                        similarity, structural_diff = ssim_metrics(prev_img, curr_img)
                        phash = phash_difference(prev_img, curr_img, self._settings.diff_phash_size)
                        edges = edge_difference(
                            prev_img,
                            curr_img,
                            self._settings.diff_canny_low,
                            self._settings.diff_canny_high,
                        )
                        score = combine_metrics(pixel, structural_diff, phash, edges, weights)
                        scores.append(score)
                        comparisons.append(
                            DifferenceRecord(
                                previous_frame_index=previous.index,
                                current_frame_index=current.index,
                                previous_timestamp_seconds=previous.timestamp_seconds,
                                current_timestamp_seconds=current.timestamp_seconds,
                                delta_seconds=max(0.0, current.timestamp_seconds - previous.timestamp_seconds),
                                pixel_difference=pixel,
                                ssim_similarity=similarity,
                                ssim_difference=structural_diff,
                                phash_difference=phash,
                                edge_difference=edges,
                                difference_score=score,
                                previous_is_black=previous.is_black,
                                current_is_black=current.is_black,
                            )
                        )
                    except Exception:
                        invalid_count += 1
                        comparisons.append(self._invalid_record(previous, current, "metric_calculation_failed"))

            percent = int(pair_index * 100 / max(1, total_pairs))
            if percent > last_reported:
                progress_callback(float(percent))
                last_reported = percent

        if total_pairs > 0 and invalid_count / total_pairs > self._settings.max_invalid_comparison_ratio:
            raise TooManyInvalidComparisonsError(
                f"Invalid comparison ratio {invalid_count / total_pairs:.2%} exceeds the configured limit."
            )

        stats = _distribution(scores)
        stats.total_comparisons = total_pairs
        stats.valid_comparisons = len(scores)
        stats.invalid_comparisons = invalid_count
        manifest = DifferenceManifest(
            algorithm_version=DIFFERENCE_ALGORITHM_VERSION,
            preprocessing_fingerprint=preprocessing.artifact_fingerprint,
            config_fingerprint=difference_config_fingerprint(self._settings),
            artifact_fingerprint="pending",
            stats=stats,
            comparisons=comparisons,
            comparison_seconds=round(time.monotonic() - started, 3),
        )
        manifest.artifact_fingerprint = difference_artifact_fingerprint(manifest)
        atomic_write_json(self._workspace.differences_path(job_id), manifest.model_dump(mode="json"))
        progress_callback(100.0)
        return manifest

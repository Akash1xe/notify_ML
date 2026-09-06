from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.core.config import AppSettings
from app.core.exceptions import FramePreprocessingError, JobCancelledError, TooManyInvalidFramesError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import (
    FrameQuality,
    PreprocessingManifest,
    PreprocessingStats,
    SamplingManifest,
)


PREPROCESSING_ALGORITHM_VERSION = "1"
ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def preprocessing_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": PREPROCESSING_ALGORITHM_VERSION,
        "analysis_width": settings.analysis_frame_width,
        "max_invalid_frame_ratio": settings.max_invalid_frame_ratio,
        "dark_frame_threshold": settings.dark_frame_threshold,
        "black_pixel_value_threshold": settings.black_pixel_value_threshold,
        "black_pixel_ratio_threshold": settings.black_pixel_ratio_threshold,
        "blur_variance_threshold": settings.blur_variance_threshold,
        "low_information_edge_density": settings.low_information_edge_density,
        "low_information_contrast": settings.low_information_contrast,
    }


def preprocessing_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(preprocessing_config_payload(settings))


def preprocessing_artifact_fingerprint(manifest: PreprocessingManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "sampling_fingerprint": manifest.sampling_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "analysis_width": manifest.analysis_width,
            "stats": manifest.stats.model_dump(mode="json"),
            "frames": [
                {
                    "index": frame.index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "source_path": frame.source_path,
                    "processed_path": frame.processed_path,
                    "source_width": frame.source_width,
                    "source_height": frame.source_height,
                    "analysis_width": frame.analysis_width,
                    "analysis_height": frame.analysis_height,
                    "brightness": frame.brightness,
                    "contrast": frame.contrast,
                    "sharpness_score": frame.sharpness_score,
                    "edge_density": frame.edge_density,
                    "black_pixel_ratio": frame.black_pixel_ratio,
                    "quality_score": frame.quality_score,
                    "is_valid": frame.is_valid,
                    "is_black": frame.is_black,
                    "is_very_dark": frame.is_very_dark,
                    "is_blurry": frame.is_blurry,
                    "is_low_information": frame.is_low_information,
                    "invalid_reason": frame.invalid_reason,
                }
                for frame in manifest.frames
            ],
        }
    )


def resize_for_analysis(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= target_width:
        return image.copy()
    ratio = target_width / float(width)
    target_height = max(1, int(round(height * ratio)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def compute_quality_metrics(image: np.ndarray, settings: AppSettings) -> dict[str, float | bool]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(min(1.0, np.std(gray) / 127.5))
    sharpness = float(max(0.0, cv2.Laplacian(gray, cv2.CV_64F).var()))
    edges = cv2.Canny(gray, settings.diff_canny_low, settings.diff_canny_high)
    edge_density = float(np.count_nonzero(edges) / edges.size) if edges.size else 0.0
    black_ratio = float(np.mean(gray <= settings.black_pixel_value_threshold))
    very_dark = brightness < settings.dark_frame_threshold
    is_black = very_dark and black_ratio >= settings.black_pixel_ratio_threshold
    is_blurry = sharpness < settings.blur_variance_threshold
    low_information = (
        edge_density < settings.low_information_edge_density
        and contrast < settings.low_information_contrast
    )
    sharpness_reference = max(1.0, settings.blur_variance_threshold * 4.0)
    sharpness_quality = min(1.0, sharpness / sharpness_reference)
    brightness_quality = max(0.0, 1.0 - abs(brightness - 0.5) * 1.5)
    quality_score = float(
        max(
            0.0,
            min(
                1.0,
                0.55 * sharpness_quality
                + 0.25 * brightness_quality
                + 0.20 * min(1.0, edge_density * 8.0 + contrast),
            ),
        )
    )
    if is_black:
        quality_score = 0.0
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness_score": sharpness,
        "edge_density": edge_density,
        "black_pixel_ratio": black_ratio,
        "quality_score": quality_score,
        "is_very_dark": very_dark,
        "is_black": is_black,
        "is_blurry": is_blurry,
        "is_low_information": low_information,
    }


class FramePreprocessor:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def _safe_path(self, job_id: str, relative: str) -> Path:
        root = self._workspace.workspace(job_id)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FramePreprocessingError("Frame path escaped the job workspace.") from exc
        return candidate

    def process(
        self,
        *,
        job_id: str,
        sampling: SamplingManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> PreprocessingManifest:
        started = time.monotonic()
        temp_dir = self._workspace.processed_frames_temp_dir(job_id)
        final_dir = self._workspace.processed_frames_dir(job_id)
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        records: list[FrameQuality] = []
        total = len(sampling.frames)
        invalid = black = blurry = low_information = 0
        last_reported = -1

        try:
            for offset, sampled in enumerate(sampling.frames, start=1):
                if cancel_check():
                    raise JobCancelledError("Frame preprocessing was cancelled.")
                source = self._safe_path(job_id, sampled.relative_path)
                image = cv2.imread(str(source), cv2.IMREAD_COLOR) if source.exists() else None
                if image is None or image.size == 0:
                    invalid += 1
                    records.append(
                        FrameQuality(
                            index=sampled.index,
                            timestamp_seconds=sampled.timestamp_seconds,
                            source_path=sampled.relative_path,
                            is_valid=False,
                            invalid_reason="image_decode_failed",
                        )
                    )
                else:
                    source_height, source_width = image.shape[:2]
                    processed = resize_for_analysis(image, self._settings.analysis_frame_width)
                    metrics = compute_quality_metrics(processed, self._settings)
                    output_name = f"frame_{sampled.index:08d}.jpg"
                    output_temp = temp_dir / output_name
                    if not cv2.imwrite(
                        str(output_temp),
                        processed,
                        [cv2.IMWRITE_JPEG_QUALITY, self._settings.frame_jpeg_quality],
                    ) or not output_temp.exists() or output_temp.stat().st_size <= 0:
                        invalid += 1
                        records.append(
                            FrameQuality(
                                index=sampled.index,
                                timestamp_seconds=sampled.timestamp_seconds,
                                source_path=sampled.relative_path,
                                is_valid=False,
                                invalid_reason="processed_frame_write_failed",
                            )
                        )
                    else:
                        if bool(metrics["is_black"]):
                            black += 1
                        if bool(metrics["is_blurry"]):
                            blurry += 1
                        if bool(metrics["is_low_information"]):
                            low_information += 1
                        analysis_height, analysis_width = processed.shape[:2]
                        records.append(
                            FrameQuality(
                                index=sampled.index,
                                timestamp_seconds=sampled.timestamp_seconds,
                                source_path=sampled.relative_path,
                                processed_path=f"frames/processed/{output_name}",
                                source_width=source_width,
                                source_height=source_height,
                                analysis_width=analysis_width,
                                analysis_height=analysis_height,
                                **metrics,
                            )
                        )

                percent = int(offset * 100 / max(1, total))
                step = self._settings.preprocessing_progress_step_percent
                bucket = percent // step
                if bucket > last_reported:
                    progress_callback(float(percent))
                    last_reported = bucket

            invalid_ratio = invalid / max(1, total)
            if invalid_ratio > self._settings.max_invalid_frame_ratio:
                raise TooManyInvalidFramesError(
                    f"Invalid frame ratio {invalid_ratio:.2%} exceeds the configured limit."
                )

            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temp_dir, final_dir)
            stats = PreprocessingStats(
                total_frames=total,
                valid_frames=total - invalid,
                corrupt_frames=invalid,
                black_frames=black,
                blurry_frames=blurry,
                low_information_frames=low_information,
            )
            manifest = PreprocessingManifest(
                algorithm_version=PREPROCESSING_ALGORITHM_VERSION,
                sampling_fingerprint=sampling.artifact_fingerprint,
                config_fingerprint=preprocessing_config_fingerprint(self._settings),
                artifact_fingerprint="pending",
                analysis_width=self._settings.analysis_frame_width,
                stats=stats,
                frames=records,
                preprocessing_seconds=round(time.monotonic() - started, 3),
            )
            manifest.artifact_fingerprint = preprocessing_artifact_fingerprint(manifest)
            atomic_write_json(
                self._workspace.preprocessing_manifest_path(job_id),
                manifest.model_dump(mode="json"),
            )
            progress_callback(100.0)
            return manifest
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

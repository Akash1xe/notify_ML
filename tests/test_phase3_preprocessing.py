from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.exceptions import JobCancelledError, TooManyInvalidFramesError
from app.video_analysis.preprocessing import (
    FramePreprocessor,
    compute_quality_metrics,
    preprocessing_config_fingerprint,
    resize_for_analysis,
)
from tests.phase3_helpers import build_phase3_workspace, create_sample_manifest, synthetic_frame


def test_resize_preserves_aspect_ratio():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized = resize_for_analysis(image, 640)
    assert resized.shape[:2] == (360, 640)


def test_resize_does_not_upscale():
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    resized = resize_for_analysis(image, 640)
    assert resized.shape[:2] == (180, 320)


def test_brightness_normalization(tmp_path: Path):
    settings, _, _ = build_phase3_workspace(tmp_path)
    black = compute_quality_metrics(synthetic_frame("black"), settings)
    gray = compute_quality_metrics(synthetic_frame("gray"), settings)
    white = compute_quality_metrics(synthetic_frame("blank"), settings)
    assert black["brightness"] == pytest.approx(0.0, abs=0.01)
    assert gray["brightness"] == pytest.approx(0.5, abs=0.02)
    assert white["brightness"] == pytest.approx(1.0, abs=0.01)


def test_black_frame_detection_is_not_just_darkness(tmp_path: Path):
    settings, _, _ = build_phase3_workspace(tmp_path)
    black = compute_quality_metrics(synthetic_frame("black"), settings)
    normal = compute_quality_metrics(synthetic_frame("text"), settings)
    assert black["is_black"] is True
    assert normal["is_black"] is False


def test_blur_metric_drops_after_gaussian_blur(tmp_path: Path):
    settings, _, _ = build_phase3_workspace(tmp_path)
    sharp = synthetic_frame("text")
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    sharp_metrics = compute_quality_metrics(sharp, settings)
    blur_metrics = compute_quality_metrics(blurred, settings)
    assert sharp_metrics["sharpness_score"] > blur_metrics["sharpness_score"]


def test_edge_density_and_contrast_are_higher_for_pattern(tmp_path: Path):
    settings, _, _ = build_phase3_workspace(tmp_path)
    blank = compute_quality_metrics(synthetic_frame("blank"), settings)
    checker = compute_quality_metrics(synthetic_frame("checker"), settings)
    assert checker["edge_density"] > blank["edge_density"]
    assert checker["contrast"] > blank["contrast"]


def test_preprocessor_writes_separate_processed_frames(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path)
    _, sampling = create_sample_manifest(settings, workspace, job_id, ["blank", "text", "dense"])
    result = FramePreprocessor(settings, workspace).process(
        job_id=job_id,
        sampling=sampling,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    assert result.stats.valid_frames == 3
    assert workspace.processed_frames_dir(job_id).is_dir()
    assert len(list(workspace.processed_frames_dir(job_id).glob("*.jpg"))) == 3
    assert workspace.sampled_frames_dir(job_id).is_dir()
    assert workspace.preprocessing_manifest_path(job_id).exists()
    assert all(frame.analysis_width <= settings.analysis_frame_width for frame in result.frames if frame.is_valid)


def test_corrupt_frame_below_ratio_is_recorded_and_processing_continues(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, max_invalid_frame_ratio=0.3)
    _, sampling = create_sample_manifest(settings, workspace, job_id, ["blank"] * 5)
    corrupt = workspace.sampled_frames_dir(job_id) / "frame_00000003.jpg"
    corrupt.write_bytes(b"not-an-image")
    # Keep the Phase-3.1 manifest representative of the file presence; Phase-3.2
    # owns decode validation rather than byte fingerprinting every sample.
    result = FramePreprocessor(settings, workspace).process(
        job_id=job_id,
        sampling=sampling,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    assert result.stats.corrupt_frames == 1
    assert result.stats.valid_frames == 4
    assert result.frames[2].is_valid is False


def test_too_many_corrupt_frames_fails(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, max_invalid_frame_ratio=0.1)
    _, sampling = create_sample_manifest(settings, workspace, job_id, ["blank", "blank"])
    (workspace.sampled_frames_dir(job_id) / "frame_00000001.jpg").write_bytes(b"broken")
    with pytest.raises(TooManyInvalidFramesError):
        FramePreprocessor(settings, workspace).process(
            job_id=job_id,
            sampling=sampling,
            progress_callback=lambda _: None,
            cancel_check=lambda: False,
        )
    assert not workspace.processed_frames_temp_dir(job_id).exists()


def test_preprocessing_cancellation_cleans_temp_directory(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path)
    _, sampling = create_sample_manifest(settings, workspace, job_id, ["blank", "text", "dense"])
    calls = {"count": 0}

    def cancelled():
        calls["count"] += 1
        return calls["count"] >= 2

    with pytest.raises(JobCancelledError):
        FramePreprocessor(settings, workspace).process(
            job_id=job_id,
            sampling=sampling,
            progress_callback=lambda _: None,
            cancel_check=cancelled,
        )
    assert not workspace.processed_frames_temp_dir(job_id).exists()


def test_preprocessing_config_change_changes_fingerprint(tmp_path: Path):
    a, _, _ = build_phase3_workspace(tmp_path / "a", analysis_frame_width=640)
    b, _, _ = build_phase3_workspace(tmp_path / "b", analysis_frame_width=800)
    assert preprocessing_config_fingerprint(a) != preprocessing_config_fingerprint(b)

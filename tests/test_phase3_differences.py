from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.video_analysis.differences import (
    VisualDifferenceService,
    combine_metrics,
    edge_difference,
    mean_pixel_difference,
    normalize_weights,
    phash_difference,
    ssim_metrics,
)
from app.video_analysis.preprocessing import FramePreprocessor
from tests.phase3_helpers import build_phase3_workspace, create_sample_manifest, synthetic_frame


def gray(kind: str) -> np.ndarray:
    return cv2.cvtColor(synthetic_frame(kind), cv2.COLOR_BGR2GRAY)


def test_identical_frames_have_near_zero_metrics():
    a = gray("text")
    similarity, structural = ssim_metrics(a, a)
    assert mean_pixel_difference(a, a) == pytest.approx(0.0)
    assert similarity == pytest.approx(1.0)
    assert structural == pytest.approx(0.0)
    assert phash_difference(a, a, 8) == pytest.approx(0.0)
    assert edge_difference(a, a, 100, 200) == pytest.approx(0.0)


def test_completely_different_frames_have_high_change():
    black = gray("black")
    white = gray("blank")
    similarity, structural = ssim_metrics(black, white)
    assert mean_pixel_difference(black, white) > 0.95
    assert structural > 0.8
    assert similarity < 0.2


def test_small_writing_addition_changes_less_than_scene_replacement():
    blank = gray("blank")
    text = gray("text")
    black = gray("black")
    local = mean_pixel_difference(blank, text)
    replacement = mean_pixel_difference(blank, black)
    assert 0 < local < replacement


def test_small_cursor_change_is_small():
    blank = gray("blank")
    cursor = gray("cursor")
    assert mean_pixel_difference(blank, cursor) < 0.01


def test_brightness_shift_preserves_structure_better_than_pixel_value():
    base = gray("text")
    brighter = cv2.convertScaleAbs(base, alpha=0.9, beta=20)
    pixel = mean_pixel_difference(base, brighter)
    similarity, _ = ssim_metrics(base, brighter)
    assert pixel > 0
    assert similarity > 0.8


def test_edge_difference_detects_new_lines():
    assert edge_difference(gray("blank"), gray("dense"), 100, 200) > 0


def test_phash_distance_orders_identical_and_different():
    text = gray("text")
    dense = gray("dense")
    assert phash_difference(text, text, 8) == 0
    assert phash_difference(text, dense, 8) > 0


def test_weight_normalization_accepts_non_normalized_weights(tmp_path: Path):
    settings, _, _ = build_phase3_workspace(
        tmp_path,
        diff_pixel_weight=2,
        diff_ssim_weight=3,
        diff_phash_weight=1,
        diff_edge_weight=4,
    )
    weights = normalize_weights(settings)
    assert sum(weights) == pytest.approx(1.0)
    assert weights == pytest.approx((0.2, 0.3, 0.1, 0.4))


def test_all_zero_weights_rejected(tmp_path: Path):
    with pytest.raises(ValidationError):
        build_phase3_workspace(
            tmp_path,
            diff_pixel_weight=0,
            diff_ssim_weight=0,
            diff_phash_weight=0,
            diff_edge_weight=0,
        )


def test_combined_weighting_formula():
    score = combine_metrics(0.2, 0.4, 0.1, 0.3, (0.25, 0.35, 0.2, 0.2))
    assert score == pytest.approx(0.27)


def test_difference_service_uses_persisted_timestamps(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path)
    _, sampling = create_sample_manifest(settings, workspace, job_id, ["blank", "text", "dense"], fps=2)
    preprocessing = FramePreprocessor(settings, workspace).process(
        job_id=job_id,
        sampling=sampling,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    result = VisualDifferenceService(settings, workspace).process(
        job_id=job_id,
        preprocessing=preprocessing,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    assert result.stats.total_comparisons == 2
    assert result.comparisons[0].previous_timestamp_seconds == 0
    assert result.comparisons[0].current_timestamp_seconds == 0.5
    assert result.comparisons[0].delta_seconds == 0.5
    assert 0 <= result.comparisons[0].difference_score <= 1
    assert workspace.differences_path(job_id).exists()


def test_difference_service_records_invalid_pair(tmp_path: Path):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, max_invalid_frame_ratio=0.6, max_invalid_comparison_ratio=1)
    _, sampling = create_sample_manifest(settings, workspace, job_id, ["blank", "text", "dense"])
    (workspace.sampled_frames_dir(job_id) / "frame_00000002.jpg").write_bytes(b"broken")
    preprocessing = FramePreprocessor(settings, workspace).process(
        job_id=job_id,
        sampling=sampling,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    result = VisualDifferenceService(settings, workspace).process(
        job_id=job_id,
        preprocessing=preprocessing,
        progress_callback=lambda _: None,
        cancel_check=lambda: False,
    )
    assert result.stats.invalid_comparisons == 2
    assert all(not item.valid for item in result.comparisons)

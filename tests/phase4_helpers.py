from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2

from app.core.config import AppSettings
from app.ingestion.models import IngestionResult, StageTimings
from app.jobs.checkpoints import CheckpointStore
from app.jobs.repository import JobRepository
from app.jobs.service import JobService
from app.media.probe import fingerprint
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.cache import (
    CP_FRAME_ANALYSIS,
    CP_FRAME_DIFFERENCES,
    CP_FRAMES_PREPROCESSED,
    CP_FRAMES_SAMPLED,
    CP_MAJOR_CHANGES,
    CP_TIMELINE,
)
from app.video_analysis.changes import (
    MAJOR_CHANGE_ALGORITHM_VERSION,
    major_change_artifact_fingerprint,
    major_change_config_fingerprint,
)
from app.video_analysis.differences import (
    DIFFERENCE_ALGORITHM_VERSION,
    difference_artifact_fingerprint,
    difference_config_fingerprint,
)
from app.video_analysis.models import (
    DifferenceManifest,
    DifferenceRecord,
    DifferenceStats,
    FrameAnalysisSummary,
    FrameQuality,
    MajorChangeEvent,
    MajorChangeStats,
    MajorChangeThresholds,
    MajorChangesManifest,
    MajorEventType,
    Phase3DiskUsage,
    Phase3Timings,
    PreprocessingManifest,
    PreprocessingStats,
    SampledFrame,
    SamplingManifest,
    TimelineManifest,
    TimelineSegment,
    TimelineState,
    TimelineStats,
    TimelineThresholds,
)
from app.video_analysis.preprocessing import (
    PREPROCESSING_ALGORITHM_VERSION,
    compute_quality_metrics,
    preprocessing_artifact_fingerprint,
    preprocessing_config_fingerprint,
)
from app.video_analysis.sampling import (
    SAMPLING_ALGORITHM_VERSION,
    sampling_artifact_fingerprint,
    sampling_config_fingerprint,
)
from app.video_analysis.timeline import (
    TIMELINE_ALGORITHM_VERSION,
    timeline_artifact_fingerprint,
    timeline_config_fingerprint,
)
from tests.phase3_helpers import synthetic_frame


def _score_for_time(timestamp: int, segments: list[tuple[TimelineState, float, float]]) -> float:
    for state, start, end in segments:
        if start < timestamp <= end + 1e-9 or (timestamp == 1 and start == 0 and timestamp <= end):
            if state is TimelineState.STABLE:
                return 0.02
            if state is TimelineState.CHANGING:
                return 0.18
            if state in {TimelineState.MAJOR_TRANSITION, TimelineState.BLACK_TRANSITION}:
                return 0.65
            return 0.0
    return 0.02


def build_phase4_context(
    tmp_path: Path,
    *,
    settings_overrides: dict | None = None,
    segments: list[tuple[TimelineState, float, float]] | None = None,
    score_overrides: dict[int, float] | None = None,
    frame_kinds: dict[int, str] | None = None,
    major_event_times: list[float] | None = None,
):
    settings_values = {
        "storage_root": tmp_path / "jobs",
        "processor_mode": "candidates",
        "frame_disk_safety_margin_mb": 0,
        "audio_disk_safety_margin_mb": 0,
        "download_disk_safety_margin_mb": 0,
        "log_level": "CRITICAL",
    }
    settings_values.update(settings_overrides or {})
    settings = AppSettings(**settings_values)
    workspace = WorkspaceManager(settings.storage_root)
    workspace.ensure_root()
    jobs = JobService(JobRepository(workspace), workspace)
    checkpoints = CheckpointStore(workspace)
    job = jobs.create_job("https://youtube.com/watch?v=dQw4w9WgXcQ")
    jobs.mark_running(job.id)

    segments = segments or [
        (TimelineState.CHANGING, 0.0, 8.0),
        (TimelineState.STABLE, 8.0, 16.0),
        (TimelineState.CHANGING, 16.0, 20.0),
    ]
    duration = max(end for _, _, end in segments)
    frame_count = int(duration) + 1
    frame_kinds = frame_kinds or {}
    source = workspace.source_dir(job.id) / "video.mp4"
    source.write_bytes(b"phase4-source-video")
    sampled_dir = workspace.sampled_frames_dir(job.id)
    processed_dir = workspace.processed_frames_dir(job.id)
    sampled_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    sampled_records: list[SampledFrame] = []
    quality_records: list[FrameQuality] = []
    for index in range(1, frame_count + 1):
        timestamp = float(index - 1)
        kind = frame_kinds.get(index - 1, "text")
        image = synthetic_frame(kind)
        sampled_path = sampled_dir / f"frame_{index:08d}.jpg"
        processed_path = processed_dir / f"frame_{index:08d}.jpg"
        assert cv2.imwrite(str(sampled_path), image)
        assert cv2.imwrite(str(processed_path), image)
        metrics = compute_quality_metrics(image, settings)
        sampled_records.append(
            SampledFrame(
                index=index,
                timestamp_seconds=timestamp,
                relative_path=f"frames/sampled/{sampled_path.name}",
                file_size_bytes=sampled_path.stat().st_size,
            )
        )
        quality_records.append(
            FrameQuality(
                index=index,
                timestamp_seconds=timestamp,
                source_path=f"frames/sampled/{sampled_path.name}",
                processed_path=f"frames/processed/{processed_path.name}",
                source_width=image.shape[1],
                source_height=image.shape[0],
                analysis_width=image.shape[1],
                analysis_height=image.shape[0],
                brightness=float(metrics["brightness"]),
                contrast=float(metrics["contrast"]),
                sharpness_score=float(metrics["sharpness_score"]),
                edge_density=float(metrics["edge_density"]),
                black_pixel_ratio=float(metrics["black_pixel_ratio"]),
                quality_score=float(metrics["quality_score"]),
                is_valid=True,
                is_black=bool(metrics["is_black"]),
                is_very_dark=bool(metrics["is_very_dark"]),
                is_blurry=bool(metrics["is_blurry"]),
                is_low_information=bool(metrics["is_low_information"]),
            )
        )

    sampling = SamplingManifest(
        algorithm_version=SAMPLING_ALGORITHM_VERSION,
        source_video="source/video.mp4",
        source_fingerprint=fingerprint(source),
        sample_fps=1.0,
        interval_seconds=1.0,
        video_duration_seconds=duration,
        expected_frame_count=frame_count,
        actual_frame_count=frame_count,
        jpeg_quality=settings.frame_jpeg_quality,
        config_fingerprint=sampling_config_fingerprint(settings),
        artifact_fingerprint="pending",
        frames=sampled_records,
    )
    sampling.artifact_fingerprint = sampling_artifact_fingerprint(sampling)
    atomic_write_json(workspace.frame_manifest_path(job.id), sampling.model_dump(mode="json"))

    preprocessing = PreprocessingManifest(
        algorithm_version=PREPROCESSING_ALGORITHM_VERSION,
        sampling_fingerprint=sampling.artifact_fingerprint,
        config_fingerprint=preprocessing_config_fingerprint(settings),
        artifact_fingerprint="pending",
        analysis_width=settings.analysis_frame_width,
        stats=PreprocessingStats(
            total_frames=len(quality_records),
            valid_frames=len(quality_records),
            corrupt_frames=0,
            black_frames=sum(item.is_black for item in quality_records),
            blurry_frames=sum(item.is_blurry for item in quality_records),
            low_information_frames=sum(item.is_low_information for item in quality_records),
        ),
        frames=quality_records,
    )
    preprocessing.artifact_fingerprint = preprocessing_artifact_fingerprint(preprocessing)
    atomic_write_json(workspace.preprocessing_manifest_path(job.id), preprocessing.model_dump(mode="json"))

    comparisons: list[DifferenceRecord] = []
    scores: list[float] = []
    for timestamp in range(1, int(duration) + 1):
        score = (score_overrides or {}).get(timestamp, _score_for_time(timestamp, segments))
        scores.append(score)
        comparisons.append(
            DifferenceRecord(
                previous_frame_index=timestamp,
                current_frame_index=timestamp + 1,
                previous_timestamp_seconds=float(timestamp - 1),
                current_timestamp_seconds=float(timestamp),
                delta_seconds=1.0,
                pixel_difference=score,
                ssim_similarity=max(0.0, 1.0 - score),
                ssim_difference=score,
                phash_difference=score,
                edge_difference=score,
                difference_score=score,
                previous_is_black=quality_records[timestamp - 1].is_black,
                current_is_black=quality_records[timestamp].is_black,
                valid=True,
            )
        )
    sorted_scores = sorted(scores)
    import numpy as np
    stats = DifferenceStats(
        total_comparisons=len(comparisons),
        valid_comparisons=len(comparisons),
        invalid_comparisons=0,
        mean_difference_score=float(np.mean(scores)) if scores else 0.0,
        median_difference_score=float(np.median(scores)) if scores else 0.0,
        max_difference_score=max(scores) if scores else 0.0,
        p50=float(np.percentile(scores, 50)) if scores else 0.0,
        p75=float(np.percentile(scores, 75)) if scores else 0.0,
        p90=float(np.percentile(scores, 90)) if scores else 0.0,
        p95=float(np.percentile(scores, 95)) if scores else 0.0,
        p99=float(np.percentile(scores, 99)) if scores else 0.0,
    )
    differences = DifferenceManifest(
        algorithm_version=DIFFERENCE_ALGORITHM_VERSION,
        preprocessing_fingerprint=preprocessing.artifact_fingerprint,
        config_fingerprint=difference_config_fingerprint(settings),
        artifact_fingerprint="pending",
        stats=stats,
        comparisons=comparisons,
    )
    differences.artifact_fingerprint = difference_artifact_fingerprint(differences)
    atomic_write_json(workspace.differences_path(job.id), differences.model_dump(mode="json"))

    major_event_times = major_event_times or []
    events = []
    for event_id, event_time in enumerate(major_event_times, start=1):
        current = max(2, min(frame_count, int(round(event_time)) + 1))
        events.append(
            MajorChangeEvent(
                event_id=event_id,
                event_type=MajorEventType.MAJOR_VISUAL_CHANGE,
                timestamp_seconds=event_time,
                start_timestamp_seconds=max(0.0, event_time - 0.2),
                end_timestamp_seconds=min(duration, event_time + 0.2),
                previous_frame_index=current - 1,
                current_frame_index=current,
                difference_score=0.65,
                threshold=0.30,
                robust_score=1.0,
                metric_agreement=1.0,
                confidence=0.95,
            )
        )
    thresholds = MajorChangeThresholds(
        median_score=float(np.median(scores)) if scores else 0.0,
        mad=0.0,
        p90=float(np.percentile(scores, 90)) if scores else 0.0,
        p95=float(np.percentile(scores, 95)) if scores else 0.0,
        p99=float(np.percentile(scores, 99)) if scores else 0.0,
        major_threshold=0.30,
        very_major_threshold=0.60,
    )
    major = MajorChangesManifest(
        algorithm_version=MAJOR_CHANGE_ALGORITHM_VERSION,
        differences_fingerprint=differences.artifact_fingerprint,
        config_fingerprint=major_change_config_fingerprint(settings),
        artifact_fingerprint="pending",
        thresholds=thresholds,
        stats=MajorChangeStats(
            total_comparisons=len(comparisons),
            major_events=len(events),
            very_major_events=0,
            black_transitions=0,
            invalid_gaps=0,
            event_ratio=(len(events) / len(comparisons) if comparisons else 0.0),
        ),
        events=events,
    )
    major.artifact_fingerprint = major_change_artifact_fingerprint(major)
    atomic_write_json(workspace.major_changes_path(job.id), major.model_dump(mode="json"))

    timeline_segments = []
    for segment_id, (state, start, end) in enumerate(segments, start=1):
        segment_scores = [
            record.difference_score
            for record in comparisons
            if record.difference_score is not None and start < record.current_timestamp_seconds <= end + 1e-9
        ]
        timeline_segments.append(
            TimelineSegment(
                segment_id=segment_id,
                state=state,
                start_timestamp_seconds=start,
                end_timestamp_seconds=end,
                duration_seconds=end - start,
                start_frame_index=max(1, int(start) + 1),
                end_frame_index=min(frame_count, int(end) + 1),
                mean_difference_score=float(np.mean(segment_scores)) if segment_scores else None,
                median_difference_score=float(np.median(segment_scores)) if segment_scores else None,
                min_difference_score=min(segment_scores) if segment_scores else None,
                max_difference_score=max(segment_scores) if segment_scores else None,
                change_intensity=(min(1.0, float(np.mean(segment_scores)) / 0.30) if state is TimelineState.CHANGING and segment_scores else None),
                comparison_count=len(segment_scores),
            )
        )
    total = sum(item.duration_seconds for item in timeline_segments)
    stable_duration = sum(item.duration_seconds for item in timeline_segments if item.state is TimelineState.STABLE)
    changing_duration = sum(item.duration_seconds for item in timeline_segments if item.state is TimelineState.CHANGING)
    transition_duration = sum(item.duration_seconds for item in timeline_segments if item.state in {TimelineState.MAJOR_TRANSITION, TimelineState.BLACK_TRANSITION})
    invalid_duration = sum(item.duration_seconds for item in timeline_segments if item.state is TimelineState.INVALID)
    timeline = TimelineManifest(
        algorithm_version=TIMELINE_ALGORITHM_VERSION,
        differences_fingerprint=differences.artifact_fingerprint,
        major_changes_fingerprint=major.artifact_fingerprint,
        config_fingerprint=timeline_config_fingerprint(settings),
        artifact_fingerprint="pending",
        thresholds=TimelineThresholds(
            enter_stable_threshold=0.05,
            exit_stable_threshold=0.08,
            major_threshold=0.30,
        ),
        stats=TimelineStats(
            total_duration_seconds=total,
            stable_duration_seconds=stable_duration,
            changing_duration_seconds=changing_duration,
            transition_duration_seconds=transition_duration,
            invalid_duration_seconds=invalid_duration,
            stable_segment_count=sum(item.state is TimelineState.STABLE for item in timeline_segments),
            changing_segment_count=sum(item.state is TimelineState.CHANGING for item in timeline_segments),
            transition_segment_count=sum(item.state in {TimelineState.MAJOR_TRANSITION, TimelineState.BLACK_TRANSITION} for item in timeline_segments),
            invalid_segment_count=sum(item.state is TimelineState.INVALID for item in timeline_segments),
            stable_ratio=(stable_duration / total if total else 0.0),
            changing_ratio=(changing_duration / total if total else 0.0),
            invalid_ratio=(invalid_duration / total if total else 0.0),
        ),
        segments=timeline_segments,
    )
    timeline.artifact_fingerprint = timeline_artifact_fingerprint(timeline)
    atomic_write_json(workspace.timeline_path(job.id), timeline.model_dump(mode="json"))

    summary = FrameAnalysisSummary(
        sampled_frame_count=frame_count,
        processed_frame_count=frame_count,
        invalid_frame_count=0,
        difference_count=len(comparisons),
        major_event_count=len(events),
        stable_segment_count=timeline.stats.stable_segment_count,
        changing_segment_count=timeline.stats.changing_segment_count,
        analysis_start_seconds=0.0,
        analysis_end_seconds=duration,
        timings=Phase3Timings(),
        disk_usage=Phase3DiskUsage(),
        sampling_fingerprint=sampling.artifact_fingerprint,
        preprocessing_fingerprint=preprocessing.artifact_fingerprint,
        differences_fingerprint=differences.artifact_fingerprint,
        major_changes_fingerprint=major.artifact_fingerprint,
        timeline_fingerprint=timeline.artifact_fingerprint,
    )
    atomic_write_json(workspace.analysis_summary_path(job.id), summary.model_dump(mode="json"))
    ingestion = IngestionResult(
        video_id="dQw4w9WgXcQ",
        title="Synthetic Lecture",
        source_video="source/video.mp4",
        audio_file="audio/audio.wav",
        duration_seconds=duration,
        width=640,
        height=360,
        fps=30,
        video_codec="h264",
        audio_sample_rate=16000,
        audio_channels=1,
        video_size_bytes=source.stat().st_size,
        audio_size_bytes=1,
        workspace_size_bytes=source.stat().st_size,
        timings=StageTimings(),
    )
    atomic_write_json(workspace.ingestion_path(job.id), ingestion.model_dump(mode="json"))
    for checkpoint in (CP_FRAMES_SAMPLED, CP_FRAMES_PREPROCESSED, CP_FRAME_DIFFERENCES, CP_MAJOR_CHANGES, CP_TIMELINE, CP_FRAME_ANALYSIS):
        checkpoints.mark_completed(job.id, checkpoint)
    return settings, workspace, jobs, checkpoints, job, sampling, preprocessing, differences, major, timeline

from pathlib import Path

from app.core.config import AppSettings
from app.jobs.checkpoints import CheckpointStore
from app.video_analysis.cache import (
    CP_FRAME_DIFFERENCES,
    CP_FRAMES_PREPROCESSED,
    CP_MAJOR_CHANGES,
    CP_TIMELINE,
    FrameAnalysisCacheManager,
)
from app.video_analysis.changes import MajorChangeDetector
from app.video_analysis.differences import VisualDifferenceService
from app.video_analysis.models import AnalysisArtifactState, Phase3ResumeStage
from app.video_analysis.preprocessing import FramePreprocessor
from app.video_analysis.timeline import TemporalTimelineService
from tests.phase3_helpers import build_phase3_workspace, create_sample_manifest


def build_artifacts(tmp_path: Path, **overrides):
    settings, workspace, job_id = build_phase3_workspace(tmp_path, **overrides)
    source, sampling = create_sample_manifest(
        settings, workspace, job_id, ["blank", "text", "text", "dense", "dense", "blank"]
    )
    cp = CheckpointStore(workspace)
    preprocessing = FramePreprocessor(settings, workspace).process(
        job_id=job_id, sampling=sampling, progress_callback=lambda _: None, cancel_check=lambda: False
    )
    cp.mark_completed(job_id, CP_FRAMES_PREPROCESSED)
    differences = VisualDifferenceService(settings, workspace).process(
        job_id=job_id, preprocessing=preprocessing, progress_callback=lambda _: None, cancel_check=lambda: False
    )
    cp.mark_completed(job_id, CP_FRAME_DIFFERENCES)
    major = MajorChangeDetector(settings, workspace).process(
        job_id=job_id, differences=differences, progress_callback=lambda _: None, cancel_check=lambda: False
    )
    cp.mark_completed(job_id, CP_MAJOR_CHANGES)
    timeline = TemporalTimelineService(settings, workspace).process(
        job_id=job_id, differences=differences, major_changes=major,
        progress_callback=lambda _: None, cancel_check=lambda: False
    )
    cp.mark_completed(job_id, CP_TIMELINE)
    return settings, workspace, job_id, source, cp


def test_all_stage_artifacts_are_reusable(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    snapshot = FrameAnalysisCacheManager(settings, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.sampling.state is AnalysisArtifactState.VALID
    assert snapshot.preprocessing.state is AnalysisArtifactState.VALID
    assert snapshot.differences.state is AnalysisArtifactState.VALID
    assert snapshot.major_changes.state is AnalysisArtifactState.VALID
    assert snapshot.timeline.state is AnalysisArtifactState.VALID
    assert snapshot.resume_stage is Phase3ResumeStage.PHASE3_READY


def test_sampling_config_change_invalidates_everything_downstream(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    changed = settings.model_copy(update={"frame_sample_fps": 2.0})
    snapshot = FrameAnalysisCacheManager(changed, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.sampling.state is AnalysisArtifactState.STALE
    assert snapshot.preprocessing.state is AnalysisArtifactState.STALE
    assert snapshot.resume_stage is Phase3ResumeStage.SAMPLING_FRAMES


def test_preprocessing_config_change_preserves_sampling(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    changed = settings.model_copy(update={"analysis_frame_width": 800})
    snapshot = FrameAnalysisCacheManager(changed, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.sampling.state is AnalysisArtifactState.VALID
    assert snapshot.preprocessing.state is AnalysisArtifactState.STALE
    assert snapshot.resume_stage is Phase3ResumeStage.PREPROCESSING_FRAMES


def test_difference_weight_change_preserves_images(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    changed = settings.model_copy(update={"diff_ssim_weight": 0.8})
    snapshot = FrameAnalysisCacheManager(changed, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.preprocessing.state is AnalysisArtifactState.VALID
    assert snapshot.differences.state is AnalysisArtifactState.STALE
    assert snapshot.resume_stage is Phase3ResumeStage.DETECTING_CHANGES


def test_major_detector_config_change_invalidates_major_and_timeline(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    changed = settings.model_copy(update={"major_change_merge_window_seconds": 5.0})
    snapshot = FrameAnalysisCacheManager(changed, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.differences.state is AnalysisArtifactState.VALID
    assert snapshot.major_changes.state is AnalysisArtifactState.STALE
    assert snapshot.timeline.state is AnalysisArtifactState.STALE
    assert snapshot.resume_stage is Phase3ResumeStage.DETECTING_MAJOR_CHANGES


def test_timeline_config_change_invalidates_timeline_only(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    changed = settings.model_copy(update={"min_stable_duration_seconds": 5.0})
    snapshot = FrameAnalysisCacheManager(changed, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.major_changes.state is AnalysisArtifactState.VALID
    assert snapshot.timeline.state is AnalysisArtifactState.STALE
    assert snapshot.resume_stage is Phase3ResumeStage.DETECTING_STABILITY


def test_missing_processed_frame_invalidates_preprocessing(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    next(workspace.processed_frames_dir(job_id).glob("*.jpg")).unlink()
    snapshot = FrameAnalysisCacheManager(settings, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.preprocessing.state is AnalysisArtifactState.MISSING
    assert snapshot.resume_stage is Phase3ResumeStage.PREPROCESSING_FRAMES


def test_corrupt_difference_json_resumes_at_difference_stage(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    workspace.differences_path(job_id).write_text("{broken", encoding="utf-8")
    snapshot = FrameAnalysisCacheManager(settings, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.differences.state is AnalysisArtifactState.CORRUPT
    assert snapshot.resume_stage is Phase3ResumeStage.DETECTING_CHANGES


def test_missing_timeline_resumes_timeline_only(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    workspace.timeline_path(job_id).unlink()
    snapshot = FrameAnalysisCacheManager(settings, workspace, CheckpointStore(workspace)).inspect(job_id, source)
    assert snapshot.major_changes.state is AnalysisArtifactState.VALID
    assert snapshot.timeline.state is AnalysisArtifactState.MISSING
    assert snapshot.resume_stage is Phase3ResumeStage.DETECTING_STABILITY


def test_partial_directories_are_cleaned_safely(tmp_path: Path):
    settings, workspace, job_id, source, _ = build_artifacts(tmp_path)
    sample_tmp = workspace.sampled_frames_temp_dir(job_id)
    process_tmp = workspace.processed_frames_temp_dir(job_id)
    sample_tmp.mkdir(exist_ok=True)
    process_tmp.mkdir(exist_ok=True)
    (workspace.analysis_dir(job_id) / "differences.json.tmp").write_text("partial")
    removed = FrameAnalysisCacheManager(settings, workspace, CheckpointStore(workspace)).cleanup_partial_artifacts(job_id)
    assert removed
    assert not sample_tmp.exists()
    assert not process_tmp.exists()
    assert workspace.sampled_frames_dir(job_id).exists()
    assert workspace.processed_frames_dir(job_id).exists()


def test_reconcile_clears_stale_checkpoint_and_downstream_outputs(tmp_path: Path):
    settings, workspace, job_id, source, checkpoints = build_artifacts(tmp_path)
    workspace.differences_path(job_id).unlink()
    cache = FrameAnalysisCacheManager(settings, workspace, checkpoints)
    snapshot = cache.reconcile(job_id, source)
    assert snapshot.resume_stage is Phase3ResumeStage.DETECTING_CHANGES
    assert not checkpoints.is_completed(job_id, CP_FRAME_DIFFERENCES)
    assert not checkpoints.is_completed(job_id, CP_MAJOR_CHANGES)
    assert not checkpoints.is_completed(job_id, CP_TIMELINE)
    assert workspace.sampled_frames_dir(job_id).exists()
    assert workspace.processed_frames_dir(job_id).exists()
    assert not workspace.major_changes_path(job_id).exists()
    assert not workspace.timeline_path(job_id).exists()

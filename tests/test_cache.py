from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.ingestion.cache import (
    CP_AUDIO,
    CP_INGESTION,
    CP_MEDIA,
    CP_METADATA,
    CP_VIDEO,
    ArtifactState,
    CacheManager,
    CleanupManager,
)
from app.ingestion.youtube.models import DownloadResult, YouTubeMetadata
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage
from app.jobs.repository import JobRepository
from app.jobs.service import JobService
from app.media.models import AudioResult, AudioStreamInfo, MediaInspection, VideoStreamInfo
from app.media.probe import fingerprint
from app.storage.workspace import WorkspaceManager, atomic_write_json


def build(tmp_path: Path):
    workspace = WorkspaceManager(tmp_path / "jobs")
    workspace.ensure_root()
    service = JobService(JobRepository(workspace), workspace)
    checkpoints = CheckpointStore(workspace)
    return workspace, service, checkpoints, CacheManager(workspace, checkpoints)


def prepare_complete(tmp_path: Path):
    workspace, service, checkpoints, cache = build(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=dQw4w9WgXcQ")
    metadata = YouTubeMetadata(
        video_id="dQw4w9WgXcQ",
        canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Test",
    )
    atomic_write_json(workspace.youtube_metadata_path(job.id), metadata.model_dump(mode="json"))
    checkpoints.mark_completed(job.id, CP_METADATA)

    video = workspace.source_dir(job.id) / "video.mp4"
    video.write_bytes(b"video-data")
    download = DownloadResult(
        video_id=metadata.video_id,
        path="source/video.mp4",
        container="mp4",
        filesize_bytes=video.stat().st_size,
    )
    atomic_write_json(workspace.download_manifest_path(job.id), download.model_dump(mode="json"))
    checkpoints.mark_completed(job.id, CP_VIDEO)

    source_fp = fingerprint(video)
    media = MediaInspection(
        file_path="source/video.mp4",
        duration_seconds=10,
        file_size_bytes=video.stat().st_size,
        video=VideoStreamInfo(codec="h264", width=1280, height=720, fps=30),
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2),
        source_fingerprint=source_fp,
    )
    atomic_write_json(workspace.media_inspection_path(job.id), media.model_dump(mode="json"))
    checkpoints.mark_completed(job.id, CP_MEDIA)

    audio = workspace.audio_path(job.id)
    audio.write_bytes(b"normalized-audio")
    audio_result = AudioResult(
        path="audio/audio.wav",
        duration_seconds=10,
        file_size_bytes=audio.stat().st_size,
        source_video_id=metadata.video_id,
        source_fingerprint=source_fp,
        audio_fingerprint=fingerprint(audio),
    )
    atomic_write_json(workspace.audio_manifest_path(job.id), audio_result.model_dump(mode="json"))
    checkpoints.mark_completed(job.id, CP_AUDIO)
    checkpoints.mark_completed(job.id, CP_INGESTION)
    return workspace, service, checkpoints, cache, job, video, audio


def test_complete_cache_is_reusable(tmp_path: Path):
    _, _, _, cache, job, _, _ = prepare_complete(tmp_path)
    snapshot = cache.inspect(job.id, job.source_url)
    assert snapshot.metadata.state is ArtifactState.VALID
    assert snapshot.video.state is ArtifactState.VALID
    assert snapshot.media_inspection.state is ArtifactState.VALID
    assert snapshot.audio.state is ArtifactState.VALID
    assert snapshot.ingestion_complete.state is ArtifactState.VALID
    assert cache.determine_resume_stage(job.id, job.source_url) is JobStage.INGESTION_COMPLETE


def test_missing_audio_resumes_audio_only(tmp_path: Path):
    _, _, _, cache, job, _, audio = prepare_complete(tmp_path)
    audio.unlink()
    assert cache.determine_resume_stage(job.id, job.source_url) is JobStage.EXTRACTING_AUDIO
    snapshot = cache.inspect(job.id, job.source_url)
    assert snapshot.audio.state is ArtifactState.MISSING


def test_changed_video_invalidates_media(tmp_path: Path):
    _, _, _, cache, job, video, _ = prepare_complete(tmp_path)
    video.write_bytes(b"changed-video-content")
    snapshot = cache.inspect(job.id, job.source_url)
    assert snapshot.video.state is ArtifactState.STALE
    assert cache.determine_resume_stage(job.id, job.source_url) is JobStage.DOWNLOADING


def test_partial_files_are_removed(tmp_path: Path):
    workspace, _, _, cache, job, _, _ = prepare_complete(tmp_path)
    partial = workspace.source_dir(job.id) / "video.mp4.part"
    partial.write_bytes(b"partial")
    temp = workspace.audio_temp_path(job.id)
    temp.write_bytes(b"partial-audio")
    count, reclaimed = cache.cleanup_partial_artifacts(job.id)
    assert count == 2
    assert reclaimed > 0
    assert not partial.exists() and not temp.exists()


def test_cleanup_only_deletes_old_terminal_jobs(tmp_path: Path):
    workspace, service, _, _ = build(tmp_path)
    old = service.create_job("https://youtube.com/watch?v=dQw4w9WgXcQ")
    service.mark_running(old.id)
    service.mark_completed(old.id)
    loaded = service.get_job(old.id)
    loaded.completed_at = datetime.now(UTC) - timedelta(hours=100)
    service._repository.save(loaded)

    active = service.create_job("https://youtube.com/watch?v=abcdefghijk")
    manager = CleanupManager(workspace, service, retention_hours=72)
    dry = manager.cleanup(dry_run=True)
    assert old.id in dry.job_ids and workspace.workspace(old.id).exists()
    real = manager.cleanup(dry_run=False)
    assert real.jobs_deleted == 1
    assert not workspace.workspace(old.id).exists()
    assert workspace.workspace(active.id).exists()

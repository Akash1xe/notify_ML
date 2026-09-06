from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.core.exceptions import InvalidYouTubeURLError
from app.core.logging import JobEventLogger
from app.ingestion.cache import CP_INGESTION, CacheManager
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.youtube.models import DownloadProgress, DownloadResult, YouTubeMetadata
from app.jobs.checkpoints import CheckpointStore
from app.jobs.repository import JobRepository
from app.jobs.service import JobService
from app.main import create_app
from app.media.models import AudioResult, AudioStreamInfo, MediaInspection, VideoStreamInfo
from app.media.probe import fingerprint
from app.storage.workspace import WorkspaceManager, atomic_write_json
from tests.conftest import wait_for_status


class FakeYouTube:
    def __init__(self, workspace: WorkspaceManager, fail: bool = False):
        self.workspace = workspace
        self.fail = fail
        self.metadata_calls = 0
        self.download_calls = 0

    def extract_metadata(self, job_id: str, source_url: str) -> YouTubeMetadata:
        self.metadata_calls += 1
        if self.fail:
            raise InvalidYouTubeURLError("Only YouTube video URLs are supported.")
        result = YouTubeMetadata(
            video_id="dQw4w9WgXcQ",
            canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Kafka Tutorial",
            duration_seconds=10.0,
            width=1280,
            height=720,
            fps=30,
        )
        atomic_write_json(self.workspace.youtube_metadata_path(job_id), result.model_dump(mode="json"))
        return result

    def download(self, job_id: str, metadata: YouTubeMetadata, *, progress_callback, cancel_check):
        self.download_calls += 1
        progress_callback(DownloadProgress(download_percent=50))
        if cancel_check():
            raise RuntimeError("cancelled")
        path = self.workspace.source_dir(job_id) / "video.mp4"
        path.write_bytes(b"fake-video-bytes")
        result = DownloadResult(
            video_id=metadata.video_id,
            path="source/video.mp4",
            container="mp4",
            filesize_bytes=path.stat().st_size,
            width=1280,
            height=720,
            fps=30,
        )
        atomic_write_json(self.workspace.download_manifest_path(job_id), result.model_dump(mode="json"))
        progress_callback(DownloadProgress(download_percent=100))
        return result


class FakeMediaTools:
    def require_ffmpeg(self):
        return "ffmpeg"

    def require_ffprobe(self):
        return "ffprobe"


class FakeMediaInspector:
    def __init__(self):
        self.calls = 0

    def inspect(self, path: Path, *, logical_path: str, require_video: bool = True):
        self.calls += 1
        return MediaInspection(
            file_path=logical_path,
            container="mp4",
            duration_seconds=10.0,
            file_size_bytes=path.stat().st_size,
            video=VideoStreamInfo(codec="h264", width=1280, height=720, fps=30) if require_video else None,
            audio=AudioStreamInfo(codec="aac" if require_video else "pcm_s16le", sample_rate=48000 if require_video else 16000, channels=2 if require_video else 1),
            source_fingerprint=fingerprint(path),
        )


class FakeAudioExtractor:
    def __init__(self):
        self.calls = 0

    async def extract(
        self,
        *,
        source_path,
        source_media,
        video_id,
        source_fingerprint,
        temp_path,
        final_path,
        manifest_path,
        logical_final_path,
        progress_callback,
        cancel_check,
    ):
        self.calls += 1
        if cancel_check():
            raise RuntimeError("cancelled")
        progress_callback(50)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"fake-normalized-audio")
        result = AudioResult(
            path=logical_final_path,
            duration_seconds=10.0,
            file_size_bytes=final_path.stat().st_size,
            source_video_id=video_id,
            source_fingerprint=source_fingerprint,
            audio_fingerprint=fingerprint(final_path),
        )
        atomic_write_json(manifest_path, result.model_dump(mode="json"))
        progress_callback(100)
        return result


def build_pipeline(tmp_path: Path):
    settings = AppSettings(storage_root=tmp_path / "jobs", processor_mode="ingestion", log_level="CRITICAL")
    workspace = WorkspaceManager(settings.storage_root)
    workspace.ensure_root()
    service = JobService(JobRepository(workspace), workspace)
    checkpoints = CheckpointStore(workspace)
    cache = CacheManager(workspace, checkpoints)
    youtube = FakeYouTube(workspace)
    media_tools = FakeMediaTools()
    inspector = FakeMediaInspector()
    audio = FakeAudioExtractor()
    pipeline = IngestionPipeline(
        settings=settings,
        jobs=service,
        workspace=workspace,
        checkpoints=checkpoints,
        events=JobEventLogger(workspace),
        youtube=youtube,
        media_tools=media_tools,
        media_inspector=inspector,
        audio_extractor=audio,
        cache=cache,
    )
    return settings, workspace, service, checkpoints, cache, pipeline, youtube, inspector, audio


def test_phase2_pipeline_creates_complete_contract(tmp_path: Path):
    _, workspace, service, checkpoints, cache, pipeline, youtube, inspector, audio = build_pipeline(tmp_path)
    job = service.create_job("https://youtube.com/watch?v=dQw4w9WgXcQ")
    service.mark_running(job.id)
    asyncio.run(pipeline.process(job.id))
    final = service.get_job(job.id)
    assert final.status.value == "COMPLETED"
    assert final.progress == 100
    assert workspace.youtube_metadata_path(job.id).exists()
    assert workspace.download_manifest_path(job.id).exists()
    assert (workspace.source_dir(job.id) / "video.mp4").exists()
    assert workspace.media_inspection_path(job.id).exists()
    assert workspace.audio_path(job.id).exists()
    assert workspace.audio_manifest_path(job.id).exists()
    assert workspace.ingestion_path(job.id).exists()
    assert checkpoints.is_completed(job.id, CP_INGESTION)
    assert cache.inspect(job.id, job.source_url).ingestion_complete.valid
    assert youtube.metadata_calls == 1 and youtube.download_calls == 1
    assert inspector.calls == 1 and audio.calls == 1


def test_real_runner_uses_ingestion_orchestrator_with_mocked_boundaries(tmp_path: Path):
    settings = AppSettings(
        storage_root=tmp_path / "jobs",
        processor_mode="ingestion",
        max_concurrent_jobs=1,
        log_level="CRITICAL",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        fake_youtube = FakeYouTube(app.state.workspace_manager)
        app.state.ingestion_pipeline._youtube = fake_youtube
        app.state.ingestion_pipeline._media_tools = FakeMediaTools()
        app.state.ingestion_pipeline._media_inspector = FakeMediaInspector()
        app.state.ingestion_pipeline._audio_extractor = FakeAudioExtractor()
        created = client.post(
            "/api/jobs",
            json={"source_url": "https://youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert created.status_code == 201
        job_id = created.json()["id"]
        final = wait_for_status(client, job_id, "COMPLETED", timeout=3)
        assert final["message"] == "Phase 2 ingestion completed; ready for frame analysis"
        metadata = client.get(f"/api/jobs/{job_id}/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["title"] == "Kafka Tutorial"
        ingestion = client.get(f"/api/jobs/{job_id}/ingestion")
        assert ingestion.status_code == 200
        assert ingestion.json()["video_codec"] == "h264"


def test_domain_failure_is_classified_by_runner(tmp_path: Path):
    settings = AppSettings(
        storage_root=tmp_path / "jobs",
        processor_mode="ingestion",
        max_concurrent_jobs=1,
        log_level="CRITICAL",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.ingestion_pipeline._youtube = FakeYouTube(app.state.workspace_manager, fail=True)
        created = client.post(
            "/api/jobs",
            json={"source_url": "https://youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        final = wait_for_status(client, created.json()["id"], "FAILED")
        assert final["error"]["code"] == "invalid_youtube_url"
        assert final["error"]["category"] == "INPUT"
        assert final["error"]["failed_stage"] == "PREPARING"
        assert client.get("/health").status_code == 200

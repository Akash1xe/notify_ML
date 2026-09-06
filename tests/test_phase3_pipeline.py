from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.core.logging import JobEventLogger
from app.ingestion.models import IngestionResult, StageTimings
from app.jobs.checkpoints import CheckpointStore
from app.jobs.repository import JobRepository
from app.jobs.service import JobService
from app.main import create_app
from app.media.models import MediaInspection, VideoStreamInfo
from app.media.probe import fingerprint
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.cache import (
    CP_FRAME_ANALYSIS,
    CP_FRAMES_SAMPLED,
    FrameAnalysisCacheManager,
)
from app.video_analysis.changes import MajorChangeDetector
from app.video_analysis.differences import VisualDifferenceService
from app.video_analysis.models import SampledFrame, SamplingManifest
from app.video_analysis.pipeline import FrameAnalysisPipeline
from app.video_analysis.preprocessing import FramePreprocessor
from app.video_analysis.repository import FrameAnalysisRepository
from app.video_analysis.sampling import (
    SAMPLING_ALGORITHM_VERSION,
    sampling_artifact_fingerprint,
    sampling_config_fingerprint,
)
from app.video_analysis.timeline import TemporalTimelineService
from tests.conftest import wait_for_status
from tests.phase3_helpers import synthetic_frame
from tests.test_pipeline import FakeAudioExtractor, FakeMediaInspector, FakeMediaTools, FakeYouTube


class FakeFrameSampler:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager):
        self.settings = settings
        self.workspace = workspace
        self.calls = 0

    async def sample(self, *, job_id, source_path, source_media, progress_callback, cancel_check):
        self.calls += 1
        if cancel_check():
            from app.core.exceptions import JobCancelledError
            raise JobCancelledError("cancelled")
        kinds = ["blank", "text", "text", "dense", "dense", "blank"]
        target = self.workspace.sampled_frames_dir(job_id)
        target.mkdir(parents=True, exist_ok=True)
        records = []
        for index, kind in enumerate(kinds, start=1):
            path = target / f"frame_{index:08d}.jpg"
            cv2.imwrite(str(path), synthetic_frame(kind))
            records.append(
                SampledFrame(
                    index=index,
                    timestamp_seconds=float(index - 1),
                    relative_path=f"frames/sampled/{path.name}",
                    file_size_bytes=path.stat().st_size,
                )
            )
        result = SamplingManifest(
            algorithm_version=SAMPLING_ALGORITHM_VERSION,
            source_video=self.workspace.relative_to_workspace(job_id, source_path),
            source_fingerprint=fingerprint(source_path),
            sample_fps=1.0,
            interval_seconds=1.0,
            video_duration_seconds=6.0,
            expected_frame_count=6,
            actual_frame_count=6,
            jpeg_quality=self.settings.frame_jpeg_quality,
            config_fingerprint=sampling_config_fingerprint(self.settings),
            artifact_fingerprint="pending",
            frames=records,
            sampling_seconds=0.01,
        )
        result.artifact_fingerprint = sampling_artifact_fingerprint(result)
        atomic_write_json(self.workspace.frame_manifest_path(job_id), result.model_dump(mode="json"))
        progress_callback(100)
        return result


def build_frame_pipeline(tmp_path: Path):
    settings = AppSettings(
        storage_root=tmp_path / "jobs",
        processor_mode="analysis",
        frame_disk_safety_margin_mb=0,
        log_level="CRITICAL",
    )
    workspace = WorkspaceManager(settings.storage_root)
    workspace.ensure_root()
    jobs = JobService(JobRepository(workspace), workspace)
    checkpoints = CheckpointStore(workspace)
    cache = FrameAnalysisCacheManager(settings, workspace, checkpoints)
    repository = FrameAnalysisRepository(workspace)
    sampler = FakeFrameSampler(settings, workspace)
    pipeline = FrameAnalysisPipeline(
        settings=settings,
        jobs=jobs,
        workspace=workspace,
        checkpoints=checkpoints,
        events=JobEventLogger(workspace),
        sampler=sampler,
        preprocessor=FramePreprocessor(settings, workspace),
        differences=VisualDifferenceService(settings, workspace),
        major_changes=MajorChangeDetector(settings, workspace),
        timeline=TemporalTimelineService(settings, workspace),
        cache=cache,
        repository=repository,
    )
    job = jobs.create_job("https://youtube.com/watch?v=dQw4w9WgXcQ")
    source = workspace.source_dir(job.id) / "video.mp4"
    source.write_bytes(b"fake-phase3-source")
    media = MediaInspection(
        file_path="source/video.mp4",
        duration_seconds=6.0,
        file_size_bytes=source.stat().st_size,
        video=VideoStreamInfo(codec="h264", width=1280, height=720, fps=30),
        source_fingerprint=fingerprint(source),
    )
    atomic_write_json(workspace.media_inspection_path(job.id), media.model_dump(mode="json"))
    ingestion = IngestionResult(
        video_id="dQw4w9WgXcQ",
        title="Lecture",
        source_video="source/video.mp4",
        audio_file="audio/audio.wav",
        duration_seconds=6.0,
        width=1280,
        height=720,
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
    jobs.mark_running(job.id)
    return settings, workspace, jobs, checkpoints, pipeline, sampler, job, ingestion


def test_frame_analysis_pipeline_creates_complete_contract(tmp_path: Path):
    _, workspace, jobs, checkpoints, pipeline, sampler, job, ingestion = build_frame_pipeline(tmp_path)
    summary = asyncio.run(pipeline.process(job.id, ingestion=ingestion, finalize_job=True))
    final = jobs.get_job(job.id)
    assert final.status.value == "COMPLETED"
    assert final.progress == 100
    assert "candidate generation" in final.message
    assert summary.sampled_frame_count == 6
    assert workspace.frame_manifest_path(job.id).exists()
    assert workspace.preprocessing_manifest_path(job.id).exists()
    assert workspace.differences_path(job.id).exists()
    assert workspace.major_changes_path(job.id).exists()
    assert workspace.timeline_path(job.id).exists()
    assert workspace.analysis_summary_path(job.id).exists()
    assert checkpoints.is_completed(job.id, CP_FRAME_ANALYSIS)
    assert sampler.calls == 1


def test_full_phase3_reuse_does_not_resample(tmp_path: Path):
    _, workspace, jobs, checkpoints, pipeline, sampler, job, ingestion = build_frame_pipeline(tmp_path)
    asyncio.run(pipeline.process(job.id, ingestion=ingestion, finalize_job=False))
    assert sampler.calls == 1
    first_progress = jobs.get_job(job.id).progress
    asyncio.run(pipeline.process(job.id, ingestion=ingestion, finalize_job=False))
    assert sampler.calls == 1
    assert jobs.get_job(job.id).progress >= first_progress
    assert checkpoints.is_completed(job.id, CP_FRAME_ANALYSIS)


def test_resume_after_sampling_reuses_sampling_artifact(tmp_path: Path):
    _, workspace, jobs, checkpoints, pipeline, sampler, job, ingestion = build_frame_pipeline(tmp_path)
    asyncio.run(
        sampler.sample(
            job_id=job.id,
            source_path=workspace.source_dir(job.id) / "video.mp4",
            source_media=MediaInspection(
                file_path="source/video.mp4", duration_seconds=6,
                file_size_bytes=(workspace.source_dir(job.id)/"video.mp4").stat().st_size,
                video=VideoStreamInfo(codec="h264", width=1280, height=720, fps=30),
                source_fingerprint=fingerprint(workspace.source_dir(job.id)/"video.mp4"),
            ),
            progress_callback=lambda _: None,
            cancel_check=lambda: False,
        )
    )
    checkpoints.mark_completed(job.id, CP_FRAMES_SAMPLED)
    assert sampler.calls == 1
    asyncio.run(pipeline.process(job.id, ingestion=ingestion, finalize_job=False))
    assert sampler.calls == 1
    assert workspace.timeline_path(job.id).exists()


def test_duplicate_phase3_execution_is_serialized_and_idempotent(tmp_path: Path):
    _, _, _, _, pipeline, sampler, job, ingestion = build_frame_pipeline(tmp_path)

    async def run_both():
        await asyncio.gather(
            pipeline.process(job.id, ingestion=ingestion, finalize_job=False),
            pipeline.process(job.id, ingestion=ingestion, finalize_job=False),
        )

    asyncio.run(run_both())
    assert sampler.calls == 1


def test_fastapi_analysis_mode_runs_phase2_then_phase3_with_mocked_boundaries(tmp_path: Path):
    settings = AppSettings(
        storage_root=tmp_path / "jobs",
        processor_mode="analysis",
        max_concurrent_jobs=1,
        frame_disk_safety_margin_mb=0,
        audio_disk_safety_margin_mb=0,
        download_disk_safety_margin_mb=0,
        log_level="CRITICAL",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        fake_youtube = FakeYouTube(app.state.workspace_manager)
        app.state.ingestion_pipeline._youtube = fake_youtube
        app.state.ingestion_pipeline._media_tools = FakeMediaTools()
        app.state.ingestion_pipeline._media_inspector = FakeMediaInspector()
        app.state.ingestion_pipeline._audio_extractor = FakeAudioExtractor()
        fake_sampler = FakeFrameSampler(settings, app.state.workspace_manager)
        app.state.frame_analysis_pipeline._sampler = fake_sampler
        created = client.post(
            "/api/jobs",
            json={"source_url": "https://youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        job_id = created.json()["id"]
        final = wait_for_status(client, job_id, "COMPLETED", timeout=5)
        assert final["message"] == "Frame analysis completed; ready for stability candidate generation"
        analysis = client.get(f"/api/jobs/{job_id}/analysis")
        assert analysis.status_code == 200
        assert analysis.json()["sampled_frame_count"] == 6
        cache = client.get(f"/api/jobs/{job_id}/analysis/cache")
        assert cache.status_code == 200
        assert cache.json()["frame_analysis_complete"]["state"] == "VALID"
        assert fake_sampler.calls == 1


def test_analysis_endpoint_is_not_ready_before_phase3(tmp_path: Path):
    settings = AppSettings(storage_root=tmp_path / "jobs", processor_mode="fake", fake_processor_step_delay=0)
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/jobs", json={"source_url": "https://youtube.com/watch?v=dQw4w9WgXcQ"})
        job_id = created.json()["id"]
        wait_for_status(client, job_id, "COMPLETED")
        response = client.get(f"/api/jobs/{job_id}/analysis")
        assert response.status_code == 404

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from app.core.config import AppSettings
from app.core.exceptions import CacheValidationError, JobCancelledError
from app.core.logging import JobEventLogger
from app.ingestion.cache import (
    CP_AUDIO,
    CP_INGESTION,
    CP_MEDIA,
    CP_METADATA,
    CP_VIDEO,
    ArtifactState,
    CacheManager,
)
from app.ingestion.models import IngestionResult, StageTimings
from app.ingestion.youtube.models import DownloadProgress
from app.ingestion.youtube.service import YouTubeService
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService
from app.media.audio import AudioExtractor
from app.media.probe import MediaInspector, fingerprint
from app.media.tools import MediaToolsService
from app.storage.workspace import WorkspaceManager, atomic_write_json


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: AppSettings,
        jobs: JobService,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        events: JobEventLogger,
        youtube: YouTubeService,
        media_tools: MediaToolsService,
        media_inspector: MediaInspector,
        audio_extractor: AudioExtractor,
        cache: CacheManager,
    ) -> None:
        self._settings = settings
        self._jobs = jobs
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._events = events
        self._youtube = youtube
        self._media_tools = media_tools
        self._media_inspector = media_inspector
        self._audio_extractor = audio_extractor
        self._cache = cache

    def _cancelled(self, job_id: str) -> bool:
        return self._jobs.get_job(job_id).status is JobStatus.CANCELLED

    def _guard_cancelled(self, job_id: str) -> None:
        if self._cancelled(job_id):
            raise JobCancelledError(f"Job {job_id} was cancelled")

    def _stage(self, job_id: str, stage: JobStage, message: str) -> None:
        self._guard_cancelled(job_id)
        self._jobs.update_stage(job_id, stage, message)
        self._events.write(job_id, level="INFO", stage=stage.value, message=message)

    def _progress(self, job_id: str, value: int, message: str) -> None:
        if self._cancelled(job_id):
            raise JobCancelledError(f"Job {job_id} was cancelled")
        self._jobs.update_progress(job_id, value, message)

    def _download_progress(self, job_id: str, progress: DownloadProgress) -> None:
        if self._cancelled(job_id):
            return
        percent = progress.download_percent
        if percent is None:
            return
        global_progress = 15 + int(min(100.0, max(0.0, percent)) * 0.20)
        try:
            self._jobs.update_progress(job_id, global_progress, "Downloading lecture video")
        except Exception:
            # The cancellation path can win the race with a yt-dlp hook.
            if not self._cancelled(job_id):
                raise

    def _audio_progress(self, job_id: str, percent: float) -> None:
        if self._cancelled(job_id):
            return
        global_progress = 40 + int(min(100.0, max(0.0, percent)) * 0.10)
        try:
            self._jobs.update_progress(job_id, global_progress, "Preparing transcription audio")
        except Exception:
            if not self._cancelled(job_id):
                raise

    def _artifact_path(self, job_id: str, relative: str) -> Path:
        root = self._workspace.workspace(job_id)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CacheValidationError("Cached artifact path escaped the job workspace.") from exc
        return path

    async def process(self, job_id: str) -> None:
        started = time.monotonic()
        timings = StageTimings()
        self._cache.cleanup_partial_artifacts(job_id)
        job = self._jobs.get_job(job_id)
        source_url = job.source_url

        # ----- Metadata -----
        metadata_check, metadata = self._cache.validate_metadata(job_id, source_url)
        if not metadata_check.valid or metadata is None:
            self._cache.invalidate_from(job_id, JobStage.PREPARING)
            self._stage(job_id, JobStage.PREPARING, "Validating YouTube video")
            self._progress(job_id, 5, "Validating YouTube video")
            stage_started = time.monotonic()
            metadata = await asyncio.to_thread(self._youtube.extract_metadata, job_id, source_url)
            timings.metadata_seconds = round(time.monotonic() - stage_started, 3)
            self._checkpoints.mark_completed(job_id, CP_METADATA)
            self._progress(job_id, 15, "Lecture metadata ready")
            self._events.write(job_id, level="INFO", stage="PREPARING", message=f"Metadata ready for video {metadata.video_id}")
        else:
            self._events.write(job_id, level="INFO", stage="PREPARING", message="Metadata cache hit")

        self._guard_cancelled(job_id)
        # Audio extraction and high-quality yt-dlp merging both require FFmpeg;
        # fail before spending bandwidth when the local tools are unavailable.
        self._media_tools.require_ffmpeg()
        self._media_tools.require_ffprobe()

        # ----- Download -----
        video_check, download, video_path = self._cache.validate_video(job_id, metadata)
        if not video_check.valid or download is None or video_path is None:
            self._cache.invalidate_from(job_id, JobStage.DOWNLOADING)
            self._stage(job_id, JobStage.DOWNLOADING, "Downloading lecture video")
            stage_started = time.monotonic()
            download = await asyncio.to_thread(
                self._youtube.download,
                job_id,
                metadata,
                progress_callback=lambda item: self._download_progress(job_id, item),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings.download_seconds = round(time.monotonic() - stage_started, 3)
            self._checkpoints.mark_completed(job_id, CP_VIDEO)
            self._progress(job_id, 35, "Lecture video downloaded")
            video_path = self._artifact_path(job_id, download.path)
            self._events.write(job_id, level="INFO", stage="DOWNLOADING", message="Video download complete")
        else:
            self._events.write(job_id, level="INFO", stage="DOWNLOADING", message="Video cache hit")

        self._guard_cancelled(job_id)

        # ----- Media inspection -----
        media_check, media = self._cache.validate_media(job_id, video_path)
        if not media_check.valid or media is None:
            self._cache.invalidate_from(job_id, JobStage.INSPECTING_MEDIA)
            self._stage(job_id, JobStage.INSPECTING_MEDIA, "Inspecting downloaded media")
            self._progress(job_id, 36, "Checking media tools")
            stage_started = time.monotonic()
            logical_path = self._workspace.relative_to_workspace(job_id, video_path)
            media = await asyncio.to_thread(
                self._media_inspector.inspect,
                video_path,
                logical_path=logical_path,
                require_video=True,
            )
            timings.inspection_seconds = round(time.monotonic() - stage_started, 3)
            atomic_write_json(
                self._workspace.media_inspection_path(job_id),
                media.model_dump(mode="json"),
            )
            self._checkpoints.mark_completed(job_id, CP_MEDIA)
            self._progress(job_id, 40, "Media inspection complete")
            self._events.write(job_id, level="INFO", stage="INSPECTING_MEDIA", message="Media inspection complete")
        else:
            self._events.write(job_id, level="INFO", stage="INSPECTING_MEDIA", message="Media inspection cache hit")

        self._guard_cancelled(job_id)

        # ----- Normalized audio -----
        audio_check, audio = self._cache.validate_audio(job_id, video_path, media)
        if not audio_check.valid or audio is None:
            self._cache.invalidate_from(job_id, JobStage.EXTRACTING_AUDIO)
            self._stage(job_id, JobStage.EXTRACTING_AUDIO, "Preparing transcription audio")
            stage_started = time.monotonic()
            audio = await self._audio_extractor.extract(
                source_path=video_path,
                source_media=media,
                video_id=metadata.video_id,
                source_fingerprint=fingerprint(video_path),
                temp_path=self._workspace.audio_temp_path(job_id),
                final_path=self._workspace.audio_path(job_id),
                manifest_path=self._workspace.audio_manifest_path(job_id),
                logical_final_path="audio/audio.wav",
                progress_callback=lambda percent: self._audio_progress(job_id, percent),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings.audio_seconds = round(time.monotonic() - stage_started, 3)
            self._checkpoints.mark_completed(job_id, CP_AUDIO)
            self._progress(job_id, 50, "Transcription audio ready")
            self._events.write(job_id, level="INFO", stage="EXTRACTING_AUDIO", message="Audio extraction complete")
        else:
            self._events.write(job_id, level="INFO", stage="EXTRACTING_AUDIO", message="Audio cache hit")

        self._guard_cancelled(job_id)

        # Final validation uses the same validators that restart recovery uses.
        snapshot = self._cache.inspect(job_id, source_url)
        if not all(
            getattr(snapshot, field).state is ArtifactState.VALID
            for field in ("metadata", "video", "media_inspection", "audio")
        ):
            raise CacheValidationError("Ingestion artifacts failed final consistency validation.")

        timings.total_seconds = round(time.monotonic() - started, 3)
        result = IngestionResult(
            video_id=metadata.video_id,
            title=metadata.title,
            source_video=download.path,
            audio_file=audio.path,
            duration_seconds=media.duration_seconds,
            width=media.video.width if media.video else None,
            height=media.video.height if media.video else None,
            fps=media.video.fps if media.video else None,
            video_codec=media.video.codec if media.video else None,
            audio_sample_rate=audio.sample_rate,
            audio_channels=audio.channels,
            video_size_bytes=download.filesize_bytes,
            audio_size_bytes=audio.file_size_bytes,
            workspace_size_bytes=self._workspace.workspace_size(job_id),
            timings=timings,
        )
        atomic_write_json(self._workspace.ingestion_path(job_id), result.model_dump(mode="json"))
        self._checkpoints.mark_completed(job_id, CP_INGESTION)
        self._stage(job_id, JobStage.INGESTION_COMPLETE, "Ingestion complete")
        self._events.write(job_id, level="INFO", stage="INGESTION_COMPLETE", message="Phase 2 ingestion complete")
        self._jobs.mark_completed(
            job_id,
            message="Phase 2 ingestion completed; ready for frame analysis",
        )

    def load_result(self, job_id: str) -> IngestionResult | None:
        path = self._workspace.ingestion_path(job_id)
        if not path.exists():
            return None
        try:
            return IngestionResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

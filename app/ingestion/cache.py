from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.ingestion.youtube.models import DownloadResult, YouTubeMetadata
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService
from app.media.models import AudioResult, MediaInspection
from app.media.probe import fingerprint
from app.storage.workspace import WorkspaceManager


CP_METADATA = "YOUTUBE_METADATA_EXTRACTED"
CP_VIDEO = "VIDEO_DOWNLOADED"
CP_MEDIA = "MEDIA_INSPECTED"
CP_AUDIO = "AUDIO_EXTRACTED"
CP_INGESTION = "INGESTION_COMPLETE"


class ArtifactState(str, Enum):
    MISSING = "MISSING"
    VALID = "VALID"
    STALE = "STALE"
    CORRUPT = "CORRUPT"
    PARTIAL = "PARTIAL"


class ArtifactCheck(BaseModel):
    state: ArtifactState
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.state is ArtifactState.VALID


class CacheSnapshot(BaseModel):
    metadata: ArtifactCheck
    video: ArtifactCheck
    media_inspection: ArtifactCheck
    audio: ArtifactCheck
    ingestion_complete: ArtifactCheck


class CleanupResult(BaseModel):
    jobs_scanned: int = 0
    jobs_deleted: int = 0
    partial_files_deleted: int = 0
    bytes_reclaimed: int = 0
    job_ids: list[str] = []


class CacheManager:
    def __init__(
        self,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
    ) -> None:
        self._workspace = workspace
        self._checkpoints = checkpoints

    @staticmethod
    def _load_model(path: Path, model_type):
        if not path.exists():
            return None, ArtifactCheck(state=ArtifactState.MISSING, reason="file_missing")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return model_type.model_validate(data), ArtifactCheck(state=ArtifactState.VALID)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError):
            return None, ArtifactCheck(state=ArtifactState.CORRUPT, reason="invalid_json_or_schema")

    def _resolved_artifact(self, job_id: str, relative: str) -> Path | None:
        workspace = self._workspace.workspace(job_id)
        candidate = (workspace / relative).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError:
            return None
        return candidate

    def validate_metadata(self, job_id: str, source_url: str) -> tuple[ArtifactCheck, YouTubeMetadata | None]:
        if not self._checkpoints.is_completed(job_id, CP_METADATA):
            return ArtifactCheck(state=ArtifactState.MISSING, reason="checkpoint_missing"), None
        metadata, check = self._load_model(self._workspace.youtube_metadata_path(job_id), YouTubeMetadata)
        if metadata is None:
            return check, None
        from app.ingestion.youtube.validator import normalize_youtube_url
        try:
            expected = normalize_youtube_url(source_url)
        except Exception:
            return ArtifactCheck(state=ArtifactState.STALE, reason="source_url_invalid"), None
        if metadata.video_id != expected.video_id:
            return ArtifactCheck(state=ArtifactState.STALE, reason="video_id_mismatch"), None
        return ArtifactCheck(state=ArtifactState.VALID), metadata

    def validate_video(
        self, job_id: str, metadata: YouTubeMetadata | None
    ) -> tuple[ArtifactCheck, DownloadResult | None, Path | None]:
        if metadata is None:
            return ArtifactCheck(state=ArtifactState.STALE, reason="metadata_invalid"), None, None
        if not self._checkpoints.is_completed(job_id, CP_VIDEO):
            return ArtifactCheck(state=ArtifactState.MISSING, reason="checkpoint_missing"), None, None
        result, check = self._load_model(self._workspace.download_manifest_path(job_id), DownloadResult)
        if result is None:
            return check, None, None
        if result.video_id != metadata.video_id:
            return ArtifactCheck(state=ArtifactState.STALE, reason="video_id_mismatch"), None, None
        path = self._resolved_artifact(job_id, result.path)
        if path is None:
            return ArtifactCheck(state=ArtifactState.CORRUPT, reason="unsafe_artifact_path"), None, None
        if not path.exists() or path.stat().st_size <= 0:
            return ArtifactCheck(state=ArtifactState.MISSING, reason="video_file_missing"), None, None
        if path.stat().st_size != result.filesize_bytes:
            return ArtifactCheck(state=ArtifactState.STALE, reason="video_size_changed"), None, None
        return ArtifactCheck(state=ArtifactState.VALID), result, path

    def validate_media(
        self, job_id: str, video_path: Path | None
    ) -> tuple[ArtifactCheck, MediaInspection | None]:
        if video_path is None:
            return ArtifactCheck(state=ArtifactState.STALE, reason="video_invalid"), None
        if not self._checkpoints.is_completed(job_id, CP_MEDIA):
            return ArtifactCheck(state=ArtifactState.MISSING, reason="checkpoint_missing"), None
        result, check = self._load_model(self._workspace.media_inspection_path(job_id), MediaInspection)
        if result is None:
            return check, None
        try:
            current = fingerprint(video_path)
        except OSError:
            return ArtifactCheck(state=ArtifactState.MISSING, reason="video_file_missing"), None
        if current != result.source_fingerprint:
            return ArtifactCheck(state=ArtifactState.STALE, reason="source_fingerprint_mismatch"), None
        if result.video is None:
            return ArtifactCheck(state=ArtifactState.CORRUPT, reason="video_stream_missing"), None
        return ArtifactCheck(state=ArtifactState.VALID), result

    def validate_audio(
        self,
        job_id: str,
        video_path: Path | None,
        media: MediaInspection | None,
    ) -> tuple[ArtifactCheck, AudioResult | None]:
        if video_path is None or media is None:
            return ArtifactCheck(state=ArtifactState.STALE, reason="upstream_invalid"), None
        if not self._checkpoints.is_completed(job_id, CP_AUDIO):
            return ArtifactCheck(state=ArtifactState.MISSING, reason="checkpoint_missing"), None
        result, check = self._load_model(self._workspace.audio_manifest_path(job_id), AudioResult)
        if result is None:
            return check, None
        if fingerprint(video_path) != result.source_fingerprint:
            return ArtifactCheck(state=ArtifactState.STALE, reason="source_fingerprint_mismatch"), None
        audio_path = self._resolved_artifact(job_id, result.path)
        if audio_path is None:
            return ArtifactCheck(state=ArtifactState.CORRUPT, reason="unsafe_artifact_path"), None
        if not audio_path.exists() or audio_path.stat().st_size <= 0:
            return ArtifactCheck(state=ArtifactState.MISSING, reason="audio_file_missing"), None
        if fingerprint(audio_path) != result.audio_fingerprint:
            return ArtifactCheck(state=ArtifactState.STALE, reason="audio_fingerprint_mismatch"), None
        if result.codec != "pcm_s16le" or result.sample_rate != 16000 or result.channels != 1:
            return ArtifactCheck(state=ArtifactState.CORRUPT, reason="audio_normalization_mismatch"), None
        return ArtifactCheck(state=ArtifactState.VALID), result

    def inspect(self, job_id: str, source_url: str) -> CacheSnapshot:
        metadata_check, metadata = self.validate_metadata(job_id, source_url)
        video_check, _, video_path = self.validate_video(job_id, metadata)
        media_check, media = self.validate_media(job_id, video_path)
        audio_check, _ = self.validate_audio(job_id, video_path, media)
        all_valid = all(
            check.valid for check in (metadata_check, video_check, media_check, audio_check)
        )
        ingestion = ArtifactCheck(
            state=ArtifactState.VALID
            if all_valid and self._checkpoints.is_completed(job_id, CP_INGESTION)
            else ArtifactState.MISSING,
            reason=None if all_valid and self._checkpoints.is_completed(job_id, CP_INGESTION) else "artifact_or_checkpoint_missing",
        )
        return CacheSnapshot(
            metadata=metadata_check,
            video=video_check,
            media_inspection=media_check,
            audio=audio_check,
            ingestion_complete=ingestion,
        )

    def determine_resume_stage(self, job_id: str, source_url: str) -> JobStage:
        snapshot = self.inspect(job_id, source_url)
        if not snapshot.metadata.valid:
            return JobStage.PREPARING
        if not snapshot.video.valid:
            return JobStage.DOWNLOADING
        if not snapshot.media_inspection.valid:
            return JobStage.INSPECTING_MEDIA
        if not snapshot.audio.valid:
            return JobStage.EXTRACTING_AUDIO
        return JobStage.INGESTION_COMPLETE

    def invalidate_from(self, job_id: str, stage: JobStage) -> None:
        # Completion markers are cleared before artifacts so no stale state can
        # be observed as complete during a crash.
        order = [
            (JobStage.PREPARING, CP_METADATA),
            (JobStage.DOWNLOADING, CP_VIDEO),
            (JobStage.INSPECTING_MEDIA, CP_MEDIA),
            (JobStage.EXTRACTING_AUDIO, CP_AUDIO),
        ]
        stage_index = {item[0]: index for index, item in enumerate(order)}
        start = stage_index.get(stage, len(order))
        self._checkpoints.invalidate(job_id, CP_INGESTION)
        for index, (_, checkpoint) in enumerate(order):
            if index >= start:
                self._checkpoints.invalidate(job_id, checkpoint)

        if start <= 0:
            self._workspace.youtube_metadata_path(job_id).unlink(missing_ok=True)
        if start <= 1:
            self._workspace.download_manifest_path(job_id).unlink(missing_ok=True)
            for path in self._workspace.source_dir(job_id).glob("video.*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
        if start <= 2:
            self._workspace.media_inspection_path(job_id).unlink(missing_ok=True)
        if start <= 3:
            self._workspace.audio_manifest_path(job_id).unlink(missing_ok=True)
            self._workspace.audio_path(job_id).unlink(missing_ok=True)
            self._workspace.audio_temp_path(job_id).unlink(missing_ok=True)
        self._workspace.ingestion_path(job_id).unlink(missing_ok=True)

    def cleanup_partial_artifacts(self, job_id: str) -> tuple[int, int]:
        workspace = self._workspace.workspace(job_id)
        patterns = ("*.part", "*.ytdl", "*.tmp", "*.tmp.wav")
        deleted = 0
        reclaimed = 0
        seen: set[Path] = set()
        for pattern in patterns:
            for path in workspace.rglob(pattern):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                try:
                    reclaimed += path.stat().st_size
                    path.unlink()
                    deleted += 1
                except OSError:
                    continue
        return deleted, reclaimed


class CleanupManager:
    def __init__(self, workspace: WorkspaceManager, jobs: JobService, retention_hours: int) -> None:
        self._workspace = workspace
        self._jobs = jobs
        self._retention_hours = retention_hours

    def cleanup(self, *, dry_run: bool = False) -> CleanupResult:
        jobs = self._jobs.list_jobs()
        cutoff = datetime.now(UTC) - timedelta(hours=self._retention_hours)
        result = CleanupResult(jobs_scanned=len(jobs))
        for job in jobs:
            if job.status not in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                continue
            terminal_time = job.completed_at or job.updated_at
            if terminal_time >= cutoff:
                continue
            size = self._workspace.workspace_size(job.id)
            result.job_ids.append(job.id)
            result.jobs_deleted += 1
            result.bytes_reclaimed += size
            if not dry_run:
                self._workspace.delete_workspace(job.id)
        return result

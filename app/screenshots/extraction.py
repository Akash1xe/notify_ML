from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from app.core.config import AppSettings
from app.core.exceptions import (
    JobCancelledError,
    ScreenshotDecodeError,
    ScreenshotFFmpegError,
    ScreenshotTimestampOutOfRangeError,
    SourceVideoInvalidError,
    SourceVideoMissingError,
)
from app.ingestion.models import IngestionResult
from app.jobs.checkpoints import CheckpointStore
from app.media.models import MediaInspection
from app.media.probe import fingerprint as file_fingerprint
from app.screenshots.models import (
    SOURCE_SCREENSHOT_EXTRACTION_ALGORITHM_VERSION,
    ExtractionManifest,
    ExtractionStats,
    ScreenshotExtractionStatus,
    SourceScreenshotRecord,
)
from app.screenshots.repository import ScreenshotRepository
from app.screenshots.utils import canonical_fingerprint, decode_rgb, file_sha256
from app.semantic_analysis.models import Phase7CandidateHandoff
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager

CP_SOURCE_SCREENSHOTS_EXTRACTED = 'SOURCE_SCREENSHOTS_EXTRACTED'


@dataclass(frozen=True)
class ExtractedFrameInfo:
    width: int
    height: int
    resolved_timestamp_seconds: float | None = None


class FrameExtractionBackend(Protocol):
    def extract(self, *, source: Path, timestamp_seconds: float, output: Path, timeout_seconds: float, cancel_check: Callable[[], bool] | None = None) -> ExtractedFrameInfo: ...


class FFmpegFrameExtractionBackend:
    def __init__(self, ffmpeg_binary: str = 'ffmpeg') -> None:
        self._ffmpeg = ffmpeg_binary

    def extract(self, *, source: Path, timestamp_seconds: float, output: Path, timeout_seconds: float, cancel_check: Callable[[], bool] | None = None) -> ExtractedFrameInfo:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        cmd = [
            self._ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(source), '-ss', f'{timestamp_seconds:.6f}',
            '-frames:v', '1', '-vsync', '0', str(output),
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as exc:
            raise ScreenshotFFmpegError('Unable to start FFmpeg for screenshot extraction.') from exc
        started = time.monotonic()
        try:
            while proc.poll() is None:
                if cancel_check and cancel_check():
                    proc.terminate()
                    try: proc.wait(timeout=2)
                    except subprocess.TimeoutExpired: proc.kill()
                    raise JobCancelledError('Screenshot extraction cancelled.')
                if time.monotonic() - started > timeout_seconds:
                    proc.kill()
                    raise ScreenshotFFmpegError('Screenshot extraction timed out.')
                time.sleep(0.03)
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise ScreenshotFFmpegError(f'FFmpeg screenshot extraction failed: {(stderr or "").strip()[:300]}')
            image = decode_rgb(output)
            return ExtractedFrameInfo(width=image.width, height=image.height)
        finally:
            if proc.poll() is None:
                proc.kill()


class SourceScreenshotExtractor:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager, checkpoints: CheckpointStore, repository: ScreenshotRepository, semantic_repository: SemanticRepository, backend: FrameExtractionBackend | None = None) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = repository
        self._semantic = semantic_repository
        self._backend = backend or FFmpegFrameExtractionBackend(str(settings.ffmpeg_path) if settings.ffmpeg_path else "ffmpeg")
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_screenshot_extractions)

    def _load_ingestion(self, job_id: str) -> IngestionResult:
        try:
            return IngestionResult.model_validate(json.loads(self._workspace.ingestion_path(job_id).read_text(encoding='utf-8')))
        except Exception as exc:
            raise SourceVideoMissingError('Phase-2 ingestion result is unavailable.') from exc

    def _load_media(self, job_id: str) -> MediaInspection:
        try:
            return MediaInspection.model_validate(json.loads(self._workspace.media_inspection_path(job_id).read_text(encoding='utf-8')))
        except Exception as exc:
            raise SourceVideoInvalidError('Phase-2 media inspection is unavailable.') from exc

    def _source(self, job_id: str) -> tuple[Path, str, float]:
        ingestion = self._load_ingestion(job_id)
        media = self._load_media(job_id)
        root = self._workspace.workspace(job_id)
        source = (root / ingestion.source_video).resolve()
        try: source.relative_to(root)
        except ValueError as exc: raise SourceVideoInvalidError('Source video path escaped job workspace.') from exc
        if not source.exists() or source.stat().st_size <= 0:
            raise SourceVideoMissingError('Source video is missing.')
        duration = ingestion.duration_seconds or media.duration_seconds
        if duration is None or duration <= 0:
            raise SourceVideoInvalidError('Source video duration is unavailable.')
        current_source_fingerprint = file_fingerprint(source)
        if current_source_fingerprint != media.source_fingerprint:
            raise SourceVideoInvalidError('Source video fingerprint no longer matches Phase-2 media inspection.')
        source_fingerprint = canonical_fingerprint(media.source_fingerprint)
        return source, source_fingerprint, float(duration)

    def config_fingerprint(self) -> str:
        return canonical_fingerprint({
            'algorithm': SOURCE_SCREENSHOT_EXTRACTION_ALGORITHM_VERSION,
            'format': self._settings.final_screenshot_format,
            'end_tolerance': self._settings.source_screenshot_end_tolerance_seconds,
            'min_dimension': self._settings.source_screenshot_min_dimension,
        })

    def expected_fingerprint(self, handoff: Phase7CandidateHandoff, source_fingerprint: str) -> str:
        return canonical_fingerprint({
            'candidate_id': handoff.candidate_id,
            'timestamp': round(handoff.candidate_timestamp_seconds, 6),
            'source': source_fingerprint,
            'config': self.config_fingerprint(),
        })

    async def extract_candidate(self, job_id: str, handoff: Phase7CandidateHandoff, *, cancel_check: Callable[[], bool] | None = None) -> SourceScreenshotRecord:
        source, source_fp, duration = self._source(job_id)
        timestamp = float(handoff.candidate_timestamp_seconds)
        tolerance = self._settings.source_screenshot_end_tolerance_seconds
        if timestamp < 0 or timestamp > duration + tolerance:
            raise ScreenshotTimestampOutOfRangeError('Semantic candidate timestamp is outside source video duration.')
        timestamp = min(timestamp, max(0.0, duration - 1e-6))
        final = self._workspace.screenshot_extraction_path(job_id, handoff.candidate_id)
        tmp = final.with_name(final.stem + '.tmp' + final.suffix)
        tmp.unlink(missing_ok=True)
        async with self._semaphore:
            if cancel_check and cancel_check():
                raise JobCancelledError('Screenshot extraction cancelled.')
            try:
                info = await asyncio.to_thread(
                    self._backend.extract,
                    source=source,
                    timestamp_seconds=timestamp,
                    output=tmp,
                    timeout_seconds=self._settings.source_screenshot_timeout_seconds,
                    cancel_check=cancel_check,
                )
                image = decode_rgb(tmp)
                if min(image.width, image.height) < self._settings.source_screenshot_min_dimension:
                    raise ScreenshotDecodeError('Extracted screenshot dimensions are invalid.')
                os.replace(tmp, final)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        rel_source = self._workspace.relative_to_workspace(job_id, source)
        rel_final = self._workspace.relative_to_workspace(job_id, final)
        extraction_fp = self.expected_fingerprint(handoff, source_fp)
        record = SourceScreenshotRecord(
            candidate_id=handoff.candidate_id,
            stable_window_id=handoff.stable_window_id,
            requested_timestamp_seconds=handoff.candidate_timestamp_seconds,
            resolved_timestamp_seconds=info.resolved_timestamp_seconds,
            source_video_relative_path=rel_source,
            screenshot_relative_path=rel_final,
            width=info.width,
            height=info.height,
            format='png',
            file_size_bytes=final.stat().st_size,
            file_sha256=file_sha256(final),
            source_fingerprint=source_fp,
            extraction_config_fingerprint=self.config_fingerprint(),
            extraction_fingerprint=extraction_fp,
            extraction_status=ScreenshotExtractionStatus.SUCCESS,
        )
        self._repository.save_extraction_record(job_id, record)
        return record

    def build_manifest(self, job_id: str, handoff: list[Phase7CandidateHandoff], records: list[SourceScreenshotRecord], *, cached_count: int = 0) -> ExtractionManifest:
        records = sorted(records, key=lambda r: (r.requested_timestamp_seconds, r.candidate_id))
        source_fp = records[0].source_fingerprint if records else self._source(job_id)[1]
        widths=[r.width for r in records]; heights=[r.height for r in records]
        stats = ExtractionStats(
            selected_candidate_count=len(handoff), successful_extraction_count=len(records),
            failed_extraction_count=max(0, len(handoff)-len(records)), cached_extraction_count=cached_count,
            total_output_bytes=sum(r.file_size_bytes for r in records),
            min_width=min(widths) if widths else None, max_width=max(widths) if widths else None,
            min_height=min(heights) if heights else None, max_height=max(heights) if heights else None,
            unique_resolutions=sorted({f'{r.width}x{r.height}' for r in records}),
            first_selected_timestamp=records[0].requested_timestamp_seconds if records else None,
            last_selected_timestamp=records[-1].requested_timestamp_seconds if records else None,
            warnings=['SOURCE_VIDEO_RESOLUTION_VARIATION'] if len({(r.width,r.height) for r in records}) > 1 else [],
        )
        dependency = canonical_fingerprint([(x.candidate_id, round(x.candidate_timestamp_seconds,6)) for x in handoff])
        payload = {'algorithm': SOURCE_SCREENSHOT_EXTRACTION_ALGORITHM_VERSION, 'semantic':dependency, 'source':source_fp, 'config':self.config_fingerprint(), 'records':[r.extraction_fingerprint for r in records]}
        manifest = ExtractionManifest(semantic_selections_fingerprint=dependency, source_fingerprint=source_fp, config_fingerprint=self.config_fingerprint(), artifact_fingerprint=canonical_fingerprint(payload), stats=stats, screenshots=records)
        self._repository.save_extraction_manifest(job_id, manifest)
        self._checkpoints.mark_completed(job_id, CP_SOURCE_SCREENSHOTS_EXTRACTED)
        return manifest

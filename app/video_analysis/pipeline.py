from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from app.core.config import AppSettings
from app.core.exceptions import CacheValidationError, FrameAnalysisValidationError, JobCancelledError
from app.core.logging import JobEventLogger
from app.ingestion.cache import ArtifactState, CacheManager
from app.ingestion.models import IngestionResult
from app.ingestion.pipeline import IngestionPipeline
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService
from app.media.models import MediaInspection
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.cache import (
    CP_FRAME_ANALYSIS,
    CP_FRAME_DIFFERENCES,
    CP_FRAMES_PREPROCESSED,
    CP_FRAMES_SAMPLED,
    CP_MAJOR_CHANGES,
    CP_TIMELINE,
    FrameAnalysisCacheManager,
)
from app.video_analysis.changes import MajorChangeDetector
from app.video_analysis.differences import VisualDifferenceService
from app.video_analysis.models import (
    FrameAnalysisSummary,
    Phase3DiskUsage,
    Phase3ResumeStage,
    Phase3Timings,
)
from app.video_analysis.preprocessing import FramePreprocessor
from app.video_analysis.repository import FrameAnalysisRepository
from app.video_analysis.sampling import FrameSampler
from app.video_analysis.timeline import TemporalTimelineService


class FrameAnalysisPipeline:
    def __init__(
        self,
        *,
        settings: AppSettings,
        jobs: JobService,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        events: JobEventLogger,
        sampler: FrameSampler,
        preprocessor: FramePreprocessor,
        differences: VisualDifferenceService,
        major_changes: MajorChangeDetector,
        timeline: TemporalTimelineService,
        cache: FrameAnalysisCacheManager,
        repository: FrameAnalysisRepository,
    ) -> None:
        self._settings = settings
        self._jobs = jobs
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._events = events
        self._sampler = sampler
        self._preprocessor = preprocessor
        self._differences = differences
        self._major_changes = major_changes
        self._timeline = timeline
        self._cache = cache
        self._repository = repository
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

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
        self._guard_cancelled(job_id)
        self._jobs.update_progress(job_id, value, message)

    def _mapped_progress(self, job_id: str, start: int, span: int, percent: float, message: str) -> None:
        if self._cancelled(job_id):
            return
        value = start + int(min(100.0, max(0.0, percent)) * span / 100.0)
        try:
            self._jobs.update_progress(job_id, value, message)
        except Exception:
            if not self._cancelled(job_id):
                raise

    def _resolve_source(self, job_id: str, ingestion: IngestionResult) -> Path:
        root = self._workspace.workspace(job_id)
        path = (root / ingestion.source_video).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise FrameAnalysisValidationError("Phase-2 source video path escaped the job workspace.") from exc
        if not path.exists() or path.stat().st_size <= 0:
            raise FrameAnalysisValidationError("Phase-2 source video is missing.")
        return path

    def _load_media(self, job_id: str) -> MediaInspection:
        path = self._workspace.media_inspection_path(job_id)
        try:
            return MediaInspection.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise FrameAnalysisValidationError("Phase-2 media inspection is unavailable or invalid.") from exc

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
        return total

    async def process(
        self,
        job_id: str,
        *,
        ingestion: IngestionResult | None = None,
        finalize_job: bool = True,
    ) -> FrameAnalysisSummary:
        async with self._locks[job_id]:
            return await self._process_locked(job_id, ingestion=ingestion, finalize_job=finalize_job)

    async def _process_locked(
        self,
        job_id: str,
        *,
        ingestion: IngestionResult | None,
        finalize_job: bool,
    ) -> FrameAnalysisSummary:
        started = time.monotonic()
        self._cache.cleanup_partial_artifacts(job_id)
        ingestion = ingestion or self._load_ingestion(job_id)
        source_path = self._resolve_source(job_id, ingestion)
        media = self._load_media(job_id)
        snapshot = self._cache.reconcile(job_id, source_path)
        self._events.write(
            job_id,
            level="INFO",
            stage="FRAME_ANALYSIS",
            message=f"Phase-3 resume stage selected: {snapshot.resume_stage.value}",
        )

        # ----- Sampling -----
        sampling_check, sampling = self._cache.validate_sampling(job_id, source_path)
        if not sampling_check.valid or sampling is None:
            self._cache.invalidate_from(job_id, Phase3ResumeStage.SAMPLING_FRAMES)
            self._stage(job_id, JobStage.SAMPLING_FRAMES, "Sampling lecture frames")
            self._progress(job_id, 50, "Preparing frame sampling")
            sampling = await self._sampler.sample(
                job_id=job_id,
                source_path=source_path,
                source_media=media,
                progress_callback=lambda percent: self._mapped_progress(
                    job_id, 50, 10, percent, "Sampling lecture frames"
                ),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_FRAMES_SAMPLED)
            self._progress(job_id, 60, "Frame sampling complete")
        else:
            self._events.write(job_id, level="INFO", stage="SAMPLING_FRAMES", message="Frame sampling cache hit")
            self._progress(job_id, 60, "Frame sampling cache reused")
        self._guard_cancelled(job_id)

        # ----- Preprocessing -----
        preprocessing_check, preprocessing = self._cache.validate_preprocessing(job_id, sampling)
        if not preprocessing_check.valid or preprocessing is None:
            self._cache.invalidate_from(job_id, Phase3ResumeStage.PREPROCESSING_FRAMES)
            self._stage(job_id, JobStage.PREPROCESSING_FRAMES, "Analyzing frame quality")
            preprocessing = await asyncio.to_thread(
                self._preprocessor.process,
                job_id=job_id,
                sampling=sampling,
                progress_callback=lambda percent: self._mapped_progress(
                    job_id, 60, 7, percent, "Analyzing frame quality"
                ),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_FRAMES_PREPROCESSED)
            self._progress(job_id, 67, "Frame preprocessing complete")
        else:
            self._events.write(job_id, level="INFO", stage="PREPROCESSING_FRAMES", message="Frame preprocessing cache hit")
            self._progress(job_id, 67, "Frame preprocessing cache reused")
        self._guard_cancelled(job_id)

        # ----- Difference scoring -----
        differences_check, differences = self._cache.validate_differences(job_id, preprocessing)
        if not differences_check.valid or differences is None:
            self._cache.invalidate_from(job_id, Phase3ResumeStage.DETECTING_CHANGES)
            self._stage(job_id, JobStage.DETECTING_CHANGES, "Measuring visual changes")
            differences = await asyncio.to_thread(
                self._differences.process,
                job_id=job_id,
                preprocessing=preprocessing,
                progress_callback=lambda percent: self._mapped_progress(
                    job_id, 67, 7, percent, "Measuring visual changes"
                ),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_FRAME_DIFFERENCES)
            self._progress(job_id, 74, "Visual difference scoring complete")
        else:
            self._events.write(job_id, level="INFO", stage="DETECTING_CHANGES", message="Visual difference cache hit")
            self._progress(job_id, 74, "Visual difference cache reused")
        self._guard_cancelled(job_id)

        # ----- Major changes -----
        major_check, major = self._cache.validate_major_changes(job_id, differences)
        if not major_check.valid or major is None:
            self._cache.invalidate_from(job_id, Phase3ResumeStage.DETECTING_MAJOR_CHANGES)
            self._stage(job_id, JobStage.DETECTING_MAJOR_CHANGES, "Detecting major visual transitions")
            major = await asyncio.to_thread(
                self._major_changes.process,
                job_id=job_id,
                differences=differences,
                progress_callback=lambda percent: self._mapped_progress(
                    job_id, 74, 5, percent, "Detecting major visual transitions"
                ),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_MAJOR_CHANGES)
            self._progress(job_id, 79, "Major change detection complete")
        else:
            self._events.write(job_id, level="INFO", stage="DETECTING_MAJOR_CHANGES", message="Major change cache hit")
            self._progress(job_id, 79, "Major change cache reused")
        if major.stats.event_ratio > self._settings.max_major_event_ratio_warning:
            self._events.write(
                job_id,
                level="WARNING",
                stage="DETECTING_MAJOR_CHANGES",
                message="Major visual event density is unusually high",
            )
        self._guard_cancelled(job_id)

        # ----- Temporal timeline -----
        timeline_check, timeline = self._cache.validate_timeline(job_id, differences, major)
        if not timeline_check.valid or timeline is None:
            self._cache.invalidate_from(job_id, Phase3ResumeStage.DETECTING_STABILITY)
            self._stage(job_id, JobStage.DETECTING_STABILITY, "Building visual activity timeline")
            timeline = await asyncio.to_thread(
                self._timeline.process,
                job_id=job_id,
                differences=differences,
                major_changes=major,
                progress_callback=lambda percent: self._mapped_progress(
                    job_id, 79, 5, percent, "Building visual activity timeline"
                ),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_TIMELINE)
            self._progress(job_id, 84, "Visual timeline complete")
        else:
            self._events.write(job_id, level="INFO", stage="DETECTING_STABILITY", message="Timeline cache hit")
            self._progress(job_id, 84, "Visual timeline cache reused")
        self._guard_cancelled(job_id)

        # Final validation reuses the same cache validators used for recovery.
        final_snapshot = self._cache.inspect(job_id, source_path)
        if not all(
            getattr(final_snapshot, field).valid
            for field in ("sampling", "preprocessing", "differences", "major_changes", "timeline")
        ):
            raise FrameAnalysisValidationError("Phase-3 artifacts failed final consistency validation.")

        analysis_start = sampling.frames[0].timestamp_seconds if sampling.frames else 0.0
        analysis_end = sampling.frames[-1].timestamp_seconds if sampling.frames else analysis_start
        analysis_dir = self._workspace.analysis_dir(job_id)
        disk = Phase3DiskUsage(
            sampled_frames_bytes=self._dir_size(self._workspace.sampled_frames_dir(job_id)),
            processed_frames_bytes=self._dir_size(self._workspace.processed_frames_dir(job_id)),
            analysis_metadata_bytes=self._dir_size(analysis_dir),
        )
        disk.phase3_total_bytes = (
            disk.sampled_frames_bytes + disk.processed_frames_bytes + disk.analysis_metadata_bytes
        )
        timings = Phase3Timings(
            sampling_seconds=sampling.sampling_seconds,
            preprocessing_seconds=preprocessing.preprocessing_seconds,
            difference_seconds=differences.comparison_seconds,
            major_change_seconds=major.detection_seconds,
            timeline_seconds=timeline.timeline_seconds,
            total_phase3_seconds=round(time.monotonic() - started, 3),
        )
        summary = FrameAnalysisSummary(
            sampled_frame_count=sampling.actual_frame_count,
            processed_frame_count=preprocessing.stats.valid_frames,
            invalid_frame_count=preprocessing.stats.corrupt_frames,
            difference_count=differences.stats.total_comparisons,
            major_event_count=major.stats.major_events + major.stats.very_major_events,
            stable_segment_count=timeline.stats.stable_segment_count,
            changing_segment_count=timeline.stats.changing_segment_count,
            analysis_start_seconds=analysis_start,
            analysis_end_seconds=analysis_end,
            timings=timings,
            disk_usage=disk,
            sampling_fingerprint=sampling.artifact_fingerprint,
            preprocessing_fingerprint=preprocessing.artifact_fingerprint,
            differences_fingerprint=differences.artifact_fingerprint,
            major_changes_fingerprint=major.artifact_fingerprint,
            timeline_fingerprint=timeline.artifact_fingerprint,
        )
        atomic_write_json(
            self._workspace.analysis_summary_path(job_id),
            summary.model_dump(mode="json"),
        )
        self._checkpoints.mark_completed(job_id, CP_FRAME_ANALYSIS)
        self._stage(job_id, JobStage.FRAME_ANALYSIS_COMPLETE, "Frame analysis complete")
        self._progress(job_id, 86, "Frame analysis complete")
        completed = self._cache.inspect(job_id, source_path)
        if not completed.frame_analysis_complete.valid:
            raise FrameAnalysisValidationError("FRAME_ANALYSIS_COMPLETE failed validation after finalization.")
        self._events.write(job_id, level="INFO", stage="FRAME_ANALYSIS_COMPLETE", message="Phase 3 frame analysis complete")
        if finalize_job:
            self._jobs.mark_completed(
                job_id,
                message="Frame analysis completed; ready for stability candidate generation",
            )
        return summary

    def _load_ingestion(self, job_id: str) -> IngestionResult:
        path = self._workspace.ingestion_path(job_id)
        try:
            return IngestionResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise FrameAnalysisValidationError("Phase-2 ingestion result is unavailable or invalid.") from exc

    def load_summary(self, job_id: str) -> FrameAnalysisSummary | None:
        try:
            return self._repository.load_summary(job_id)
        except Exception:
            return None


class NotifyPipeline:
    """Current application processor: Phase 2 ingestion followed by Phase 3 visual analysis."""

    def __init__(
        self,
        *,
        ingestion: IngestionPipeline,
        frame_analysis: FrameAnalysisPipeline,
        ingestion_cache: CacheManager,
        jobs: JobService,
        candidate_analysis=None,
        transcription=None,
        semantic=None,
        screenshots=None,
        document=None,
    ) -> None:
        self._ingestion = ingestion
        self._frame_analysis = frame_analysis
        self._ingestion_cache = ingestion_cache
        self._jobs = jobs
        self._candidate_analysis = candidate_analysis
        self._transcription = transcription
        self._semantic = semantic
        self._screenshots = screenshots
        self._document = document

    async def process(self, job_id: str) -> None:
        ingestion = await self._ingestion.process(job_id, finalize_job=False)
        current = self._jobs.get_job(job_id)
        if current.status is JobStatus.CANCELLED:
            raise JobCancelledError(f"Job {job_id} was cancelled")
        await self._frame_analysis.process(
            job_id,
            ingestion=ingestion,
            finalize_job=self._candidate_analysis is None and self._transcription is None and self._semantic is None and self._screenshots is None and self._document is None,
        )
        if self._candidate_analysis is not None:
            current = self._jobs.get_job(job_id)
            if current.status is JobStatus.CANCELLED:
                raise JobCancelledError(f"Job {job_id} was cancelled")
            await self._candidate_analysis.process(job_id, finalize_job=self._transcription is None and self._semantic is None and self._screenshots is None and self._document is None)
        if self._transcription is not None:
            current = self._jobs.get_job(job_id)
            if current.status is JobStatus.CANCELLED:
                raise JobCancelledError(f"Job {job_id} was cancelled")
            await self._transcription.process(job_id, finalize_job=self._semantic is None and self._screenshots is None and self._document is None)
        if self._semantic is not None:
            current = self._jobs.get_job(job_id)
            if current.status is JobStatus.CANCELLED:
                raise JobCancelledError(f"Job {job_id} was cancelled")
            await self._semantic.process(job_id, finalize_job=self._screenshots is None and self._document is None)
        if self._screenshots is not None:
            current = self._jobs.get_job(job_id)
            if current.status is JobStatus.CANCELLED:
                raise JobCancelledError(f"Job {job_id} was cancelled")
            await self._screenshots.process(job_id, finalize_job=self._document is None)
        if self._document is not None:
            current = self._jobs.get_job(job_id)
            if current.status is JobStatus.CANCELLED:
                raise JobCancelledError(f"Job {job_id} was cancelled")
            await self._document.process(job_id, finalize_job=True)

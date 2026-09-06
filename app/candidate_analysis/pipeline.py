from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict

from pydantic import ValidationError

from app.candidate_analysis.boundaries import StableBoundaryDetector
from app.candidate_analysis.cache import (
    CP_BOUNDARIES,
    CP_CANDIDATE_HEURISTICS,
    CP_CANDIDATES_GENERATED,
    CP_CANDIDATES_RANKED,
    CP_CANDIDATES_READY,
    CP_STABILITY_WINDOWS,
    CandidateAnalysisCacheManager,
)
from app.candidate_analysis.generation import CandidateGenerator
from app.candidate_analysis.heuristics import CandidateHeuristicAnalyzer
from app.candidate_analysis.models import CandidateAnalysisSummary, CandidateResumeStage, Phase4Timings
from app.candidate_analysis.ranking import CandidateRankingService
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.candidate_analysis.stability import StabilityWindowDetector
from app.core.config import AppSettings
from app.core.exceptions import CandidateAnalysisValidationError, JobCancelledError
from app.core.logging import JobEventLogger
from app.ingestion.models import IngestionResult
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.cache import FrameAnalysisCacheManager
from app.video_analysis.repository import FrameAnalysisRepository


class CandidateAnalysisPipeline:
    def __init__(
        self,
        *,
        settings: AppSettings,
        jobs: JobService,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        events: JobEventLogger,
        frame_cache: FrameAnalysisCacheManager,
        frame_repository: FrameAnalysisRepository,
        stability: StabilityWindowDetector,
        boundaries: StableBoundaryDetector,
        generator: CandidateGenerator,
        heuristics: CandidateHeuristicAnalyzer,
        ranking: CandidateRankingService,
        cache: CandidateAnalysisCacheManager,
        repository: CandidateAnalysisRepository,
    ) -> None:
        self._settings = settings
        self._jobs = jobs
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._events = events
        self._frame_cache = frame_cache
        self._frame_repository = frame_repository
        self._stability = stability
        self._boundaries = boundaries
        self._generator = generator
        self._heuristics = heuristics
        self._ranking = ranking
        self._cache = cache
        self._repository = repository
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _cancelled(self, job_id: str) -> bool:
        return self._jobs.get_job(job_id).status is JobStatus.CANCELLED

    def _guard_cancelled(self, job_id: str) -> None:
        if self._cancelled(job_id):
            raise JobCancelledError(f"Job {job_id} was cancelled")

    def _stage(self, job_id: str, message: str) -> None:
        self._guard_cancelled(job_id)
        self._jobs.update_stage(job_id, JobStage.GENERATING_CANDIDATES, message)
        self._events.write(job_id, level="INFO", stage=JobStage.GENERATING_CANDIDATES.value, message=message)

    def _progress(self, job_id: str, value: int, message: str) -> None:
        self._guard_cancelled(job_id)
        self._jobs.update_progress(job_id, value, message)

    def _mapped_progress(self, job_id: str, start: int, width: int, percent: float, message: str) -> None:
        value = start + int(round(width * max(0.0, min(100.0, percent)) / 100.0))
        self._progress(job_id, min(start + width, value), message)

    def _phase3_inputs(self, job_id: str):
        ingestion_path = self._workspace.ingestion_path(job_id)
        try:
            ingestion = IngestionResult.model_validate(json.loads(ingestion_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise CandidateAnalysisValidationError("Phase-2 ingestion result is unavailable or invalid.") from exc
        root = self._workspace.workspace(job_id)
        source = (root / ingestion.source_video).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise CandidateAnalysisValidationError("Phase-2 source path escaped the job workspace.") from exc
        phase3_snapshot = self._frame_cache.inspect(job_id, source)
        if not phase3_snapshot.frame_analysis_complete.valid:
            raise CandidateAnalysisValidationError("FRAME_ANALYSIS_COMPLETE is unavailable or stale.")
        return (
            self._frame_repository.load_sampling(job_id),
            self._frame_repository.load_preprocessing(job_id),
            self._frame_repository.load_differences(job_id),
            self._frame_repository.load_major_changes(job_id),
            self._frame_repository.load_timeline(job_id),
        )

    async def process(self, job_id: str, *, finalize_job: bool = True) -> CandidateAnalysisSummary:
        async with self._locks[job_id]:
            return await self._process_locked(job_id, finalize_job=finalize_job)

    async def _process_locked(self, job_id: str, *, finalize_job: bool) -> CandidateAnalysisSummary:
        started = time.monotonic()
        self._guard_cancelled(job_id)
        sampling, preprocessing, differences, major_changes, timeline = self._phase3_inputs(job_id)
        phase3 = dict(
            sampling=sampling,
            preprocessing=preprocessing,
            differences=differences,
            major_changes=major_changes,
            timeline=timeline,
        )
        removed = self._cache.cleanup_partial_artifacts(job_id)
        if removed:
            self._events.write(job_id, level="INFO", stage="GENERATING_CANDIDATES", message=f"Removed {len(removed)} partial Phase-4 artifacts")
        snapshot = self._cache.reconcile(job_id, **phase3)
        self._events.write(job_id, level="INFO", stage="GENERATING_CANDIDATES", message=f"Phase-4 resume stage: {snapshot.resume_stage.value}")

        windows_check, windows = self._cache.validate_stability_windows(job_id, timeline, differences, preprocessing, major_changes)
        if not windows_check.valid or windows is None:
            self._cache.invalidate_from(job_id, CandidateResumeStage.STABILITY_WINDOWS)
            self._stage(job_id, "Detecting stable visual windows")
            windows = await asyncio.to_thread(
                self._stability.process,
                job_id=job_id,
                timeline=timeline,
                differences=differences,
                preprocessing=preprocessing,
                major_changes=major_changes,
                progress_callback=lambda p: self._mapped_progress(job_id, 86, 2, p, "Analyzing stable visual periods"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_STABILITY_WINDOWS)
            self._progress(job_id, 88, "Stability windows ready")
        else:
            self._progress(job_id, 88, "Stability-window cache reused")
        self._guard_cancelled(job_id)

        boundary_check, boundaries = self._cache.validate_boundaries(job_id, windows, timeline, differences, major_changes)
        if not boundary_check.valid or boundaries is None:
            self._cache.invalidate_from(job_id, CandidateResumeStage.BOUNDARY_DETECTION)
            self._stage(job_id, "Detecting stable visual boundaries")
            boundaries = await asyncio.to_thread(
                self._boundaries.process,
                job_id=job_id,
                windows=windows,
                timeline=timeline,
                differences=differences,
                major_changes=major_changes,
                progress_callback=lambda p: self._mapped_progress(job_id, 88, 2, p, "Measuring visual activity drops"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_BOUNDARIES)
            self._progress(job_id, 90, "Stable boundaries ready")
        else:
            self._progress(job_id, 90, "Stable-boundary cache reused")
        self._guard_cancelled(job_id)

        generated_check, generated = self._cache.validate_generated_candidates(job_id, windows, boundaries, sampling, preprocessing)
        if not generated_check.valid or generated is None:
            self._cache.invalidate_from(job_id, CandidateResumeStage.CANDIDATE_GENERATION)
            self._stage(job_id, "Generating stable-frame candidates")
            generated = await asyncio.to_thread(
                self._generator.process,
                job_id=job_id,
                windows=windows,
                boundaries=boundaries,
                sampling=sampling,
                preprocessing=preprocessing,
                progress_callback=lambda p: self._mapped_progress(job_id, 90, 2, p, "Mapping candidate timestamps to analysis frames"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_CANDIDATES_GENERATED)
            self._progress(job_id, 92, "Candidate frames generated")
        else:
            self._progress(job_id, 92, "Candidate-generation cache reused")
        self._guard_cancelled(job_id)

        scored_check, scored = self._cache.validate_scored_candidates(job_id, generated, windows, boundaries, preprocessing, differences, timeline)
        if not scored_check.valid or scored is None:
            self._cache.invalidate_from(job_id, CandidateResumeStage.CANDIDATE_HEURISTICS)
            self._stage(job_id, "Scoring candidate visual states")
            scored = await asyncio.to_thread(
                self._heuristics.process,
                job_id=job_id,
                generated=generated,
                windows=windows,
                boundaries=boundaries,
                preprocessing=preprocessing,
                differences=differences,
                timeline=timeline,
                progress_callback=lambda p: self._mapped_progress(job_id, 92, 2, p, "Estimating candidate visual completeness"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_CANDIDATE_HEURISTICS)
            self._progress(job_id, 94, "Candidate heuristics complete")
        else:
            self._progress(job_id, 94, "Candidate-heuristic cache reused")
        self._guard_cancelled(job_id)

        ranking_check, ranked, selections = self._cache.validate_ranking(job_id, scored, windows)
        if not ranking_check.valid or ranked is None or selections is None:
            self._cache.invalidate_from(job_id, CandidateResumeStage.CANDIDATE_RANKING)
            self._stage(job_id, "Ranking candidate visual states")
            ranked, selections = await asyncio.to_thread(
                self._ranking.process,
                job_id=job_id,
                scored=scored,
                windows=windows,
                progress_callback=lambda p: self._mapped_progress(job_id, 94, 2, p, "Selecting window representatives"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            self._checkpoints.mark_completed(job_id, CP_CANDIDATES_RANKED)
            self._progress(job_id, 96, "Candidate ranking complete")
        else:
            self._progress(job_id, 96, "Candidate-ranking cache reused")
        self._guard_cancelled(job_id)

        final_snapshot = self._cache.inspect(job_id, **phase3)
        if not all(
            getattr(final_snapshot, field).valid
            for field in ("stability_windows", "boundaries", "generated_candidates", "scored_candidates", "ranking")
        ):
            raise CandidateAnalysisValidationError("Phase-4 artifacts failed final consistency validation.")

        duration = sampling.video_duration_seconds
        timings = Phase4Timings(
            stability_window_seconds=windows.detection_seconds,
            boundary_detection_seconds=boundaries.detection_seconds,
            candidate_generation_seconds=generated.generation_seconds,
            heuristic_scoring_seconds=scored.scoring_seconds,
            ranking_seconds=ranked.ranking_seconds,
            total_phase4_seconds=round(time.monotonic() - started, 3),
        )
        summary = CandidateAnalysisSummary(
            stability_window_count=windows.stats.total_stable_segments,
            valid_stability_window_count=windows.stats.valid_stability_windows,
            valid_boundary_count=boundaries.stats.valid_boundaries,
            generated_candidate_count=generated.stats.candidates_generated,
            valid_candidate_count=scored.stats.valid_candidates,
            primary_candidate_count=ranked.stats.primary_candidates,
            alternate_candidate_count=ranked.stats.alternate_candidates,
            ambiguous_window_count=ranked.stats.ambiguous_windows,
            clear_winner_count=ranked.stats.clear_winner_windows,
            window_without_candidate_count=ranked.stats.windows_without_candidate,
            primary_candidates_per_minute=(ranked.stats.primary_candidates * 60.0 / duration if duration > 0 else 0.0),
            generated_candidates_per_minute=(generated.stats.valid_candidates * 60.0 / duration if duration > 0 else 0.0),
            timings=timings,
            stability_windows_fingerprint=windows.artifact_fingerprint,
            boundaries_fingerprint=boundaries.artifact_fingerprint,
            generated_candidates_fingerprint=generated.artifact_fingerprint,
            scored_candidates_fingerprint=scored.artifact_fingerprint,
            ranked_candidates_fingerprint=ranked.artifact_fingerprint,
            selections_fingerprint=selections.artifact_fingerprint,
        )
        atomic_write_json(self._workspace.candidate_summary_path(job_id), summary.model_dump(mode="json"))
        self._checkpoints.mark_completed(job_id, CP_CANDIDATES_READY)
        self._progress(job_id, 97, "Visual candidate analysis complete")
        completed = self._cache.inspect(job_id, **phase3)
        if not completed.candidates_ready.valid:
            raise CandidateAnalysisValidationError("CANDIDATES_READY failed validation after finalization.")
        self._events.write(job_id, level="INFO", stage="GENERATING_CANDIDATES", message="Phase 4 visual candidate analysis complete")
        if finalize_job:
            self._jobs.mark_completed(
                job_id,
                message="Visual candidates ready for transcript and semantic analysis",
            )
        return summary

    def load_summary(self, job_id: str) -> CandidateAnalysisSummary | None:
        try:
            return self._repository.load_summary(job_id)
        except Exception:
            return None

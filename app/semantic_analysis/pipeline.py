from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from app.core.exceptions import JobCancelledError, Phase6ValidationError
from app.core.logging import JobEventLogger
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService
from app.semantic_analysis.analysis import CP_SEMANTIC_ANALYSIS_COMPLETE, SemanticCandidateAnalyzer
from app.semantic_analysis.cache import SemanticCacheCoordinator
from app.semantic_analysis.decision import CP_SEMANTIC_CANDIDATES_READY, SemanticDecisionEngine
from app.semantic_analysis.models import Phase6ResumeStage, Phase6Timings, SemanticSummary
from app.semantic_analysis.preparation import CP_SEMANTIC_INPUT_READY, SemanticInputPreparationService
from app.semantic_analysis.repository import SemanticRepository
from app.semantic_analysis.temporal_context import CP_TEMPORAL_VISUAL_CONTEXT_READY, TemporalVisualContextService


class SemanticPipeline:
    def __init__(
        self,
        *,
        jobs: JobService,
        checkpoints: CheckpointStore,
        events: JobEventLogger,
        preparation: SemanticInputPreparationService,
        temporal_context: TemporalVisualContextService,
        analyzer: SemanticCandidateAnalyzer,
        decision: SemanticDecisionEngine,
        cache: SemanticCacheCoordinator,
        repository: SemanticRepository,
    ) -> None:
        self._jobs = jobs
        self._checkpoints = checkpoints
        self._events = events
        self._preparation = preparation
        self._temporal = temporal_context
        self._analyzer = analyzer
        self._decision = decision
        self._cache = cache
        self._repository = repository
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _cancelled(self, job_id: str) -> bool:
        return self._jobs.get_job(job_id).status is JobStatus.CANCELLED

    def _guard(self, job_id: str) -> None:
        if self._cancelled(job_id):
            raise JobCancelledError(f"Job {job_id} was cancelled")

    def _stage(self, job_id: str, stage: JobStage, message: str) -> None:
        self._guard(job_id)
        self._jobs.update_stage(job_id, stage, message)
        self._events.write(job_id, level="INFO", stage=stage.value, message=message)

    def _progress(self, job_id: str, value: int, message: str) -> None:
        if self._cancelled(job_id):
            return
        current = self._jobs.get_job(job_id)
        # Existing Phase-5 progress already reaches 99; never move backwards.
        self._jobs.update_progress(job_id, max(current.progress, value), message)

    async def process(self, job_id: str, *, finalize_job: bool = True) -> SemanticSummary:
        async with self._locks[job_id]:
            return await self._process_locked(job_id, finalize_job=finalize_job)

    async def _process_locked(self, job_id: str, *, finalize_job: bool) -> SemanticSummary:
        if not self._checkpoints.is_completed(job_id, "TRANSCRIPT_CONTEXT_READY"):
            raise Phase6ValidationError("TRANSCRIPT_CONTEXT_READY is required before Phase 6.")
        total_started = time.monotonic()
        removed = self._cache.cleanup_partial_artifacts(job_id)
        initial = self._cache.reconcile(job_id)
        self._events.write(
            job_id,
            level="INFO",
            stage="SEMANTIC_ANALYSIS",
            message=f"Phase-6 resume stage: {initial.resume_plan.resume_stage.value}; removed {len(removed)} temporary artifacts",
        )
        timings = {"input": 0.0, "context": 0.0, "analysis": 0.0, "decision": 0.0}

        input_check, semantic_input = self._cache.validate_semantic_input(job_id)
        if not input_check.valid or semantic_input is None:
            self._cache.invalidate_from(job_id, Phase6ResumeStage.SEMANTIC_INPUT_PREPARATION)
            self._stage(job_id, JobStage.PREPARING_SEMANTIC_INPUT, "Preparing visual candidates for semantic analysis")
            started = time.monotonic()
            semantic_input = await asyncio.to_thread(
                self._preparation.process,
                job_id,
                progress_callback=lambda p: self._progress(job_id, 99, "Preparing semantic input"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings["input"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_SEMANTIC_INPUT_READY)
        self._guard(job_id)

        temporal_check, temporal = self._cache.validate_temporal_context(job_id, semantic_input)
        if not temporal_check.valid or temporal is None:
            self._cache.invalidate_from(job_id, Phase6ResumeStage.TEMPORAL_CONTEXT_CONSTRUCTION)
            self._stage(job_id, JobStage.BUILDING_VISUAL_CONTEXT, "Building temporal visual context")
            started = time.monotonic()
            temporal = await asyncio.to_thread(
                self._temporal.process,
                job_id,
                progress_callback=lambda p: self._progress(job_id, 99, "Building temporal visual context"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings["context"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_TEMPORAL_VISUAL_CONTEXT_READY)
        self._guard(job_id)

        analysis_check, semantic_results, candidate_checks = self._cache.validate_semantic_analysis(job_id, semantic_input, temporal)
        if not analysis_check.valid or semantic_results is None:
            self._cache.invalidate_from(job_id, Phase6ResumeStage.SEMANTIC_ANALYSIS)
            self._stage(job_id, JobStage.ANALYZING_VISUAL_CANDIDATES, "Analyzing visual candidates with local vision model")
            started = time.monotonic()
            ids_to_analyze = [x.candidate_id for x in candidate_checks if x.state.value != "VALID"]
            if candidate_checks and not ids_to_analyze:
                semantic_results = await asyncio.to_thread(self._analyzer.rebuild_aggregate, job_id, cached_count=len(candidate_checks))
            elif not semantic_input.inputs:
                semantic_results = await asyncio.to_thread(self._analyzer.rebuild_aggregate, job_id, cached_count=0)
            else:
                semantic_results = await asyncio.to_thread(
                    self._analyzer.process,
                    job_id,
                    candidate_ids=ids_to_analyze or None,
                    progress_callback=lambda p: self._progress(job_id, 99, "Analyzing visual candidates with local vision model"),
                    cancel_check=lambda: self._cancelled(job_id),
                )
            timings["analysis"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_SEMANTIC_ANALYSIS_COMPLETE)
        self._guard(job_id)

        decision_check, selections = self._cache.validate_decision(job_id, semantic_results)
        if not decision_check.valid or selections is None:
            self._cache.invalidate_from(job_id, Phase6ResumeStage.SEMANTIC_DECISION)
            self._stage(job_id, JobStage.SELECTING_SEMANTIC_CANDIDATES, "Selecting useful completed visual states")
            started = time.monotonic()
            selections = await asyncio.to_thread(
                self._decision.process,
                job_id,
                progress_callback=lambda p: self._progress(job_id, 99, "Selecting semantic candidates"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings["decision"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_SEMANTIC_CANDIDATES_READY)
        self._guard(job_id)

        final = self._cache.inspect(job_id)
        if final.resume_plan.resume_stage is not Phase6ResumeStage.PHASE6_READY:
            raise Phase6ValidationError("Phase-6 artifacts failed final consistency validation.")
        models = semantic_results.stats.model_usage_counts
        summary = SemanticSummary(
            semantic_input_count=semantic_input.stats.input_count,
            semantic_analysis_count=semantic_results.stats.successful_analysis_count,
            stable_window_count=selections.stats.window_count,
            selected_candidate_count=selections.stats.selected_window_count,
            rejected_window_count=selections.stats.rejected_window_count,
            primary_selected_count=selections.stats.primary_selected_count,
            alternate_selected_count=selections.stats.alternate_selected_count,
            semantic_ambiguous_window_count=selections.stats.semantic_ambiguous_window_count,
            mean_selected_semantic_score=selections.stats.mean_selected_semantic_score,
            model_usage_counts=models,
            fallback_candidate_count=semantic_results.stats.fallback_candidate_count,
            mixed_models_used=len(models) > 1,
            semantic_input_fingerprint=semantic_input.artifact_fingerprint,
            temporal_contexts_fingerprint=temporal.artifact_fingerprint,
            semantic_results_fingerprint=semantic_results.artifact_fingerprint,
            semantic_selections_fingerprint=selections.artifact_fingerprint,
            timings=Phase6Timings(
                semantic_input_seconds=timings["input"],
                temporal_context_seconds=timings["context"],
                semantic_inference_seconds=timings["analysis"],
                semantic_decision_seconds=timings["decision"],
                total_phase6_seconds=time.monotonic() - total_started,
            ),
        )
        self._repository.save_summary(job_id, summary)
        self._events.write(job_id, level="INFO", stage="SEMANTIC_ANALYSIS", message="Phase 6 semantic visual analysis complete")
        if finalize_job:
            self._jobs.mark_completed(job_id, message="Semantic visual candidates ready for final screenshot selection")
        return summary

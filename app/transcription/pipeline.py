from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from app.core.exceptions import JobCancelledError, Phase5ValidationError
from app.core.logging import JobEventLogger
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService
from app.transcription.alignment import CP_CANDIDATE_TRANSCRIPT_ALIGNED, CandidateTranscriptAlignmentService
from app.transcription.cache import TranscriptCacheCoordinator
from app.transcription.context import CP_TRANSCRIPT_CONTEXT_READY, CandidateTranscriptContextService
from app.transcription.engine import CP_RAW_TRANSCRIPTION, TranscriptionEngine
from app.transcription.models import Phase5Timings, TranscriptResumeStage, TranscriptSummary
from app.transcription.normalization import CP_TRANSCRIPT_NORMALIZED, TranscriptNormalizationService
from app.transcription.preparation import CP_AUDIO_READY, AudioPreparationService
from app.transcription.repository import TranscriptionRepository


class TranscriptionPipeline:
    def __init__(
        self,
        *,
        jobs: JobService,
        checkpoints: CheckpointStore,
        events: JobEventLogger,
        preparation: AudioPreparationService,
        engine: TranscriptionEngine,
        normalization: TranscriptNormalizationService,
        alignment: CandidateTranscriptAlignmentService,
        context: CandidateTranscriptContextService,
        cache: TranscriptCacheCoordinator,
        repository: TranscriptionRepository,
    ) -> None:
        self._jobs = jobs
        self._checkpoints = checkpoints
        self._events = events
        self._preparation = preparation
        self._engine = engine
        self._normalization = normalization
        self._alignment = alignment
        self._context = context
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
        if not self._cancelled(job_id):
            self._jobs.update_progress(job_id, value, message)

    async def process(self, job_id: str, *, finalize_job: bool = True) -> TranscriptSummary:
        async with self._locks[job_id]:
            return await self._process_locked(job_id, finalize_job=finalize_job)

    async def _process_locked(self, job_id: str, *, finalize_job: bool) -> TranscriptSummary:
        total_started = time.monotonic()
        removed = self._cache.cleanup_partial_artifacts(job_id)
        snapshot = self._cache.reconcile(job_id)
        self._events.write(
            job_id,
            level="INFO",
            stage="TRANSCRIPTION",
            message=f"Phase-5 resume stage: {snapshot.resume_stage.value}; removed {len(removed)} temporary artifacts",
        )
        timings = {"prep": 0.0, "raw": 0.0, "norm": 0.0, "align": 0.0, "context": 0.0}

        prep_check, preparation = self._cache.validate_preparation(job_id)
        if not prep_check.valid or preparation is None:
            self._cache.invalidate_from(job_id, TranscriptResumeStage.AUDIO_PREPARATION)
            self._stage(job_id, JobStage.VALIDATING_AUDIO, "Validating lecture audio")
            started = time.monotonic()
            preparation = await asyncio.to_thread(
                self._preparation.process,
                job_id,
                progress_callback=lambda p: self._progress(job_id, 97, "Preparing transcription plan"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings["prep"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_AUDIO_READY)
            self._progress(job_id, 97, "Audio ready for transcription")
        else:
            self._progress(job_id, 97, "Audio preparation cache reused")
        self._guard(job_id)

        raw_check, raw, _, _ = self._cache.validate_raw_transcription(job_id, preparation)
        if not raw_check.valid or raw is None:
            self._cache.invalidate_from(job_id, TranscriptResumeStage.RAW_TRANSCRIPTION)
            self._stage(job_id, JobStage.TRANSCRIBING_AUDIO, "Transcribing lecture audio")
            started = time.monotonic()
            raw = await asyncio.to_thread(
                self._engine.process,
                job_id,
                preparation=preparation,
                progress_callback=lambda p: self._progress(job_id, 97 + int(p >= 50), "Transcribing lecture audio"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings["raw"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_RAW_TRANSCRIPTION)
            self._progress(job_id, 98, "Raw transcription complete")
        else:
            self._progress(job_id, 98, "Raw transcription cache reused")
        self._guard(job_id)

        norm_check, transcript = self._cache.validate_normalized(job_id, raw)
        if not norm_check.valid or transcript is None:
            self._cache.invalidate_from(job_id, TranscriptResumeStage.NORMALIZATION)
            self._stage(job_id, JobStage.NORMALIZING_TRANSCRIPT, "Normalizing lecture transcript")
            started = time.monotonic()
            transcript = await asyncio.to_thread(
                self._normalization.process,
                job_id,
                raw=raw,
                progress_callback=lambda p: self._progress(job_id, 98, "Resolving transcript overlaps"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings["norm"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_TRANSCRIPT_NORMALIZED)
            self._progress(job_id, 98, "Transcript normalized")
        else:
            self._progress(job_id, 98, "Normalized transcript cache reused")
        self._guard(job_id)

        align_check, alignment = self._cache.validate_alignment(job_id, transcript)
        if not align_check.valid or alignment is None:
            self._cache.invalidate_from(job_id, TranscriptResumeStage.CANDIDATE_ALIGNMENT)
            self._stage(job_id, JobStage.ALIGNING_TRANSCRIPT, "Aligning transcript with visual candidates")
            started = time.monotonic()
            alignment = await asyncio.to_thread(
                self._alignment.process,
                job_id,
                transcript=transcript,
                progress_callback=lambda p: self._progress(job_id, 99, "Aligning transcript with visual candidates"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings["align"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_CANDIDATE_TRANSCRIPT_ALIGNED)
            self._progress(job_id, 99, "Candidate transcript alignment complete")
        else:
            self._progress(job_id, 99, "Candidate alignment cache reused")
        self._guard(job_id)

        ctx_check, contexts = self._cache.validate_contexts(job_id, transcript, alignment)
        if not ctx_check.valid or contexts is None:
            self._cache.invalidate_from(job_id, TranscriptResumeStage.CONTEXT_EXTRACTION)
            self._stage(job_id, JobStage.BUILDING_TRANSCRIPT_CONTEXT, "Building transcript context for visual candidates")
            started = time.monotonic()
            contexts = await asyncio.to_thread(
                self._context.process,
                job_id,
                transcript=transcript,
                alignment=alignment,
                progress_callback=lambda p: self._progress(job_id, 99, "Building candidate transcript context"),
                cancel_check=lambda: self._cancelled(job_id),
            )
            timings["context"] = time.monotonic() - started
            self._checkpoints.mark_completed(job_id, CP_TRANSCRIPT_CONTEXT_READY)
            self._progress(job_id, 99, "Transcript context ready")
        else:
            self._progress(job_id, 99, "Transcript-context cache reused")
        self._guard(job_id)

        final = self._cache.inspect(job_id)
        if not final.transcript_context_ready.valid:
            raise Phase5ValidationError("Phase-5 artifacts failed final consistency validation.")
        summary = TranscriptSummary(
            audio_duration_seconds=preparation.audio.duration_seconds,
            transcription_chunk_count=len(preparation.chunks),
            raw_segment_count=raw.stats.raw_segment_count,
            normalized_segment_count=transcript.stats.normalized_segment_count,
            primary_candidate_count=contexts.stats.primary_context_count,
            alternate_candidate_count=contexts.stats.alternate_context_count,
            aligned_candidate_count=alignment.stats.candidate_count,
            contexts_with_speech=contexts.stats.contexts_with_speech,
            contexts_without_speech=contexts.stats.contexts_without_speech,
            speech_coverage_ratio=transcript.stats.speech_coverage_ratio,
            detected_language=transcript.primary_language,
            real_time_factor=raw.stats.real_time_factor,
            preparation_fingerprint=preparation.artifact_fingerprint,
            raw_transcript_fingerprint=raw.artifact_fingerprint,
            normalized_transcript_fingerprint=transcript.artifact_fingerprint,
            alignment_fingerprint=alignment.artifact_fingerprint,
            contexts_fingerprint=contexts.artifact_fingerprint,
            timings=Phase5Timings(
                audio_preparation_seconds=timings["prep"],
                raw_transcription_seconds=timings["raw"],
                normalization_seconds=timings["norm"],
                alignment_seconds=timings["align"],
                context_extraction_seconds=timings["context"],
                total_phase5_seconds=time.monotonic() - total_started,
            ),
        )
        self._repository.save_summary(job_id, summary)
        self._events.write(job_id, level="INFO", stage="TRANSCRIPTION", message="Phase 5 transcript context pipeline complete")
        if finalize_job:
            self._jobs.mark_completed(job_id, message="Transcript context ready for visual reasoning")
        return summary

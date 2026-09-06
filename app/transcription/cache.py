from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.candidate_analysis.cache import CP_CANDIDATES_READY
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.exceptions import Phase5ValidationError
from app.jobs.checkpoints import CheckpointStore
from app.storage.workspace import WorkspaceManager
from app.transcription.alignment import CP_CANDIDATE_TRANSCRIPT_ALIGNED, CandidateTranscriptAlignmentService
from app.transcription.context import CP_TRANSCRIPT_CONTEXT_READY, CandidateTranscriptContextService
from app.transcription.engine import CP_RAW_TRANSCRIPTION, TranscriptionEngine
from app.transcription.models import (
    CandidateAlignmentManifest,
    CandidateContextsManifest,
    NormalizedTranscriptManifest,
    RawTranscriptManifest,
    TranscriptCacheSnapshot,
    TranscriptResumeStage,
    TranscriptionPreparationManifest,
)
from app.transcription.normalization import CP_TRANSCRIPT_NORMALIZED, TranscriptNormalizationService
from app.transcription.preparation import CP_AUDIO_READY, AudioPreparationService
from app.transcription.repository import TranscriptionRepository
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import AnalysisArtifactCheck, AnalysisArtifactState


class TranscriptCacheCoordinator:
    def __init__(
        self,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        repository: TranscriptionRepository,
        candidates: CandidateAnalysisRepository,
        preparation: AudioPreparationService,
        engine: TranscriptionEngine,
        normalization: TranscriptNormalizationService,
        alignment: CandidateTranscriptAlignmentService,
        context: CandidateTranscriptContextService,
    ) -> None:
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = repository
        self._candidates = candidates
        self._preparation = preparation
        self._engine = engine
        self._normalization = normalization
        self._alignment = alignment
        self._context = context

    @staticmethod
    def _check(state: AnalysisArtifactState, reason: str | None = None) -> AnalysisArtifactCheck:
        return AnalysisArtifactCheck(state=state, reason=reason)

    def cleanup_partial_artifacts(self, job_id: str) -> list[str]:
        transcript_dir = self._workspace.transcript_dir(job_id)
        removed: list[str] = []
        if not transcript_dir.exists():
            return removed
        for path in transcript_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.endswith(".tmp") or ".tmp." in path.name:
                try:
                    removed.append(path.relative_to(transcript_dir).as_posix())
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        shutil.rmtree(self._workspace.transcript_temp_audio_dir(job_id), ignore_errors=True)
        return removed

    def _current_audio_matches(self, job_id: str, prep: TranscriptionPreparationManifest) -> bool:
        try:
            import os
            from app.media.models import AudioResult
            path = self._workspace.audio_manifest_path(job_id)
            audio = AudioResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
            root = self._workspace.workspace(job_id)
            audio_path = (root / audio.path).resolve()
            audio_path.relative_to(root)
            stat = audio_path.stat()
            if stat.st_size != audio.audio_fingerprint.file_size_bytes or stat.st_mtime_ns != audio.audio_fingerprint.mtime_ns:
                return False
            expected = stable_hash(
                {
                    "manifest": audio.audio_fingerprint.model_dump(mode="json"),
                    "size": prep.audio.file_size_bytes,
                    "duration": round(prep.audio.duration_seconds, 6),
                    "sample_rate": prep.audio.sample_rate_hz,
                    "channels": prep.audio.channels,
                    "bits": prep.audio.bits_per_sample,
                }
            )
            return expected == prep.audio_fingerprint
        except Exception:
            return False

    def validate_preparation(self, job_id: str):
        path = self._workspace.transcript_preparation_path(job_id)
        if not path.exists():
            return self._check(AnalysisArtifactState.MISSING, "artifact_missing"), None
        try:
            value = self._repository.load_preparation(job_id)
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_json"), None
        if not self._checkpoints.is_completed(job_id, CP_AUDIO_READY):
            return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), value
        if not self._checkpoints.is_completed(job_id, CP_CANDIDATES_READY):
            return self._check(AnalysisArtifactState.STALE, "candidates_not_ready"), value
        if not value.is_valid or value.rejection_reasons:
            return self._check(AnalysisArtifactState.CORRUPT, "audio_preparation_invalid"), value
        if value.config_fingerprint != self._preparation.config_fingerprint():
            return self._check(AnalysisArtifactState.STALE, "configuration_mismatch"), value
        if not self._current_audio_matches(job_id, value):
            return self._check(AnalysisArtifactState.STALE, "audio_dependency_mismatch"), value
        if any(c.logical_start_seconds >= c.logical_end_seconds for c in value.chunks):
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_chunk_bounds"), value
        return self._check(AnalysisArtifactState.VALID), value

    def validate_raw_transcription(self, job_id: str, prep: TranscriptionPreparationManifest | None):
        if prep is None:
            return self._check(AnalysisArtifactState.STALE, "upstream_invalid"), None, 0, []
        valid_chunks = []
        invalid_ids: list[int] = []
        for chunk in prep.chunks:
            item = self._engine.validate_cached_chunk(job_id, prep, chunk)
            if item is None:
                invalid_ids.append(chunk.chunk_id)
            else:
                valid_chunks.append(item)
        total = len(prep.chunks)
        path = self._workspace.raw_transcript_path(job_id)
        if invalid_ids:
            state = AnalysisArtifactState.PARTIAL if valid_chunks else AnalysisArtifactState.MISSING
            return self._check(state, "partial_chunk_set"), None, len(valid_chunks), invalid_ids
        if not path.exists():
            return self._check(AnalysisArtifactState.PARTIAL, "raw_assembly_missing"), None, len(valid_chunks), []
        try:
            value = self._repository.load_raw_transcript(job_id)
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_raw_transcript"), None, len(valid_chunks), []
        if not self._checkpoints.is_completed(job_id, CP_RAW_TRANSCRIPTION):
            return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), value, len(valid_chunks), []
        if value.preparation_fingerprint != prep.artifact_fingerprint or value.audio_fingerprint != prep.audio_fingerprint:
            return self._check(AnalysisArtifactState.STALE, "dependency_mismatch"), value, len(valid_chunks), []
        if value.config_fingerprint != self._engine.config_fingerprint():
            return self._check(AnalysisArtifactState.STALE, "configuration_mismatch"), value, len(valid_chunks), []
        expected_chunk_fps = [c.artifact_fingerprint for c in sorted(valid_chunks, key=lambda x: x.chunk.chunk_id)]
        if value.chunk_fingerprints != expected_chunk_fps:
            return self._check(AnalysisArtifactState.STALE, "chunk_fingerprint_mismatch"), value, len(valid_chunks), []
        ids = [s.segment_id for s in value.segments]
        if len(ids) != len(set(ids)):
            return self._check(AnalysisArtifactState.CORRUPT, "duplicate_segment_id"), value, len(valid_chunks), []
        return self._check(AnalysisArtifactState.VALID), value, len(valid_chunks), []

    def validate_normalized(self, job_id: str, raw: RawTranscriptManifest | None):
        if raw is None:
            return self._check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        path = self._workspace.normalized_transcript_path(job_id)
        if not path.exists():
            return self._check(AnalysisArtifactState.MISSING, "artifact_missing"), None
        try:
            value = self._repository.load_transcript(job_id)
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_json"), None
        if not self._checkpoints.is_completed(job_id, CP_TRANSCRIPT_NORMALIZED):
            return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), value
        if value.raw_transcript_fingerprint != raw.artifact_fingerprint:
            return self._check(AnalysisArtifactState.STALE, "dependency_mismatch"), value
        if value.config_fingerprint != self._normalization.config_fingerprint():
            return self._check(AnalysisArtifactState.STALE, "configuration_mismatch"), value
        ids = [s.segment_id for s in value.segments]
        if len(ids) != len(set(ids)) or ids != sorted(ids):
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_segment_ids"), value
        if any(a.start_seconds > b.start_seconds for a, b in zip(value.segments, value.segments[1:])):
            return self._check(AnalysisArtifactState.CORRUPT, "segments_out_of_order"), value
        return self._check(AnalysisArtifactState.VALID), value

    def validate_alignment(self, job_id: str, transcript: NormalizedTranscriptManifest | None):
        if transcript is None:
            return self._check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        path = self._workspace.candidate_alignment_path(job_id)
        if not path.exists():
            return self._check(AnalysisArtifactState.MISSING, "artifact_missing"), None
        try:
            value = self._repository.load_alignment(job_id)
            selections = self._candidates.load_selections(job_id)
            handoff = self._candidates.load_handoff(job_id, include_alternates=True)
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_reference_source"), None
        if not self._checkpoints.is_completed(job_id, CP_CANDIDATE_TRANSCRIPT_ALIGNED):
            return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), value
        if value.normalized_transcript_fingerprint != transcript.artifact_fingerprint or value.selections_fingerprint != selections.artifact_fingerprint:
            return self._check(AnalysisArtifactState.STALE, "dependency_mismatch"), value
        if value.config_fingerprint != self._alignment.config_fingerprint():
            return self._check(AnalysisArtifactState.STALE, "configuration_mismatch"), value
        candidate_ids = {c.candidate_id for c in handoff}
        alignment_ids = [a.candidate_id for a in value.alignments]
        if len(alignment_ids) != len(set(alignment_ids)) or set(alignment_ids) != candidate_ids:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_candidate_reference"), value
        segment_ids = {s.segment_id for s in transcript.segments}
        for a in value.alignments:
            refs = list(a.overlapping_segment_ids) + [x for x in (a.primary_segment_id, a.previous_segment_id, a.next_segment_id, a.boundary_speech_segment_id) if x is not None]
            if any(ref not in segment_ids for ref in refs):
                return self._check(AnalysisArtifactState.CORRUPT, "invalid_segment_reference"), value
        return self._check(AnalysisArtifactState.VALID), value

    def validate_contexts(
        self,
        job_id: str,
        transcript: NormalizedTranscriptManifest | None,
        alignment: CandidateAlignmentManifest | None,
    ):
        if transcript is None or alignment is None:
            return self._check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        path = self._workspace.transcript_contexts_path(job_id)
        if not path.exists():
            return self._check(AnalysisArtifactState.MISSING, "artifact_missing"), None
        try:
            value = self._repository.load_contexts(job_id)
            selections = self._candidates.load_selections(job_id)
            handoff = self._candidates.load_handoff(job_id, include_alternates=True)
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_reference_source"), None
        if not self._checkpoints.is_completed(job_id, CP_TRANSCRIPT_CONTEXT_READY):
            return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), value
        if (
            value.normalized_transcript_fingerprint != transcript.artifact_fingerprint
            or value.alignment_fingerprint != alignment.artifact_fingerprint
            or value.selections_fingerprint != selections.artifact_fingerprint
        ):
            return self._check(AnalysisArtifactState.STALE, "dependency_mismatch"), value
        if value.config_fingerprint != self._context.config_fingerprint():
            return self._check(AnalysisArtifactState.STALE, "configuration_mismatch"), value
        candidate_ids = {c.candidate_id for c in handoff}
        context_ids = [c.candidate_id for c in value.contexts]
        if len(context_ids) != len(set(context_ids)) or set(context_ids) != candidate_ids:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_candidate_reference"), value
        segment_ids = {s.segment_id for s in transcript.segments}
        for c in value.contexts:
            refs = c.before_segment_ids + c.current_segment_ids + c.after_segment_ids
            if any(ref not in segment_ids for ref in refs):
                return self._check(AnalysisArtifactState.CORRUPT, "invalid_segment_reference"), value
        return self._check(AnalysisArtifactState.VALID), value

    def inspect(self, job_id: str) -> TranscriptCacheSnapshot:
        prep_check, prep = self.validate_preparation(job_id)
        raw_check, raw, valid_chunks, invalid_chunk_ids = self.validate_raw_transcription(job_id, prep if prep_check.valid else None)
        norm_check, norm = self.validate_normalized(job_id, raw if raw_check.valid else None)
        align_check, alignment = self.validate_alignment(job_id, norm if norm_check.valid else None)
        ctx_check, contexts = self.validate_contexts(
            job_id,
            norm if norm_check.valid else None,
            alignment if align_check.valid else None,
        )
        ready = (
            self._check(AnalysisArtifactState.VALID)
            if ctx_check.valid and self._checkpoints.is_completed(job_id, CP_TRANSCRIPT_CONTEXT_READY)
            else self._check(AnalysisArtifactState.STALE, "phase5_incomplete")
        )
        if not prep_check.valid:
            resume = TranscriptResumeStage.AUDIO_PREPARATION
        elif not raw_check.valid:
            resume = TranscriptResumeStage.RAW_TRANSCRIPTION
        elif not norm_check.valid:
            resume = TranscriptResumeStage.NORMALIZATION
        elif not align_check.valid:
            resume = TranscriptResumeStage.CANDIDATE_ALIGNMENT
        elif not ctx_check.valid:
            resume = TranscriptResumeStage.CONTEXT_EXTRACTION
        else:
            resume = TranscriptResumeStage.PHASE5_READY
        return TranscriptCacheSnapshot(
            preparation=prep_check,
            raw_transcription=raw_check,
            normalized_transcript=norm_check,
            candidate_alignment=align_check,
            contexts=ctx_check,
            transcript_context_ready=ready,
            valid_raw_chunks=valid_chunks,
            total_raw_chunks=(len(prep.chunks) if prep else 0),
            invalid_raw_chunk_ids=invalid_chunk_ids,
            resume_stage=resume,
        )

    def invalidate_from(self, job_id: str, stage: TranscriptResumeStage) -> None:
        order = [
            TranscriptResumeStage.AUDIO_PREPARATION,
            TranscriptResumeStage.RAW_TRANSCRIPTION,
            TranscriptResumeStage.NORMALIZATION,
            TranscriptResumeStage.CANDIDATE_ALIGNMENT,
            TranscriptResumeStage.CONTEXT_EXTRACTION,
        ]
        if stage is TranscriptResumeStage.PHASE5_READY:
            return
        idx = order.index(stage)
        stages = order[idx:]
        if TranscriptResumeStage.AUDIO_PREPARATION in stages:
            self._workspace.transcript_preparation_path(job_id).unlink(missing_ok=True)
            shutil.rmtree(self._workspace.raw_transcript_chunks_dir(job_id), ignore_errors=True)
            self._checkpoints.invalidate(job_id, CP_AUDIO_READY)
        if TranscriptResumeStage.RAW_TRANSCRIPTION in stages:
            self._workspace.raw_transcript_path(job_id).unlink(missing_ok=True)
            self._checkpoints.invalidate(job_id, CP_RAW_TRANSCRIPTION)
        if TranscriptResumeStage.NORMALIZATION in stages:
            self._workspace.normalized_transcript_path(job_id).unlink(missing_ok=True)
            self._checkpoints.invalidate(job_id, CP_TRANSCRIPT_NORMALIZED)
        if TranscriptResumeStage.CANDIDATE_ALIGNMENT in stages:
            self._workspace.candidate_alignment_path(job_id).unlink(missing_ok=True)
            self._checkpoints.invalidate(job_id, CP_CANDIDATE_TRANSCRIPT_ALIGNED)
        if TranscriptResumeStage.CONTEXT_EXTRACTION in stages:
            self._workspace.transcript_contexts_path(job_id).unlink(missing_ok=True)
            self._workspace.transcript_summary_path(job_id).unlink(missing_ok=True)
            self._checkpoints.invalidate(job_id, CP_TRANSCRIPT_CONTEXT_READY)

    def reconcile(self, job_id: str) -> TranscriptCacheSnapshot:
        snapshot = self.inspect(job_id)
        # Clear impossible downstream checkpoints without destroying valid upstream artifacts.
        if not snapshot.preparation.valid:
            for cp in (CP_AUDIO_READY, CP_RAW_TRANSCRIPTION, CP_TRANSCRIPT_NORMALIZED, CP_CANDIDATE_TRANSCRIPT_ALIGNED, CP_TRANSCRIPT_CONTEXT_READY):
                self._checkpoints.invalidate(job_id, cp)
        elif not snapshot.raw_transcription.valid:
            for cp in (CP_RAW_TRANSCRIPTION, CP_TRANSCRIPT_NORMALIZED, CP_CANDIDATE_TRANSCRIPT_ALIGNED, CP_TRANSCRIPT_CONTEXT_READY):
                self._checkpoints.invalidate(job_id, cp)
        elif not snapshot.normalized_transcript.valid:
            for cp in (CP_TRANSCRIPT_NORMALIZED, CP_CANDIDATE_TRANSCRIPT_ALIGNED, CP_TRANSCRIPT_CONTEXT_READY):
                self._checkpoints.invalidate(job_id, cp)
        elif not snapshot.candidate_alignment.valid:
            for cp in (CP_CANDIDATE_TRANSCRIPT_ALIGNED, CP_TRANSCRIPT_CONTEXT_READY):
                self._checkpoints.invalidate(job_id, cp)
        elif not snapshot.contexts.valid:
            self._checkpoints.invalidate(job_id, CP_TRANSCRIPT_CONTEXT_READY)
        return self.inspect(job_id)

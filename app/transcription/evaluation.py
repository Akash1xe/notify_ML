from __future__ import annotations

from app.core.config import AppSettings
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.transcription.models import Phase5EvaluationReport
from app.transcription.repository import TranscriptionRepository


class Phase5Evaluator:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager, repository: TranscriptionRepository) -> None:
        self._settings = settings
        self._workspace = workspace
        self._repository = repository

    def evaluate(self, job_id: str, *, persist: bool = True) -> Phase5EvaluationReport:
        prep = self._repository.load_preparation(job_id)
        raw = self._repository.load_raw_transcript(job_id)
        transcript = self._repository.load_transcript(job_id)
        alignment = self._repository.load_alignment(job_id)
        contexts = self._repository.load_contexts(job_id)
        warnings = list(dict.fromkeys(prep.warnings + raw.warnings + transcript.warnings))
        if transcript.stats.speech_coverage_ratio < self._settings.phase5_low_speech_coverage_ratio and prep.audio.duration_seconds > 60:
            warnings.append("LOW_SPEECH_COVERAGE")
        empty_ratio = contexts.stats.contexts_without_speech / contexts.stats.context_count if contexts.stats.context_count else 0.0
        if empty_ratio >= self._settings.phase5_high_empty_context_ratio and contexts.stats.context_count:
            warnings.append("HIGH_EMPTY_CONTEXT_RATIO")
        if raw.stats.raw_segment_count and transcript.stats.deduplicated_segment_count / raw.stats.raw_segment_count > 0.5:
            warnings.append("HIGH_OVERLAP_DEDUPLICATION")
        if contexts.stats.context_count and contexts.stats.truncated_context_count / contexts.stats.context_count > 0.5:
            warnings.append("HIGH_CONTEXT_TRUNCATION")
        report = Phase5EvaluationReport(
            audio_duration_seconds=prep.audio.duration_seconds,
            sample_rate_hz=prep.audio.sample_rate_hz,
            channels=prep.audio.channels,
            silent_window_ratio=prep.diagnostics.silent_window_ratio,
            clipping_ratio=prep.diagnostics.clipping_ratio,
            chunk_count=len(prep.chunks),
            raw_segments=raw.stats.raw_segment_count,
            normalized_segments=transcript.stats.normalized_segment_count,
            duplicates_removed=transcript.stats.deduplicated_segment_count,
            segments_merged=transcript.stats.merged_segment_count,
            speech_coverage_ratio=transcript.stats.speech_coverage_ratio,
            candidate_count=alignment.stats.candidate_count,
            overlapping_speech_count=alignment.stats.overlapping_speech_count,
            contexts_with_speech=contexts.stats.contexts_with_speech,
            empty_contexts=contexts.stats.contexts_without_speech,
            sparse_contexts=contexts.stats.sparse_context_count,
            truncated_contexts=contexts.stats.truncated_context_count,
            mean_context_characters=contexts.stats.mean_context_characters,
            real_time_factor=raw.stats.real_time_factor,
            warnings=list(dict.fromkeys(warnings)),
        )
        if persist:
            atomic_write_json(self._workspace.transcript_evaluation_path(job_id), report.model_dump(mode="json"))
        return report

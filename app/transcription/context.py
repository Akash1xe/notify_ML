from __future__ import annotations

import statistics
from typing import Callable

import numpy as np

from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, TranscriptContextError
from app.transcription.models import (
    CandidateAlignmentManifest,
    CandidateContextsManifest,
    CandidateTranscriptContext,
    ContextStats,
    NormalizedTranscriptManifest,
)
from app.transcription.repository import TranscriptTimelineIndex, TranscriptionRepository
from app.video_analysis.fingerprints import stable_hash

TRANSCRIPT_CONTEXT_ALGORITHM_VERSION = "1"
CP_TRANSCRIPT_CONTEXT_READY = "TRANSCRIPT_CONTEXT_READY"

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


class CandidateTranscriptContextService:
    def __init__(
        self,
        settings: AppSettings,
        repository: TranscriptionRepository,
        candidates: CandidateAnalysisRepository,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._candidates = candidates

    def config_fingerprint(self) -> str:
        s = self._settings
        return stable_hash(
            {
                "algorithm_version": TRANSCRIPT_CONTEXT_ALGORITHM_VERSION,
                "before_seconds": s.transcript_context_before_seconds,
                "after_seconds": s.transcript_context_after_seconds,
                "max_boundary_extension": s.transcript_context_max_boundary_extension_seconds,
                "max_characters": s.transcript_context_max_characters,
                "max_words": s.transcript_context_max_words,
                "sparse_words": s.transcript_context_min_words_for_non_sparse,
                "truncation_priority": "current-nearest-before-nearest-after",
            }
        )

    @staticmethod
    def _join(segments) -> str:
        return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()

    @staticmethod
    def _speech_union(segments) -> float:
        intervals = sorted((s.start_seconds, s.end_seconds) for s in segments)
        if not intervals:
            return 0.0
        total = 0.0
        start, end = intervals[0]
        for left, right in intervals[1:]:
            if left <= end:
                end = max(end, right)
            else:
                total += max(0.0, end - start)
                start, end = left, right
        return total + max(0.0, end - start)

    def _budget(self, current, before, after):
        max_chars = self._settings.transcript_context_max_characters
        max_words = self._settings.transcript_context_max_words
        chosen = list(current)
        chars = len(self._join(chosen))
        words = sum(len(s.text.split()) for s in chosen)
        over_due_current = chars > max_chars or words > max_words
        before_near = list(reversed(before))
        after_near = list(after)
        pairs = []
        m = max(len(before_near), len(after_near))
        for i in range(m):
            if i < len(before_near):
                pairs.append(("before", before_near[i]))
            if i < len(after_near):
                pairs.append(("after", after_near[i]))
        selected_before = []
        selected_after = []
        if not over_due_current:
            for region, seg in pairs:
                extra_chars = len(seg.text.strip()) + (1 if chars else 0)
                extra_words = len(seg.text.split())
                if chars + extra_chars > max_chars or words + extra_words > max_words:
                    continue
                chars += extra_chars
                words += extra_words
                if region == "before":
                    selected_before.append(seg)
                else:
                    selected_after.append(seg)
        selected_before.sort(key=lambda s: (s.start_seconds, s.segment_id))
        selected_after.sort(key=lambda s: (s.start_seconds, s.segment_id))
        return (
            selected_before,
            list(current),
            selected_after,
            len(selected_before) < len(before),
            len(selected_after) < len(after),
            over_due_current,
        )

    def process(
        self,
        job_id: str,
        *,
        transcript: NormalizedTranscriptManifest,
        alignment: CandidateAlignmentManifest,
        progress_callback: ProgressCallback = lambda _: None,
        cancel_check: CancelCheck = lambda: False,
    ) -> CandidateContextsManifest:
        handoff = {c.candidate_id: c for c in self._candidates.load_handoff(job_id, include_alternates=True)}
        generated = {c.candidate_id: c for c in self._candidates.load_generated_candidates(job_id).candidates}
        boundaries = {b.boundary_id: b for b in self._candidates.load_boundaries(job_id).boundaries}
        selections = self._candidates.load_selections(job_id)
        index = TranscriptTimelineIndex(transcript)
        by_id = index.by_id
        contexts: list[CandidateTranscriptContext] = []
        audio_duration = max(
            transcript.stats.speech_end_seconds or 0.0,
            max((a.candidate_timestamp_seconds for a in alignment.alignments), default=0.0),
        )
        # Prefer actual audio duration from the raw transcript if available.
        try:
            audio_duration = self._repository.load_raw_transcript(job_id).stats.audio_duration_seconds
        except Exception:
            pass

        for i, item in enumerate(alignment.alignments):
            if cancel_check():
                raise JobCancelledError("Transcript context extraction was cancelled.")
            candidate = handoff.get(item.candidate_id)
            generated_item = generated.get(item.candidate_id)
            if candidate is None or generated_item is None:
                raise TranscriptContextError("Context extraction references an unknown retained candidate.")
            requested_start = max(0.0, candidate.timestamp_seconds - self._settings.transcript_context_before_seconds)
            requested_end = min(audio_duration, candidate.timestamp_seconds + self._settings.transcript_context_after_seconds)
            extension = self._settings.transcript_context_max_boundary_extension_seconds
            raw_segments = index.segments_between(requested_start, requested_end)
            current_ids = set(item.overlapping_segment_ids)
            current = [by_id[sid] for sid in item.overlapping_segment_ids if sid in by_id]
            before = []
            after = []
            boundary_truncated_before = False
            boundary_truncated_after = False
            for seg in raw_segments:
                if seg.segment_id in current_ids:
                    continue
                if seg.end_seconds <= candidate.timestamp_seconds:
                    if seg.start_seconds < requested_start - extension:
                        boundary_truncated_before = True
                    else:
                        before.append(seg)
                elif seg.start_seconds >= candidate.timestamp_seconds:
                    if seg.end_seconds > requested_end + extension:
                        boundary_truncated_after = True
                    else:
                        after.append(seg)
                else:
                    # Small transcript overlap not captured by alignment: treat as CURRENT.
                    current.append(seg)
            current = sorted({s.segment_id: s for s in current}.values(), key=lambda s: (s.start_seconds, s.segment_id))
            before = sorted(before, key=lambda s: (s.start_seconds, s.segment_id))
            after = sorted(after, key=lambda s: (s.start_seconds, s.segment_id))
            chosen_before, chosen_current, chosen_after, budget_tb, budget_ta, current_over = self._budget(current, before, after)
            all_selected = chosen_before + chosen_current + chosen_after
            actual_start = min((s.start_seconds for s in all_selected), default=requested_start)
            actual_end = max((s.end_seconds for s in all_selected), default=requested_end)
            speech_duration = self._speech_union(all_selected)
            context_duration = max(0.0, actual_end - actual_start)
            before_text = self._join(chosen_before)
            current_text = self._join(chosen_current)
            after_text = self._join(chosen_after)
            combined_text = " ".join(v for v in (before_text, current_text, after_text) if v).strip()
            total_words = len(combined_text.split())
            boundary = boundaries.get(candidate.boundary_id) if candidate.boundary_id is not None else None
            context_signature = stable_hash(
                {
                    "before": [s.segment_id for s in chosen_before],
                    "current": [s.segment_id for s in chosen_current],
                    "after": [s.segment_id for s in chosen_after],
                    "config": self.config_fingerprint(),
                }
            )
            contexts.append(
                CandidateTranscriptContext(
                    candidate_id=candidate.candidate_id,
                    stable_window_id=candidate.stable_window_id,
                    boundary_id=candidate.boundary_id,
                    selection_role=candidate.selection_role,
                    candidate_timestamp_seconds=candidate.timestamp_seconds,
                    stable_window_start_seconds=generated_item.stable_window_start_seconds,
                    stable_window_end_seconds=generated_item.stable_window_end_seconds,
                    visual_boundary_timestamp_seconds=(boundary.timestamp_seconds if boundary else None),
                    boundary_to_candidate_seconds=(
                        max(0.0, candidate.timestamp_seconds - boundary.timestamp_seconds) if boundary else None
                    ),
                    ranking_score=candidate.ranking_score,
                    selection_confidence=candidate.selection_confidence,
                    is_ambiguous=candidate.is_ambiguous,
                    sampled_frame_path=candidate.sampled_frame_path,
                    processed_frame_path=candidate.processed_frame_path,
                    alignment_type=item.alignment_type,
                    speech_proximity_score=item.speech_proximity_score,
                    nearest_speech_gap_seconds=item.nearest_speech_gap_seconds,
                    requested_context_start_seconds=requested_start,
                    requested_context_end_seconds=requested_end,
                    actual_context_start_seconds=max(0.0, actual_start),
                    actual_context_end_seconds=max(actual_start, actual_end),
                    before_segment_ids=[s.segment_id for s in chosen_before],
                    current_segment_ids=[s.segment_id for s in chosen_current],
                    after_segment_ids=[s.segment_id for s in chosen_after],
                    before_text=before_text,
                    current_text=current_text,
                    after_text=after_text,
                    combined_text=combined_text,
                    speech_duration_seconds=speech_duration,
                    context_speech_ratio=(min(1.0, speech_duration / context_duration) if context_duration > 0 else 0.0),
                    has_transcript_context=bool(combined_text),
                    is_sparse_context=(total_words < self._settings.transcript_context_min_words_for_non_sparse),
                    truncated_before=(boundary_truncated_before or budget_tb),
                    truncated_after=(boundary_truncated_after or budget_ta),
                    context_over_budget_due_to_current=current_over,
                    context_signature=context_signature,
                )
            )
            if alignment.alignments:
                progress_callback((i + 1) * 100.0 / len(alignment.alignments))

        contexts.sort(key=lambda c: (c.candidate_timestamp_seconds, c.selection_role.value, c.candidate_id))
        sizes = [len(c.combined_text) for c in contexts]
        word_sizes = [len(c.combined_text.split()) for c in contexts]
        durations = [c.actual_context_end_seconds - c.actual_context_start_seconds for c in contexts]
        stats = ContextStats(
            context_count=len(contexts),
            primary_context_count=sum(c.selection_role.value == "PRIMARY" for c in contexts),
            alternate_context_count=sum(c.selection_role.value == "ALTERNATE" for c in contexts),
            contexts_with_speech=sum(c.has_transcript_context for c in contexts),
            contexts_without_speech=sum(not c.has_transcript_context for c in contexts),
            sparse_context_count=sum(c.is_sparse_context for c in contexts),
            contexts_with_current_speech=sum(bool(c.current_text) for c in contexts),
            contexts_without_current_speech=sum(not bool(c.current_text) for c in contexts),
            truncated_context_count=sum(c.truncated_before or c.truncated_after for c in contexts),
            truncated_before_count=sum(c.truncated_before for c in contexts),
            truncated_after_count=sum(c.truncated_after for c in contexts),
            mean_context_characters=(statistics.mean(sizes) if sizes else 0.0),
            median_context_characters=(statistics.median(sizes) if sizes else 0.0),
            p90_context_characters=(float(np.percentile(sizes, 90)) if sizes else 0.0),
            mean_context_words=(statistics.mean(word_sizes) if word_sizes else 0.0),
            mean_actual_context_duration_seconds=(statistics.mean(durations) if durations else 0.0),
        )
        core = {
            "transcript": transcript.artifact_fingerprint,
            "alignment": alignment.artifact_fingerprint,
            "selections": selections.artifact_fingerprint,
            "config": self.config_fingerprint(),
            "contexts": [c.model_dump(mode="json") for c in contexts],
        }
        manifest = CandidateContextsManifest(
            algorithm_version=TRANSCRIPT_CONTEXT_ALGORITHM_VERSION,
            normalized_transcript_fingerprint=transcript.artifact_fingerprint,
            alignment_fingerprint=alignment.artifact_fingerprint,
            selections_fingerprint=selections.artifact_fingerprint,
            config_fingerprint=self.config_fingerprint(),
            artifact_fingerprint=stable_hash(core),
            stats=stats,
            contexts=contexts,
        )
        self._repository.save_contexts(job_id, manifest)
        progress_callback(100.0)
        return manifest

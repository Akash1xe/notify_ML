from __future__ import annotations

import math
import statistics
from typing import Callable

import numpy as np

from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, TranscriptAlignmentError
from app.transcription.models import (
    AlignmentStats,
    AlignmentType,
    CandidateAlignmentManifest,
    CandidateTranscriptAlignment,
    NormalizedTranscriptManifest,
    SpeechDirection,
)
from app.transcription.repository import TranscriptTimelineIndex, TranscriptionRepository
from app.video_analysis.fingerprints import stable_hash

TRANSCRIPT_ALIGNMENT_ALGORITHM_VERSION = "1"
CP_CANDIDATE_TRANSCRIPT_ALIGNED = "CANDIDATE_TRANSCRIPT_ALIGNED"

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


class CandidateTranscriptAlignmentService:
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
        return stable_hash(
            {
                "algorithm_version": TRANSCRIPT_ALIGNMENT_ALGORITHM_VERSION,
                "epsilon": self._settings.alignment_timestamp_epsilon_seconds,
                "proximity_saturation": self._settings.alignment_proximity_saturation_seconds,
                "word_level": self._settings.alignment_word_level_enabled,
                "multiple_overlap_policy": "midpoint-confidence-duration-id",
            }
        )

    def _primary_overlap(self, overlaps, timestamp: float):
        if not overlaps:
            return None
        return min(
            overlaps,
            key=lambda seg: (
                abs(timestamp - (seg.start_seconds + seg.end_seconds) / 2),
                seg.no_speech_prob if seg.no_speech_prob is not None else 1.0,
                -(seg.avg_logprob if seg.avg_logprob is not None else -999.0),
                seg.end_seconds - seg.start_seconds,
                seg.segment_id,
            ),
        )

    def _nearest_word(self, segments, timestamp: float):
        best = None
        for seg in segments:
            for word in seg.words:
                if word.start_seconds <= timestamp <= word.end_seconds:
                    gap = 0.0
                    overlaps = True
                else:
                    gap = min(abs(timestamp - word.start_seconds), abs(timestamp - word.end_seconds))
                    overlaps = False
                key = (gap, word.start_seconds, word.end_seconds, word.word)
                if best is None or key < best[0]:
                    best = (key, word, overlaps)
        return best

    def process(
        self,
        job_id: str,
        *,
        transcript: NormalizedTranscriptManifest,
        progress_callback: ProgressCallback = lambda _: None,
        cancel_check: CancelCheck = lambda: False,
    ) -> CandidateAlignmentManifest:
        handoff = self._candidates.load_handoff(job_id, include_alternates=True)
        selections = self._candidates.load_selections(job_id)
        windows = {w.window_id: w for w in self._candidates.load_stability_windows(job_id).windows}
        boundaries = {b.boundary_id: b for b in self._candidates.load_boundaries(job_id).boundaries}
        index = TranscriptTimelineIndex(transcript)
        alignments: list[CandidateTranscriptAlignment] = []
        epsilon = self._settings.alignment_timestamp_epsilon_seconds
        saturation = self._settings.alignment_proximity_saturation_seconds

        for i, candidate in enumerate(handoff):
            if cancel_check():
                raise JobCancelledError("Transcript alignment was cancelled.")
            timestamp = candidate.timestamp_seconds
            overlaps = index.find_overlapping(timestamp, epsilon)
            primary = self._primary_overlap(overlaps, timestamp)
            previous = index.find_previous(timestamp - epsilon)
            nxt = index.find_next(timestamp + epsilon)
            # Avoid duplicating the primary overlap into previous/next references.
            if primary is not None:
                if previous is not None and previous.segment_id in {s.segment_id for s in overlaps}:
                    pos = primary.segment_id - 2
                    previous = transcript.segments[pos] if pos >= 0 else None
                if nxt is not None and nxt.segment_id in {s.segment_id for s in overlaps}:
                    pos = max(s.segment_id for s in overlaps)
                    nxt = transcript.segments[pos] if pos < len(transcript.segments) else None

            prev_gap = max(0.0, timestamp - previous.end_seconds) if previous is not None else None
            next_gap = max(0.0, nxt.start_seconds - timestamp) if nxt is not None else None
            if primary is not None:
                alignment_type = AlignmentType.OVERLAPPING_SPEECH
                nearest_gap = 0.0
                direction = SpeechDirection.OVERLAPPING
            elif not transcript.segments:
                alignment_type = AlignmentType.NO_TRANSCRIPT
                nearest_gap = None
                direction = SpeechDirection.NONE
            elif previous is None:
                alignment_type = AlignmentType.BEFORE_FIRST_SPEECH
                nearest_gap = next_gap
                direction = SpeechDirection.AFTER if nxt is not None else SpeechDirection.NONE
            elif nxt is None:
                alignment_type = AlignmentType.AFTER_LAST_SPEECH
                nearest_gap = prev_gap
                direction = SpeechDirection.BEFORE
            else:
                alignment_type = AlignmentType.BETWEEN_SPEECH
                nearest_gap = min(prev_gap or 0.0, next_gap or 0.0)
                if abs((prev_gap or 0.0) - (next_gap or 0.0)) <= epsilon:
                    direction = SpeechDirection.EQUAL_DISTANCE
                elif (prev_gap or 0.0) < (next_gap or 0.0):
                    direction = SpeechDirection.BEFORE
                else:
                    direction = SpeechDirection.AFTER
            proximity = 0.0 if nearest_gap is None else max(0.0, 1.0 - nearest_gap / saturation)
            relative = None
            midpoint_distance = None
            if primary is not None:
                duration = primary.end_seconds - primary.start_seconds
                relative = 0.5 if duration <= 1e-9 else max(0.0, min(1.0, (timestamp - primary.start_seconds) / duration))
                midpoint_distance = abs(timestamp - (primary.start_seconds + primary.end_seconds) / 2)

            nearby = []
            if primary is not None:
                nearby.append(primary)
            if previous is not None:
                nearby.append(previous)
            if nxt is not None:
                nearby.append(nxt)
            nearest_word = self._nearest_word(nearby, timestamp) if self._settings.alignment_word_level_enabled else None
            word = nearest_word[1] if nearest_word else None
            word_gap = nearest_word[0][0] if nearest_word else None
            word_overlap = bool(nearest_word[2]) if nearest_word else False

            window = windows.get(candidate.stable_window_id)
            stable_has_speech = bool(
                window
                and index.segments_between(window.start_timestamp_seconds, window.end_timestamp_seconds)
            )
            boundary = boundaries.get(candidate.boundary_id) if candidate.boundary_id is not None else None
            boundary_overlap = (
                self._primary_overlap(index.find_overlapping(boundary.timestamp_seconds, epsilon), boundary.timestamp_seconds)
                if boundary is not None
                else None
            )
            alignments.append(
                CandidateTranscriptAlignment(
                    candidate_id=candidate.candidate_id,
                    stable_window_id=candidate.stable_window_id,
                    boundary_id=candidate.boundary_id,
                    selection_role=candidate.selection_role,
                    candidate_timestamp_seconds=timestamp,
                    alignment_type=alignment_type,
                    overlapping_segment_ids=[s.segment_id for s in overlaps],
                    primary_segment_id=(primary.segment_id if primary else None),
                    previous_segment_id=(previous.segment_id if previous else None),
                    next_segment_id=(nxt.segment_id if nxt else None),
                    previous_gap_seconds=prev_gap,
                    next_gap_seconds=next_gap,
                    nearest_speech_gap_seconds=nearest_gap,
                    nearest_speech_direction=direction,
                    speech_proximity_score=proximity,
                    segment_relative_position=relative,
                    distance_from_segment_midpoint_seconds=midpoint_distance,
                    nearest_word=(word.word.strip() if word else None),
                    nearest_word_start_seconds=(word.start_seconds if word else None),
                    nearest_word_end_seconds=(word.end_seconds if word else None),
                    nearest_word_gap_seconds=word_gap,
                    candidate_overlaps_word=word_overlap,
                    stable_window_has_speech=stable_has_speech,
                    boundary_speech_segment_id=(boundary_overlap.segment_id if boundary_overlap else None),
                )
            )
            if handoff:
                progress_callback((i + 1) * 100.0 / len(handoff))

        alignments.sort(key=lambda x: (x.candidate_timestamp_seconds, x.selection_role.value, x.candidate_id))
        gaps = [a.nearest_speech_gap_seconds for a in alignments if a.nearest_speech_gap_seconds is not None]
        proximities = [a.speech_proximity_score for a in alignments]
        stats = AlignmentStats(
            candidate_count=len(alignments),
            primary_candidate_count=sum(a.selection_role.value == "PRIMARY" for a in alignments),
            alternate_candidate_count=sum(a.selection_role.value == "ALTERNATE" for a in alignments),
            overlapping_speech_count=sum(a.alignment_type is AlignmentType.OVERLAPPING_SPEECH for a in alignments),
            between_speech_count=sum(a.alignment_type is AlignmentType.BETWEEN_SPEECH for a in alignments),
            before_first_speech_count=sum(a.alignment_type is AlignmentType.BEFORE_FIRST_SPEECH for a in alignments),
            after_last_speech_count=sum(a.alignment_type is AlignmentType.AFTER_LAST_SPEECH for a in alignments),
            no_transcript_count=sum(a.alignment_type is AlignmentType.NO_TRANSCRIPT for a in alignments),
            mean_nearest_speech_gap_seconds=(statistics.mean(gaps) if gaps else 0.0),
            median_nearest_speech_gap_seconds=(statistics.median(gaps) if gaps else 0.0),
            p90_nearest_speech_gap_seconds=(float(np.percentile(gaps, 90)) if gaps else 0.0),
            max_nearest_speech_gap_seconds=(max(gaps) if gaps else 0.0),
            mean_speech_proximity_score=(statistics.mean(proximities) if proximities else 0.0),
        )
        core = {
            "transcript": transcript.artifact_fingerprint,
            "selections": selections.artifact_fingerprint,
            "config": self.config_fingerprint(),
            "alignments": [a.model_dump(mode="json") for a in alignments],
        }
        manifest = CandidateAlignmentManifest(
            algorithm_version=TRANSCRIPT_ALIGNMENT_ALGORITHM_VERSION,
            normalized_transcript_fingerprint=transcript.artifact_fingerprint,
            selections_fingerprint=selections.artifact_fingerprint,
            config_fingerprint=self.config_fingerprint(),
            artifact_fingerprint=stable_hash(core),
            stats=stats,
            alignments=alignments,
        )
        self._repository.save_alignment(job_id, manifest)
        progress_callback(100.0)
        return manifest

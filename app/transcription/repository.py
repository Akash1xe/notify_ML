from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import Phase5ValidationError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.transcription.models import (
    CandidateAlignmentManifest,
    CandidateContextsManifest,
    NormalizedTranscriptManifest,
    RawChunkTranscript,
    RawTranscriptManifest,
    SemanticCandidateInput,
    TranscriptSummary,
    TranscriptionPreparationManifest,
)

T = TypeVar("T", bound=BaseModel)


class TranscriptTimelineIndex:
    def __init__(self, transcript: NormalizedTranscriptManifest) -> None:
        self.segments = transcript.segments
        self.starts = [s.start_seconds for s in self.segments]
        self.ends = [s.end_seconds for s in self.segments]
        self.by_id = {s.segment_id: s for s in self.segments}

    def find_overlapping(self, timestamp: float, epsilon: float = 0.0):
        if not self.segments:
            return []
        right = bisect.bisect_right(self.starts, timestamp + epsilon)
        items = []
        for idx in range(max(0, right - 8), right):
            seg = self.segments[idx]
            if seg.end_seconds + epsilon >= timestamp:
                items.append(seg)
        return items

    def find_previous(self, timestamp: float):
        if not self.segments:
            return None
        idx = bisect.bisect_right(self.ends, timestamp) - 1
        return self.segments[idx] if idx >= 0 else None

    def find_next(self, timestamp: float):
        if not self.segments:
            return None
        idx = bisect.bisect_left(self.starts, timestamp)
        return self.segments[idx] if idx < len(self.segments) else None

    def segments_between(self, start: float, end: float):
        if not self.segments or end < start:
            return []
        left = max(0, bisect.bisect_left(self.ends, start))
        right = bisect.bisect_right(self.starts, end)
        return [s for s in self.segments[left:right] if s.end_seconds >= start and s.start_seconds <= end]


class TranscriptionRepository:
    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    @staticmethod
    def _load(path: Path, model: type[T]) -> T:
        try:
            return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise Phase5ValidationError(f"Unable to load {path.name}.") from exc

    @staticmethod
    def _save(path: Path, model: BaseModel) -> None:
        atomic_write_json(path, model.model_dump(mode="json"))

    def load_preparation(self, job_id: str) -> TranscriptionPreparationManifest:
        return self._load(self._workspace.transcript_preparation_path(job_id), TranscriptionPreparationManifest)

    def save_preparation(self, job_id: str, value: TranscriptionPreparationManifest) -> None:
        self._save(self._workspace.transcript_preparation_path(job_id), value)

    def load_raw_chunk(self, job_id: str, chunk_id: int) -> RawChunkTranscript:
        return self._load(self._workspace.raw_transcript_chunk_path(job_id, chunk_id), RawChunkTranscript)

    def save_raw_chunk(self, job_id: str, chunk_id: int, value: RawChunkTranscript) -> None:
        path = self._workspace.raw_transcript_chunk_path(job_id, chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save(path, value)

    def load_raw_transcript(self, job_id: str) -> RawTranscriptManifest:
        return self._load(self._workspace.raw_transcript_path(job_id), RawTranscriptManifest)

    def save_raw_transcript(self, job_id: str, value: RawTranscriptManifest) -> None:
        self._save(self._workspace.raw_transcript_path(job_id), value)

    def load_transcript(self, job_id: str) -> NormalizedTranscriptManifest:
        return self._load(self._workspace.normalized_transcript_path(job_id), NormalizedTranscriptManifest)

    def save_transcript(self, job_id: str, value: NormalizedTranscriptManifest) -> None:
        self._save(self._workspace.normalized_transcript_path(job_id), value)

    def timeline_index(self, job_id: str) -> TranscriptTimelineIndex:
        return TranscriptTimelineIndex(self.load_transcript(job_id))

    def load_alignment(self, job_id: str) -> CandidateAlignmentManifest:
        return self._load(self._workspace.candidate_alignment_path(job_id), CandidateAlignmentManifest)

    def save_alignment(self, job_id: str, value: CandidateAlignmentManifest) -> None:
        self._save(self._workspace.candidate_alignment_path(job_id), value)

    def load_contexts(self, job_id: str) -> CandidateContextsManifest:
        return self._load(self._workspace.transcript_contexts_path(job_id), CandidateContextsManifest)

    def save_contexts(self, job_id: str, value: CandidateContextsManifest) -> None:
        self._save(self._workspace.transcript_contexts_path(job_id), value)

    def get_candidate_context(self, job_id: str, candidate_id: int):
        manifest = self.load_contexts(job_id)
        for item in manifest.contexts:
            if item.candidate_id == candidate_id:
                return item
        raise Phase5ValidationError("Candidate transcript context was not found.")

    def load_summary(self, job_id: str) -> TranscriptSummary:
        return self._load(self._workspace.transcript_summary_path(job_id), TranscriptSummary)

    def save_summary(self, job_id: str, value: TranscriptSummary) -> None:
        self._save(self._workspace.transcript_summary_path(job_id), value)

    def load_semantic_inputs(self, job_id: str) -> list[SemanticCandidateInput]:
        contexts = self.load_contexts(job_id)
        return [
            SemanticCandidateInput(
                candidate_id=item.candidate_id,
                selection_role=item.selection_role,
                candidate_timestamp_seconds=item.candidate_timestamp_seconds,
                stable_window_start_seconds=item.stable_window_start_seconds,
                stable_window_end_seconds=item.stable_window_end_seconds,
                visual_boundary_timestamp_seconds=item.visual_boundary_timestamp_seconds,
                sampled_frame_path=item.sampled_frame_path,
                processed_frame_path=item.processed_frame_path,
                ranking_score=item.ranking_score,
                selection_confidence=item.selection_confidence,
                is_ambiguous=item.is_ambiguous,
                alignment_type=item.alignment_type,
                speech_proximity_score=item.speech_proximity_score,
                before_text=item.before_text,
                current_text=item.current_text,
                after_text=item.after_text,
            )
            for item in contexts.contexts
        ]

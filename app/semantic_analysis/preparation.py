from __future__ import annotations

import statistics
from pathlib import Path
from typing import Callable

from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, SemanticInputError
from app.jobs.checkpoints import CheckpointStore
from app.semantic_analysis.models import (
    SEMANTIC_INPUT_ALGORITHM_VERSION,
    SemanticFrameRef,
    SemanticInputManifest,
    SemanticInputRecord,
    SemanticInputStats,
)
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager
from app.transcription.repository import TranscriptionRepository
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.repository import FrameAnalysisRepository

CP_SEMANTIC_INPUT_READY = "SEMANTIC_INPUT_READY"

CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[int], None]


class SemanticInputPreparationService:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        semantic_repository: SemanticRepository,
        candidate_repository: CandidateAnalysisRepository,
        transcription_repository: TranscriptionRepository,
        frame_repository: FrameAnalysisRepository,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = semantic_repository
        self._candidates = candidate_repository
        self._transcripts = transcription_repository
        self._frames = frame_repository

    def config_fingerprint(self) -> str:
        return stable_hash(
            {
                "algorithm_version": SEMANTIC_INPUT_ALGORITHM_VERSION,
                "max_context_characters": self._settings.semantic_input_max_context_characters,
                "frame_policy": "processed-candidate-frame-v1",
            }
        )

    def _safe_frame(self, job_id: str, relative_path: str) -> Path:
        root = self._workspace.workspace(job_id)
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SemanticInputError("SEMANTIC_INPUT_UNSAFE_FRAME_PATH") from exc
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise SemanticInputError("SEMANTIC_INPUT_FRAME_MISSING")
        if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise SemanticInputError("SEMANTIC_INPUT_INVALID_FRAME")
        return candidate

    def process(
        self,
        job_id: str,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> SemanticInputManifest:
        if not self._checkpoints.is_completed(job_id, "CANDIDATES_READY"):
            raise SemanticInputError("CANDIDATES_READY checkpoint is required.")
        if not self._checkpoints.is_completed(job_id, "TRANSCRIPT_CONTEXT_READY"):
            raise SemanticInputError("TRANSCRIPT_CONTEXT_READY checkpoint is required.")

        handoff = self._candidates.load_handoff(job_id, include_alternates=True)
        ranked = self._candidates.load_ranked_candidates(job_id)
        selections = self._candidates.load_selections(job_id)
        stability = self._candidates.load_stability_windows(job_id)
        boundaries = self._candidates.load_boundaries(job_id)
        contexts = self._transcripts.load_contexts(job_id)
        alignment = self._transcripts.load_alignment(job_id)
        sampling = self._frames.load_sampling(job_id)
        preprocessing = self._frames.load_preprocessing(job_id)

        ctx_by_id = {item.candidate_id: item for item in contexts.contexts}
        ranked_by_id = {item.candidate_id: item for item in ranked.candidates}
        quality_by_index = {item.index: item for item in preprocessing.frames}
        windows_by_id = {item.window_id: item for item in stability.windows}
        boundaries_by_id = {item.boundary_id: item for item in boundaries.boundaries}

        if len({item.candidate_id for item in handoff}) != len(handoff):
            raise SemanticInputError("SEMANTIC_INPUT_DUPLICATE_CANDIDATE")

        records: list[SemanticInputRecord] = []
        context_lengths: list[int] = []
        for ordinal, item in enumerate(handoff):
            if cancel_check and cancel_check():
                raise JobCancelledError(f"Job {job_id} was cancelled")
            context = ctx_by_id.get(item.candidate_id)
            ranked_item = ranked_by_id.get(item.candidate_id)
            window = windows_by_id.get(item.stable_window_id)
            if context is None or ranked_item is None or window is None:
                raise SemanticInputError("SEMANTIC_INPUT_REFERENCE_MISMATCH")
            if context.selection_role is not item.selection_role:
                raise SemanticInputError("SEMANTIC_INPUT_REFERENCE_MISMATCH")
            if context.candidate_timestamp_seconds != item.timestamp_seconds:
                raise SemanticInputError("SEMANTIC_INPUT_REFERENCE_MISMATCH")
            total_text = len(context.before_text) + len(context.current_text) + len(context.after_text)
            if total_text > self._settings.semantic_input_max_context_characters:
                raise SemanticInputError("Semantic transcript context exceeds Phase-6.1 safety limit.")

            quality = quality_by_index.get(item.frame_index)
            if quality is None or not quality.is_valid or quality.is_black:
                raise SemanticInputError("SEMANTIC_INPUT_INVALID_FRAME")
            self._safe_frame(job_id, item.processed_frame_path)
            self._safe_frame(job_id, item.sampled_frame_path)
            boundary = boundaries_by_id.get(item.boundary_id) if item.boundary_id else None
            boundary_ts = boundary.timestamp_seconds if boundary else context.visual_boundary_timestamp_seconds
            boundary_score = boundary.boundary_score if boundary else item.boundary_score
            boundary_type = boundary.boundary_type if boundary else None
            seconds_after_boundary = (
                item.timestamp_seconds - boundary_ts if boundary_ts is not None else None
            )
            relative_position = (
                (item.timestamp_seconds - window.start_timestamp_seconds) / window.duration_seconds
                if window.duration_seconds > 0
                else 0.0
            )
            relative_position = min(1.0, max(0.0, relative_position))
            semantic_frame = SemanticFrameRef(
                frame_index=item.frame_index,
                timestamp_seconds=item.timestamp_seconds,
                relative_path=item.processed_frame_path,
                width=quality.analysis_width,
                height=quality.analysis_height,
                quality_score=quality.quality_score,
                sharpness_score=quality.sharpness_score,
                is_valid=quality.is_valid,
                is_black=quality.is_black,
            )
            base = {
                "candidate_id": item.candidate_id,
                "candidate_timestamp_seconds": item.timestamp_seconds,
                "semantic_frame": semantic_frame.model_dump(mode="json"),
                "stable_window_id": item.stable_window_id,
                "stable_window_start_seconds": window.start_timestamp_seconds,
                "stable_window_end_seconds": window.end_timestamp_seconds,
                "boundary_id": item.boundary_id,
                "before_text": context.before_text,
                "current_text": context.current_text,
                "after_text": context.after_text,
                "alignment_type": context.alignment_type.value,
                "speech_proximity_score": context.speech_proximity_score,
            }
            input_fp = stable_hash(base)
            records.append(
                SemanticInputRecord(
                    candidate_id=item.candidate_id,
                    selection_role=item.selection_role,
                    candidate_timestamp_seconds=item.timestamp_seconds,
                    semantic_frame=semantic_frame,
                    sampled_frame_path=item.sampled_frame_path,
                    processed_frame_path=item.processed_frame_path,
                    ranking_score=item.ranking_score,
                    ranking_position=ranked_item.rank_within_window,
                    selection_confidence=item.selection_confidence,
                    is_ambiguous=item.is_ambiguous,
                    stable_window_id=item.stable_window_id,
                    stable_window_start_seconds=window.start_timestamp_seconds,
                    stable_window_end_seconds=window.end_timestamp_seconds,
                    relative_position_in_stable_window=relative_position,
                    boundary_id=item.boundary_id,
                    boundary_type=boundary_type,
                    visual_boundary_timestamp_seconds=boundary_ts,
                    boundary_score=boundary_score,
                    seconds_after_boundary=seconds_after_boundary,
                    alignment_type=context.alignment_type,
                    speech_proximity_score=context.speech_proximity_score,
                    nearest_speech_gap_seconds=context.nearest_speech_gap_seconds,
                    before_text=context.before_text,
                    current_text=context.current_text,
                    after_text=context.after_text,
                    has_transcript_context=context.has_transcript_context,
                    comparison_group_id=f"window:{item.stable_window_id}",
                    semantic_input_fingerprint=input_fp,
                )
            )
            context_lengths.append(total_text)
            if progress_callback:
                progress_callback(int(((ordinal + 1) / max(1, len(handoff))) * 100))

        role_order = {"PRIMARY": 0, "ALTERNATE": 1}
        records.sort(
            key=lambda x: (
                x.candidate_timestamp_seconds,
                role_order.get(x.selection_role.value, 2),
                x.candidate_id,
            )
        )
        empty_ratio = sum(not x.has_transcript_context for x in records) / len(records) if records else 0.0
        ambiguous_ratio = sum(x.is_ambiguous for x in records) / len(records) if records else 0.0
        warnings: list[str] = []
        if empty_ratio >= 0.5 and records:
            warnings.append("HIGH_EMPTY_CONTEXT_RATIO")
        if ambiguous_ratio >= 0.8 and records:
            warnings.append("HIGH_AMBIGUOUS_CANDIDATE_RATIO")
        stats = SemanticInputStats(
            input_count=len(records),
            primary_input_count=sum(x.selection_role.value == "PRIMARY" for x in records),
            alternate_input_count=sum(x.selection_role.value == "ALTERNATE" for x in records),
            inputs_with_transcript=sum(x.has_transcript_context for x in records),
            inputs_without_transcript=sum(not x.has_transcript_context for x in records),
            ambiguous_input_count=sum(x.is_ambiguous for x in records),
            unique_stable_window_count=len({x.stable_window_id for x in records}),
            mean_context_characters=statistics.fmean(context_lengths) if context_lengths else 0.0,
            median_context_characters=statistics.median(context_lengths) if context_lengths else 0.0,
            warnings=warnings,
        )
        config_fp = self.config_fingerprint()
        artifact_payload = {
            "candidate_selections_fingerprint": selections.artifact_fingerprint,
            "ranked_candidates_fingerprint": ranked.artifact_fingerprint,
            "transcript_contexts_fingerprint": contexts.artifact_fingerprint,
            "transcript_alignment_fingerprint": alignment.artifact_fingerprint,
            "frame_manifest_fingerprint": sampling.artifact_fingerprint,
            "preprocessing_fingerprint": preprocessing.artifact_fingerprint,
            "config_fingerprint": config_fp,
            "records": [x.model_dump(mode="json") for x in records],
        }
        manifest = SemanticInputManifest(
            candidate_selections_fingerprint=selections.artifact_fingerprint,
            ranked_candidates_fingerprint=ranked.artifact_fingerprint,
            transcript_contexts_fingerprint=contexts.artifact_fingerprint,
            transcript_alignment_fingerprint=alignment.artifact_fingerprint,
            frame_manifest_fingerprint=sampling.artifact_fingerprint,
            preprocessing_fingerprint=preprocessing.artifact_fingerprint,
            config_fingerprint=config_fp,
            artifact_fingerprint=stable_hash(artifact_payload),
            stats=stats,
            inputs=records,
        )
        self._repository.save_input_manifest(job_id, manifest)
        return manifest

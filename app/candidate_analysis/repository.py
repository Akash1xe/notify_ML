from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.candidate_analysis.models import (
    BoundariesManifest,
    CandidateAnalysisSummary,
    CandidateHandoff,
    CandidateSourceType,
    GeneratedCandidatesManifest,
    RankedCandidatesManifest,
    ScoredCandidatesManifest,
    SelectionRole,
    SelectionsManifest,
    StabilityWindowsManifest,
)
from app.core.exceptions import CandidateAnalysisValidationError
from app.storage.workspace import WorkspaceManager

T = TypeVar("T", bound=BaseModel)


class CandidateAnalysisRepository:
    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    @staticmethod
    def _load(path: Path, model: type[T]) -> T:
        try:
            return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise CandidateAnalysisValidationError(f"Unable to load {path.name}.") from exc

    def load_stability_windows(self, job_id: str) -> StabilityWindowsManifest:
        return self._load(self._workspace.stability_windows_path(job_id), StabilityWindowsManifest)

    def load_boundaries(self, job_id: str) -> BoundariesManifest:
        return self._load(self._workspace.boundaries_path(job_id), BoundariesManifest)

    def load_generated_candidates(self, job_id: str) -> GeneratedCandidatesManifest:
        return self._load(self._workspace.generated_candidates_path(job_id), GeneratedCandidatesManifest)

    def load_scored_candidates(self, job_id: str) -> ScoredCandidatesManifest:
        return self._load(self._workspace.scored_candidates_path(job_id), ScoredCandidatesManifest)

    def load_ranked_candidates(self, job_id: str) -> RankedCandidatesManifest:
        return self._load(self._workspace.ranked_candidates_path(job_id), RankedCandidatesManifest)

    def load_selections(self, job_id: str) -> SelectionsManifest:
        return self._load(self._workspace.candidate_selections_path(job_id), SelectionsManifest)

    def load_summary(self, job_id: str) -> CandidateAnalysisSummary:
        return self._load(self._workspace.candidate_summary_path(job_id), CandidateAnalysisSummary)

    def load_handoff(self, job_id: str, *, include_alternates: bool = True) -> list[CandidateHandoff]:
        generated = self.load_generated_candidates(job_id)
        scored = self.load_scored_candidates(job_id)
        ranked = self.load_ranked_candidates(job_id)
        selections = self.load_selections(job_id)
        generated_by_id = {item.candidate_id: item for item in generated.candidates}
        scored_by_id = {item.candidate_id: item for item in scored.candidates}
        ranked_by_id = {item.candidate_id: item for item in ranked.candidates}
        selection_by_window = {item.stable_window_id: item for item in selections.windows}
        selected_ids: list[int] = []
        for selection in selections.windows:
            if selection.primary_candidate_id is not None:
                selected_ids.append(selection.primary_candidate_id)
            if include_alternates:
                selected_ids.extend(selection.alternate_candidate_ids)
        handoff: list[CandidateHandoff] = []
        for candidate_id in selected_ids:
            generated_item = generated_by_id.get(candidate_id)
            scored_item = scored_by_id.get(candidate_id)
            ranked_item = ranked_by_id.get(candidate_id)
            if not generated_item or not scored_item or not ranked_item:
                raise CandidateAnalysisValidationError("Candidate handoff references are inconsistent.")
            if generated_item.frame_index is None or generated_item.frame_timestamp_seconds is None:
                raise CandidateAnalysisValidationError("Selected candidate has no mapped frame.")
            if not generated_item.sampled_frame_path or not generated_item.processed_frame_path:
                raise CandidateAnalysisValidationError("Selected candidate frame paths are missing.")
            window_selection = selection_by_window[generated_item.stable_window_id]
            handoff.append(
                CandidateHandoff(
                    candidate_id=candidate_id,
                    stable_window_id=generated_item.stable_window_id,
                    boundary_id=generated_item.boundary_id,
                    timestamp_seconds=generated_item.frame_timestamp_seconds,
                    frame_index=generated_item.frame_index,
                    sampled_frame_path=generated_item.sampled_frame_path,
                    processed_frame_path=generated_item.processed_frame_path,
                    selection_role=ranked_item.selection_role,
                    ranking_score=ranked_item.ranking_score,
                    selection_confidence=window_selection.selection_confidence,
                    is_ambiguous=window_selection.is_ambiguous,
                    boundary_score=scored_item.boundary_score,
                    completeness_heuristic_score=scored_item.completeness_heuristic_score,
                    visual_quality_score=scored_item.visual_quality_score,
                    transition_safety_score=scored_item.transition_safety_score,
                )
            )
        handoff.sort(key=lambda item: (item.timestamp_seconds, item.candidate_id))
        return handoff

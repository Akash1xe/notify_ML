from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import Phase6ValidationError
from app.semantic_analysis.models import (
    CandidateSemanticArtifact,
    Phase6EvaluationReport,
    Phase7CandidateHandoff,
    SemanticInputManifest,
    SemanticSelectionsManifest,
    SemanticSummary,
    SemanticResultsManifest,
    TemporalContextsManifest,
)
from app.storage.workspace import WorkspaceManager, atomic_write_json

T = TypeVar("T", bound=BaseModel)


class SemanticRepository:
    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    @staticmethod
    def _load(path: Path, model: type[T]) -> T:
        try:
            return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise Phase6ValidationError(f"Unable to load {path.name}.") from exc

    @staticmethod
    def _save(path: Path, value: BaseModel) -> None:
        atomic_write_json(path, value.model_dump(mode="json"))

    def load_input_manifest(self, job_id: str) -> SemanticInputManifest:
        return self._load(self._workspace.semantic_input_manifest_path(job_id), SemanticInputManifest)

    def save_input_manifest(self, job_id: str, value: SemanticInputManifest) -> None:
        self._save(self._workspace.semantic_input_manifest_path(job_id), value)

    def get_semantic_input(self, job_id: str, candidate_id: int):
        for item in self.load_input_manifest(job_id).inputs:
            if item.candidate_id == candidate_id:
                return item
        raise Phase6ValidationError("Semantic input candidate is not available.")

    def load_temporal_contexts(self, job_id: str) -> TemporalContextsManifest:
        return self._load(self._workspace.semantic_temporal_contexts_path(job_id), TemporalContextsManifest)

    def save_temporal_contexts(self, job_id: str, value: TemporalContextsManifest) -> None:
        self._save(self._workspace.semantic_temporal_contexts_path(job_id), value)

    def get_temporal_context(self, job_id: str, candidate_id: int):
        for item in self.load_temporal_contexts(job_id).contexts:
            if item.candidate_id == candidate_id:
                return item
        raise Phase6ValidationError("Temporal context is not available.")

    def load_candidate_artifact(self, job_id: str, candidate_id: int) -> CandidateSemanticArtifact:
        return self._load(self._workspace.semantic_candidate_result_path(job_id, candidate_id), CandidateSemanticArtifact)

    def save_candidate_artifact(self, job_id: str, candidate_id: int, value: CandidateSemanticArtifact) -> None:
        self._save(self._workspace.semantic_candidate_result_path(job_id, candidate_id), value)

    def load_results(self, job_id: str) -> SemanticResultsManifest:
        return self._load(self._workspace.semantic_results_path(job_id), SemanticResultsManifest)

    def save_results(self, job_id: str, value: SemanticResultsManifest) -> None:
        self._save(self._workspace.semantic_results_path(job_id), value)

    def load_selections(self, job_id: str) -> SemanticSelectionsManifest:
        return self._load(self._workspace.semantic_selections_path(job_id), SemanticSelectionsManifest)

    def save_selections(self, job_id: str, value: SemanticSelectionsManifest) -> None:
        self._save(self._workspace.semantic_selections_path(job_id), value)

    def load_summary(self, job_id: str) -> SemanticSummary:
        return self._load(self._workspace.semantic_summary_path(job_id), SemanticSummary)

    def save_summary(self, job_id: str, value: SemanticSummary) -> None:
        self._save(self._workspace.semantic_summary_path(job_id), value)

    def save_evaluation(self, job_id: str, value: Phase6EvaluationReport) -> None:
        self._save(self._workspace.semantic_evaluation_path(job_id), value)

    def load_selected_candidates(self, job_id: str) -> list[int]:
        return list(self.load_selections(job_id).selected_candidate_ids)

    def load_phase7_handoff(self, job_id: str) -> list[Phase7CandidateHandoff]:
        inputs = {item.candidate_id: item for item in self.load_input_manifest(job_id).inputs}
        results = {item.candidate_id: item for item in self.load_results(job_id).results}
        selections = self.load_selections(job_id)
        decisions = {item.candidate_id: item for item in selections.candidate_decisions}
        windows = {item.stable_window_id: item for item in selections.window_decisions}
        handoff: list[Phase7CandidateHandoff] = []
        for candidate_id in selections.selected_candidate_ids:
            semantic_input = inputs.get(candidate_id)
            semantic_result = results.get(candidate_id)
            decision = decisions.get(candidate_id)
            if semantic_input is None or semantic_result is None or decision is None:
                raise Phase6ValidationError("Phase-7 handoff references are inconsistent.")
            window = windows.get(semantic_input.stable_window_id)
            if window is None or window.selected_candidate_id != candidate_id:
                raise Phase6ValidationError("Phase-7 selected window reference is inconsistent.")
            handoff.append(
                Phase7CandidateHandoff(
                    candidate_id=candidate_id,
                    stable_window_id=semantic_input.stable_window_id,
                    candidate_timestamp_seconds=semantic_input.candidate_timestamp_seconds,
                    analysis_frame_path=semantic_input.processed_frame_path,
                    content_type=semantic_result.content_type,
                    completion_state=semantic_result.completion_state,
                    educational_usefulness=semantic_result.educational_usefulness,
                    semantic_decision_score=decision.semantic_decision_score,
                    selection_role=semantic_input.selection_role,
                    semantic_window_decision=window.decision,
                )
            )
        handoff.sort(key=lambda x: (x.candidate_timestamp_seconds, x.candidate_id))
        return handoff

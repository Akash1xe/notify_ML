from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.candidate_analysis.boundaries import boundary_artifact_fingerprint, boundary_config_fingerprint
from app.candidate_analysis.generation import generation_artifact_fingerprint, generation_config_fingerprint
from app.candidate_analysis.heuristics import heuristics_artifact_fingerprint, heuristics_config_fingerprint
from app.candidate_analysis.models import (
    BoundariesManifest,
    CandidateAnalysisSummary,
    CandidateCacheSnapshot,
    CandidateResumeStage,
    GeneratedCandidatesManifest,
    RankedCandidatesManifest,
    ScoredCandidatesManifest,
    SelectionsManifest,
    StabilityWindowsManifest,
)
from app.candidate_analysis.ranking import (
    ranked_artifact_fingerprint,
    ranking_config_fingerprint,
    selections_artifact_fingerprint,
)
from app.candidate_analysis.stability import stability_artifact_fingerprint, stability_config_fingerprint
from app.core.config import AppSettings
from app.jobs.checkpoints import CheckpointStore
from app.storage.workspace import WorkspaceManager
from app.video_analysis.models import (
    AnalysisArtifactCheck,
    AnalysisArtifactState,
    DifferenceManifest,
    MajorChangesManifest,
    PreprocessingManifest,
    SamplingManifest,
    TimelineManifest,
)

CP_STABILITY_WINDOWS = "STABILITY_WINDOWS_DETECTED"
CP_BOUNDARIES = "STABILITY_BOUNDARIES_DETECTED"
CP_CANDIDATES_GENERATED = "CANDIDATE_FRAMES_GENERATED"
CP_CANDIDATE_HEURISTICS = "CANDIDATE_HEURISTICS_COMPLETE"
CP_CANDIDATES_RANKED = "CANDIDATES_RANKED"
CP_CANDIDATES_READY = "CANDIDATES_READY"

T = TypeVar("T", bound=BaseModel)


def _check(state: AnalysisArtifactState, reason: str | None = None) -> AnalysisArtifactCheck:
    return AnalysisArtifactCheck(state=state, reason=reason)


class CandidateAnalysisCacheManager:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints

    @staticmethod
    def _load_model(path: Path, model: type[T]) -> tuple[T | None, AnalysisArtifactCheck]:
        if not path.exists():
            return None, _check(AnalysisArtifactState.MISSING, "artifact_missing")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return model.model_validate(payload), _check(AnalysisArtifactState.VALID)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError):
            return None, _check(AnalysisArtifactState.CORRUPT, "invalid_json_or_schema")

    def _safe_existing_relative(self, job_id: str, relative: str | None) -> bool:
        if not relative:
            return False
        root = self._workspace.workspace(job_id)
        try:
            candidate = (root / relative).resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            return False
        return candidate.exists() and candidate.is_file()

    def validate_stability_windows(
        self,
        job_id: str,
        timeline: TimelineManifest,
        differences: DifferenceManifest,
        preprocessing: PreprocessingManifest,
        major_changes: MajorChangesManifest,
    ) -> tuple[AnalysisArtifactCheck, StabilityWindowsManifest | None]:
        if not self._checkpoints.is_completed(job_id, CP_STABILITY_WINDOWS):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(self._workspace.stability_windows_path(job_id), StabilityWindowsManifest)
        if manifest is None:
            return result, None
        if manifest.timeline_fingerprint != timeline.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "timeline_dependency_mismatch"), None
        if manifest.differences_fingerprint != differences.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "differences_dependency_mismatch"), None
        if manifest.preprocessing_fingerprint != preprocessing.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "preprocessing_dependency_mismatch"), None
        if manifest.major_changes_fingerprint != major_changes.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "major_changes_dependency_mismatch"), None
        if manifest.config_fingerprint != stability_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if stability_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        timeline_ids = {segment.segment_id for segment in timeline.segments}
        if any(window.source_segment_id not in timeline_ids for window in manifest.windows):
            return _check(AnalysisArtifactState.CORRUPT, "timeline_reference_missing"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def validate_boundaries(
        self,
        job_id: str,
        windows: StabilityWindowsManifest | None,
        timeline: TimelineManifest,
        differences: DifferenceManifest,
        major_changes: MajorChangesManifest,
    ) -> tuple[AnalysisArtifactCheck, BoundariesManifest | None]:
        if windows is None:
            return _check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        if not self._checkpoints.is_completed(job_id, CP_BOUNDARIES):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(self._workspace.boundaries_path(job_id), BoundariesManifest)
        if manifest is None:
            return result, None
        if manifest.stability_windows_fingerprint != windows.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "stability_windows_dependency_mismatch"), None
        if manifest.timeline_fingerprint != timeline.artifact_fingerprint or manifest.differences_fingerprint != differences.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "phase3_dependency_mismatch"), None
        if manifest.major_changes_fingerprint != major_changes.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "major_changes_dependency_mismatch"), None
        if manifest.config_fingerprint != boundary_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if boundary_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        window_ids = {window.window_id for window in windows.windows}
        if any(item.stable_window_id not in window_ids for item in manifest.boundaries):
            return _check(AnalysisArtifactState.CORRUPT, "window_reference_missing"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def validate_generated_candidates(
        self,
        job_id: str,
        windows: StabilityWindowsManifest | None,
        boundaries: BoundariesManifest | None,
        sampling: SamplingManifest,
        preprocessing: PreprocessingManifest,
    ) -> tuple[AnalysisArtifactCheck, GeneratedCandidatesManifest | None]:
        if windows is None or boundaries is None:
            return _check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        if not self._checkpoints.is_completed(job_id, CP_CANDIDATES_GENERATED):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(self._workspace.generated_candidates_path(job_id), GeneratedCandidatesManifest)
        if manifest is None:
            return result, None
        if manifest.stability_windows_fingerprint != windows.artifact_fingerprint or manifest.boundaries_fingerprint != boundaries.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "candidate_source_dependency_mismatch"), None
        if manifest.sampling_fingerprint != sampling.artifact_fingerprint or manifest.preprocessing_fingerprint != preprocessing.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "frame_dependency_mismatch"), None
        if manifest.config_fingerprint != generation_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if generation_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        ids = [item.candidate_id for item in manifest.candidates]
        if len(ids) != len(set(ids)):
            return _check(AnalysisArtifactState.CORRUPT, "duplicate_candidate_ids"), None
        window_ids = {window.window_id for window in windows.windows}
        boundary_ids = {item.boundary_id for item in boundaries.boundaries}
        for candidate in manifest.candidates:
            if candidate.stable_window_id not in window_ids:
                return _check(AnalysisArtifactState.CORRUPT, "window_reference_missing"), None
            if candidate.boundary_id is not None and candidate.boundary_id not in boundary_ids:
                return _check(AnalysisArtifactState.CORRUPT, "boundary_reference_missing"), None
            if candidate.is_valid:
                if not self._safe_existing_relative(job_id, candidate.sampled_frame_path):
                    return _check(AnalysisArtifactState.CORRUPT, "sampled_frame_path_invalid"), None
                if not self._safe_existing_relative(job_id, candidate.processed_frame_path):
                    return _check(AnalysisArtifactState.CORRUPT, "processed_frame_path_invalid"), None
                if candidate.frame_timestamp_seconds is None:
                    return _check(AnalysisArtifactState.CORRUPT, "mapped_timestamp_missing"), None
                if not (
                    candidate.stable_window_start_seconds - 1e-6
                    <= candidate.frame_timestamp_seconds
                    <= candidate.stable_window_end_seconds + 1e-6
                ):
                    return _check(AnalysisArtifactState.CORRUPT, "candidate_outside_window"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def validate_scored_candidates(
        self,
        job_id: str,
        generated: GeneratedCandidatesManifest | None,
        windows: StabilityWindowsManifest | None,
        boundaries: BoundariesManifest | None,
        preprocessing: PreprocessingManifest,
        differences: DifferenceManifest,
        timeline: TimelineManifest,
    ) -> tuple[AnalysisArtifactCheck, ScoredCandidatesManifest | None]:
        if generated is None or windows is None or boundaries is None:
            return _check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        if not self._checkpoints.is_completed(job_id, CP_CANDIDATE_HEURISTICS):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(self._workspace.scored_candidates_path(job_id), ScoredCandidatesManifest)
        if manifest is None:
            return result, None
        expected = (
            manifest.generated_candidates_fingerprint == generated.artifact_fingerprint
            and manifest.boundaries_fingerprint == boundaries.artifact_fingerprint
            and manifest.stability_windows_fingerprint == windows.artifact_fingerprint
            and manifest.preprocessing_fingerprint == preprocessing.artifact_fingerprint
            and manifest.differences_fingerprint == differences.artifact_fingerprint
            and manifest.timeline_fingerprint == timeline.artifact_fingerprint
        )
        if not expected:
            return _check(AnalysisArtifactState.STALE, "heuristic_dependency_mismatch"), None
        if manifest.config_fingerprint != heuristics_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if heuristics_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        if {item.candidate_id for item in manifest.candidates} != {item.candidate_id for item in generated.candidates}:
            return _check(AnalysisArtifactState.CORRUPT, "candidate_reference_mismatch"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def validate_ranking(
        self,
        job_id: str,
        scored: ScoredCandidatesManifest | None,
        windows: StabilityWindowsManifest | None,
    ) -> tuple[AnalysisArtifactCheck, RankedCandidatesManifest | None, SelectionsManifest | None]:
        if scored is None or windows is None:
            return _check(AnalysisArtifactState.STALE, "upstream_invalid"), None, None
        if not self._checkpoints.is_completed(job_id, CP_CANDIDATES_RANKED):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None, None
        ranked, ranked_result = self._load_model(self._workspace.ranked_candidates_path(job_id), RankedCandidatesManifest)
        selections, selections_result = self._load_model(self._workspace.candidate_selections_path(job_id), SelectionsManifest)
        if ranked is None:
            return ranked_result, None, None
        if selections is None:
            return selections_result, None, None
        config = ranking_config_fingerprint(self._settings)
        if ranked.scored_candidates_fingerprint != scored.artifact_fingerprint or selections.scored_candidates_fingerprint != scored.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "scored_candidates_dependency_mismatch"), None, None
        if selections.stability_windows_fingerprint != windows.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "stability_windows_dependency_mismatch"), None, None
        if ranked.config_fingerprint != config or selections.config_fingerprint != config:
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None, None
        if ranked_artifact_fingerprint(ranked) != ranked.artifact_fingerprint or selections_artifact_fingerprint(selections) != selections.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None, None
        if selections.ranked_candidates_fingerprint != ranked.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "ranked_dependency_mismatch"), None, None
        scored_valid = {item.candidate_id for item in scored.candidates if item.is_valid and item.frame_timestamp_seconds is not None}
        ranked_ids = [item.candidate_id for item in ranked.candidates]
        if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != scored_valid:
            return _check(AnalysisArtifactState.CORRUPT, "ranked_candidate_reference_mismatch"), None, None
        ranked_by_id = {item.candidate_id: item for item in ranked.candidates}
        ranked_by_window: dict[int, list] = {}
        for item in ranked.candidates:
            ranked_by_window.setdefault(item.stable_window_id, []).append(item)
        for window_items in ranked_by_window.values():
            ranks = [item.rank_within_window for item in window_items]
            if len(ranks) != len(set(ranks)) or sorted(ranks) != list(range(1, len(ranks) + 1)):
                return _check(AnalysisArtifactState.CORRUPT, "invalid_window_ranks"), None, None
            if sum(item.selection_role.value == "PRIMARY" for item in window_items) > 1:
                return _check(AnalysisArtifactState.CORRUPT, "duplicate_primary_role"), None, None
        seen_windows: set[int] = set()
        for selection in selections.windows:
            if selection.stable_window_id in seen_windows:
                return _check(AnalysisArtifactState.CORRUPT, "duplicate_window_selection"), None, None
            seen_windows.add(selection.stable_window_id)
            selected = ([selection.primary_candidate_id] if selection.primary_candidate_id else []) + selection.alternate_candidate_ids
            if len(selected) != len(set(selected)):
                return _check(AnalysisArtifactState.CORRUPT, "duplicate_selection_candidate"), None, None
            for candidate_id in selected:
                item = ranked_by_id.get(candidate_id)
                if item is None or item.stable_window_id != selection.stable_window_id:
                    return _check(AnalysisArtifactState.CORRUPT, "selection_reference_mismatch"), None, None
        return _check(AnalysisArtifactState.VALID), ranked, selections

    def _completion_check(
        self,
        job_id: str,
        ranked: RankedCandidatesManifest | None,
        selections: SelectionsManifest | None,
    ) -> AnalysisArtifactCheck:
        if ranked is None or selections is None or not self._checkpoints.is_completed(job_id, CP_CANDIDATES_READY):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing")
        summary, result = self._load_model(self._workspace.candidate_summary_path(job_id), CandidateAnalysisSummary)
        if summary is None:
            return result
        if summary.ranked_candidates_fingerprint != ranked.artifact_fingerprint or summary.selections_fingerprint != selections.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "summary_dependency_mismatch")
        return _check(AnalysisArtifactState.VALID)

    def inspect(
        self,
        job_id: str,
        *,
        sampling: SamplingManifest,
        preprocessing: PreprocessingManifest,
        differences: DifferenceManifest,
        major_changes: MajorChangesManifest,
        timeline: TimelineManifest,
    ) -> CandidateCacheSnapshot:
        windows_check, windows = self.validate_stability_windows(job_id, timeline, differences, preprocessing, major_changes)
        stale = _check(AnalysisArtifactState.STALE, "upstream_invalid")
        missing_completion = _check(AnalysisArtifactState.MISSING, "artifact_or_checkpoint_missing")
        if not windows_check.valid:
            return CandidateCacheSnapshot(stability_windows=windows_check, boundaries=stale, generated_candidates=stale, scored_candidates=stale, ranking=stale, candidates_ready=missing_completion, resume_stage=CandidateResumeStage.STABILITY_WINDOWS)
        boundaries_check, boundaries = self.validate_boundaries(job_id, windows, timeline, differences, major_changes)
        if not boundaries_check.valid:
            return CandidateCacheSnapshot(stability_windows=windows_check, boundaries=boundaries_check, generated_candidates=stale, scored_candidates=stale, ranking=stale, candidates_ready=missing_completion, resume_stage=CandidateResumeStage.BOUNDARY_DETECTION)
        generated_check, generated = self.validate_generated_candidates(job_id, windows, boundaries, sampling, preprocessing)
        if not generated_check.valid:
            return CandidateCacheSnapshot(stability_windows=windows_check, boundaries=boundaries_check, generated_candidates=generated_check, scored_candidates=stale, ranking=stale, candidates_ready=missing_completion, resume_stage=CandidateResumeStage.CANDIDATE_GENERATION)
        scored_check, scored = self.validate_scored_candidates(job_id, generated, windows, boundaries, preprocessing, differences, timeline)
        if not scored_check.valid:
            return CandidateCacheSnapshot(stability_windows=windows_check, boundaries=boundaries_check, generated_candidates=generated_check, scored_candidates=scored_check, ranking=stale, candidates_ready=missing_completion, resume_stage=CandidateResumeStage.CANDIDATE_HEURISTICS)
        ranking_check, ranked, selections = self.validate_ranking(job_id, scored, windows)
        if not ranking_check.valid:
            return CandidateCacheSnapshot(stability_windows=windows_check, boundaries=boundaries_check, generated_candidates=generated_check, scored_candidates=scored_check, ranking=ranking_check, candidates_ready=missing_completion, resume_stage=CandidateResumeStage.CANDIDATE_RANKING)
        return CandidateCacheSnapshot(
            stability_windows=windows_check,
            boundaries=boundaries_check,
            generated_candidates=generated_check,
            scored_candidates=scored_check,
            ranking=ranking_check,
            candidates_ready=self._completion_check(job_id, ranked, selections),
            resume_stage=CandidateResumeStage.PHASE4_READY,
        )

    def reconcile(self, job_id: str, **phase3) -> CandidateCacheSnapshot:
        self.cleanup_partial_artifacts(job_id)
        snapshot = self.inspect(job_id, **phase3)
        if snapshot.resume_stage is not CandidateResumeStage.PHASE4_READY:
            self.invalidate_from(job_id, snapshot.resume_stage)
        elif not snapshot.candidates_ready.valid:
            self._checkpoints.invalidate(job_id, CP_CANDIDATES_READY)
            self._workspace.candidate_summary_path(job_id).unlink(missing_ok=True)
        return snapshot

    def invalidate_from(self, job_id: str, stage: CandidateResumeStage) -> None:
        order = [
            CandidateResumeStage.STABILITY_WINDOWS,
            CandidateResumeStage.BOUNDARY_DETECTION,
            CandidateResumeStage.CANDIDATE_GENERATION,
            CandidateResumeStage.CANDIDATE_HEURISTICS,
            CandidateResumeStage.CANDIDATE_RANKING,
        ]
        checkpoints = [CP_STABILITY_WINDOWS, CP_BOUNDARIES, CP_CANDIDATES_GENERATED, CP_CANDIDATE_HEURISTICS, CP_CANDIDATES_RANKED]
        paths = [
            [self._workspace.stability_windows_path(job_id)],
            [self._workspace.boundaries_path(job_id)],
            [self._workspace.generated_candidates_path(job_id)],
            [self._workspace.scored_candidates_path(job_id)],
            [self._workspace.ranked_candidates_path(job_id), self._workspace.candidate_selections_path(job_id)],
        ]
        self._checkpoints.invalidate(job_id, CP_CANDIDATES_READY)
        self._workspace.candidate_summary_path(job_id).unlink(missing_ok=True)
        if stage is CandidateResumeStage.PHASE4_READY:
            return
        start = order.index(stage)
        for checkpoint in checkpoints[start:]:
            self._checkpoints.invalidate(job_id, checkpoint)
        for group in paths[start:]:
            for path in group:
                path.unlink(missing_ok=True)

    def cleanup_partial_artifacts(self, job_id: str) -> list[str]:
        removed: list[str] = []
        candidates_dir = self._workspace.candidates_dir(job_id)
        candidates_dir.mkdir(parents=True, exist_ok=True)
        known = [
            self._workspace.stability_windows_path(job_id),
            self._workspace.boundaries_path(job_id),
            self._workspace.generated_candidates_path(job_id),
            self._workspace.scored_candidates_path(job_id),
            self._workspace.ranked_candidates_path(job_id),
            self._workspace.candidate_selections_path(job_id),
            self._workspace.candidate_summary_path(job_id),
            self._workspace.candidate_evaluation_path(job_id),
        ]
        for final_path in known:
            temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            if temp_path.exists() and temp_path.is_file():
                removed.append(self._workspace.relative_to_workspace(job_id, temp_path))
                temp_path.unlink(missing_ok=True)
        for path in candidates_dir.glob("*.tmp"):
            if path.is_file():
                relative = self._workspace.relative_to_workspace(job_id, path)
                if relative not in removed:
                    removed.append(relative)
                path.unlink(missing_ok=True)
        return removed

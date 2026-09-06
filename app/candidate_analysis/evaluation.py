from __future__ import annotations

from collections import Counter

from app.candidate_analysis.models import Phase4EvaluationReport
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.candidate_analysis.utils import mean
from app.core.config import AppSettings
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.repository import FrameAnalysisRepository


class Phase4Evaluator:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        candidate_repository: CandidateAnalysisRepository,
        frame_repository: FrameAnalysisRepository,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._candidate_repository = candidate_repository
        self._frame_repository = frame_repository

    def evaluate(self, job_id: str, *, persist: bool = False) -> Phase4EvaluationReport:
        windows = self._candidate_repository.load_stability_windows(job_id)
        boundaries = self._candidate_repository.load_boundaries(job_id)
        generated = self._candidate_repository.load_generated_candidates(job_id)
        scored = self._candidate_repository.load_scored_candidates(job_id)
        ranked = self._candidate_repository.load_ranked_candidates(job_id)
        selections = self._candidate_repository.load_selections(job_id)
        sampling = self._frame_repository.load_sampling(job_id)
        valid_boundaries = [item for item in boundaries.boundaries if item.is_valid]
        valid_generated = [item for item in generated.candidates if item.is_valid]
        valid_scored = [item for item in scored.candidates if item.is_valid]
        primary_ranked = [item for item in ranked.candidates if item.selection_role.value == "PRIMARY"]
        boundary_types = Counter(item.boundary_type.value for item in valid_boundaries)
        candidate_types = Counter(item.candidate_type.value for item in valid_generated)
        winner_types = Counter(item.candidate_type.value for item in primary_ranked)
        duration = sampling.video_duration_seconds
        primary_density = len(primary_ranked) * 60.0 / duration if duration > 0 else 0.0
        generated_density = len(valid_generated) * 60.0 / duration if duration > 0 else 0.0
        ambiguity_ratio = (
            sum(item.is_ambiguous for item in selections.windows) / len(selections.windows)
            if selections.windows
            else 0.0
        )
        warnings: list[str] = []
        valid_windows = [item for item in windows.windows if item.is_valid]
        if valid_windows and sum(item.duration_seconds < 3.0 for item in valid_windows) / len(valid_windows) > 0.7:
            warnings.append("Most stability windows are very short")
        if primary_density > self._settings.phase4_high_primary_density_per_minute:
            warnings.append("Primary candidate density is unusually high")
        if ambiguity_ratio > self._settings.phase4_high_ambiguity_ratio:
            warnings.append("Most candidate windows are ambiguous")
        if primary_ranked:
            winner, count = winner_types.most_common(1)[0]
            if count / len(primary_ranked) > self._settings.phase4_winner_type_skew_ratio:
                warnings.append(f"Primary winners are heavily skewed toward {winner}")
        if valid_boundaries and mean([item.activity_drop_score for item in valid_boundaries]) < 0.10:
            warnings.append("Stable boundaries have weak average activity drop")
        if valid_generated and len(valid_generated) > self._settings.max_total_candidates * 0.9:
            warnings.append("Generated candidate count is close to the global safety cap")
        warnings.extend(scored.stats.warnings)
        warnings.extend(ranked.stats.warnings)
        report = Phase4EvaluationReport(
            lecture_duration_seconds=duration,
            stability_windows=len(valid_windows),
            valid_boundaries=len(valid_boundaries),
            generated_candidates=len(generated.candidates),
            valid_candidates=len(valid_generated),
            primary_candidates=len(primary_ranked),
            alternate_candidates=sum(item.selection_role.value == "ALTERNATE" for item in ranked.candidates),
            ambiguous_windows=sum(item.is_ambiguous for item in selections.windows),
            clear_winners=sum(item.clear_winner for item in selections.windows),
            primary_density_per_minute=primary_density,
            generated_density_per_minute=generated_density,
            boundary_type_distribution=dict(sorted(boundary_types.items())),
            candidate_type_distribution=dict(sorted(candidate_types.items())),
            winner_type_distribution=dict(sorted(winner_types.items())),
            mean_window_duration_seconds=mean([item.duration_seconds for item in valid_windows]),
            mean_boundary_score=mean([item.boundary_score for item in valid_boundaries]),
            mean_activity_drop=mean([item.activity_drop_score for item in valid_boundaries]),
            mean_visual_quality=mean([item.visual_quality_score for item in valid_scored]),
            mean_completeness=mean([item.completeness_heuristic_score for item in valid_scored]),
            mean_transition_risk=mean([item.transition_risk_score for item in valid_scored]),
            mean_primary_ranking_score=mean([item.ranking_score for item in primary_ranked]),
            mean_score_gap=mean([item.score_gap for item in selections.windows if item.primary_candidate_id]),
            ambiguity_ratio=ambiguity_ratio,
            warnings=list(dict.fromkeys(warnings)),
        )
        if persist:
            atomic_write_json(
                self._workspace.candidate_evaluation_path(job_id), report.model_dump(mode="json")
            )
        return report

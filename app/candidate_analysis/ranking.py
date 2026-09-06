from __future__ import annotations

import functools
import time
from collections import Counter, defaultdict
from typing import Callable

from app.candidate_analysis.models import (
    CandidateSourceType,
    CandidateType,
    RankedCandidate,
    RankedCandidatesManifest,
    RankingStats,
    ScoredCandidate,
    ScoredCandidatesManifest,
    SelectionRole,
    SelectionsManifest,
    StabilityWindowsManifest,
    WindowSelection,
)
from app.candidate_analysis.utils import clamp01, mean, median, normalized_weighted, percentile
from app.core.config import AppSettings
from app.core.exceptions import CandidateRankingError, JobCancelledError
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.fingerprints import stable_hash

CANDIDATE_RANKING_ALGORITHM_VERSION = "1"
ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def ranking_config_payload(settings: AppSettings) -> dict[str, object]:
    return {
        "algorithm_version": CANDIDATE_RANKING_ALGORITHM_VERSION,
        "weights": {
            "quality": settings.rank_quality_weight,
            "completeness": settings.rank_completeness_weight,
            "safety": settings.rank_safety_weight,
            "local_stability": settings.rank_local_stability_weight,
            "boundary": settings.rank_boundary_weight,
            "density": settings.rank_density_weight,
            "accumulation": settings.rank_accumulation_weight,
        },
        "top_candidates_per_window": settings.top_candidates_per_window,
        "min_primary_score": settings.min_primary_ranking_score,
        "min_alternate_score": settings.min_alternate_ranking_score,
        "ambiguous_gap": settings.ambiguous_ranking_gap,
        "tie_epsilon": settings.ranking_tie_epsilon,
        "tie_policy": "safety_quality_completeness_stability_safe_position_earlier",
    }


def ranking_config_fingerprint(settings: AppSettings) -> str:
    return stable_hash(ranking_config_payload(settings))


def ranked_artifact_fingerprint(manifest: RankedCandidatesManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "scored_candidates_fingerprint": manifest.scored_candidates_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "stats": manifest.stats.model_dump(mode="json"),
            "candidates": [item.model_dump(mode="json") for item in manifest.candidates],
        }
    )


def selections_artifact_fingerprint(manifest: SelectionsManifest) -> str:
    return stable_hash(
        {
            "algorithm_version": manifest.algorithm_version,
            "ranked_candidates_fingerprint": manifest.ranked_candidates_fingerprint,
            "scored_candidates_fingerprint": manifest.scored_candidates_fingerprint,
            "stability_windows_fingerprint": manifest.stability_windows_fingerprint,
            "config_fingerprint": manifest.config_fingerprint,
            "stats": manifest.stats.model_dump(mode="json"),
            "windows": [item.model_dump(mode="json") for item in manifest.windows],
        }
    )


def _type_prior(candidate_type: CandidateType) -> float:
    return {
        CandidateType.SETTLED_START: 0.006,
        CandidateType.EARLY_STABLE: 0.002,
        CandidateType.MID_STABLE: 0.010,
        CandidateType.LATE_STABLE: 0.008,
        CandidateType.PRE_EXIT: 0.000,
    }[candidate_type]


def _source_prior(source_type: CandidateSourceType) -> float:
    return 0.008 if source_type is CandidateSourceType.BOUNDARY else 0.0


def _score(item: ScoredCandidate, settings: AppSettings) -> float:
    base = normalized_weighted(
        [
            (item.visual_quality_score, settings.rank_quality_weight),
            (item.completeness_heuristic_score, settings.rank_completeness_weight),
            (item.transition_safety_score, settings.rank_safety_weight),
            (item.local_stability_score, settings.rank_local_stability_weight),
            (item.boundary_score, settings.rank_boundary_weight),
            (item.content_density_score, settings.rank_density_weight),
            (item.content_accumulation_score, settings.rank_accumulation_weight),
        ]
    )
    return clamp01(base + _type_prior(item.candidate_type) + _source_prior(item.source_type))


def _compare(a: tuple[ScoredCandidate, float], b: tuple[ScoredCandidate, float], epsilon: float) -> int:
    item_a, score_a = a
    item_b, score_b = b
    if abs(score_a - score_b) > epsilon:
        return -1 if score_a > score_b else 1
    secondary = (
        (item_a.transition_safety_score, item_b.transition_safety_score),
        (item_a.visual_quality_score, item_b.visual_quality_score),
        (item_a.completeness_heuristic_score, item_b.completeness_heuristic_score),
        (item_a.local_stability_score, item_b.local_stability_score),
    )
    for value_a, value_b in secondary:
        if abs(value_a - value_b) > epsilon:
            return -1 if value_a > value_b else 1
    target = 0.65
    distance_a = abs(item_a.relative_position - target)
    distance_b = abs(item_b.relative_position - target)
    if abs(distance_a - distance_b) > epsilon:
        return -1 if distance_a < distance_b else 1
    timestamp_a = item_a.frame_timestamp_seconds or 0.0
    timestamp_b = item_b.frame_timestamp_seconds or 0.0
    if abs(timestamp_a - timestamp_b) > epsilon:
        return -1 if timestamp_a < timestamp_b else 1
    return -1 if item_a.candidate_id < item_b.candidate_id else (1 if item_a.candidate_id > item_b.candidate_id else 0)


class CandidateRankingService:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace = workspace

    def process(
        self,
        *,
        job_id: str,
        scored: ScoredCandidatesManifest,
        windows: StabilityWindowsManifest,
        progress_callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> tuple[RankedCandidatesManifest, SelectionsManifest]:
        started = time.monotonic()
        all_by_window: dict[int, list[ScoredCandidate]] = defaultdict(list)
        valid_by_window: dict[int, list[ScoredCandidate]] = defaultdict(list)
        for item in scored.candidates:
            all_by_window[item.stable_window_id].append(item)
            if item.is_valid and item.frame_timestamp_seconds is not None:
                valid_by_window[item.stable_window_id].append(item)

        valid_windows = sorted(
            (window for window in windows.windows if window.is_valid),
            key=lambda window: (window.start_timestamp_seconds, window.window_id),
        )
        ranked: list[RankedCandidate] = []
        selections: list[WindowSelection] = []
        winner_types: Counter[str] = Counter()
        primary_scores: list[float] = []
        gaps: list[float] = []

        for offset, window in enumerate(valid_windows, start=1):
            if cancel_check():
                raise JobCancelledError("Candidate ranking was cancelled.")
            candidates = valid_by_window.get(window.window_id, [])
            scored_pairs = [(item, _score(item, self._settings)) for item in candidates]
            scored_pairs.sort(
                key=functools.cmp_to_key(
                    lambda a, b: _compare(a, b, self._settings.ranking_tie_epsilon)
                )
            )
            primary_id: int | None = None
            alternate_ids: list[int] = []
            primary_score = 0.0
            second_score: float | None = None
            score_gap = 0.0
            failure_reason: str | None = None
            boundary_id = candidates[0].boundary_id if candidates else None

            if scored_pairs and scored_pairs[0][1] >= self._settings.min_primary_ranking_score:
                primary_id = scored_pairs[0][0].candidate_id
                primary_score = scored_pairs[0][1]
                primary_scores.append(primary_score)
                winner_types[scored_pairs[0][0].candidate_type.value] += 1
                if len(scored_pairs) > 1:
                    second_score = scored_pairs[1][1]
                    score_gap = clamp01(primary_score - second_score)
                    gaps.append(score_gap)
                else:
                    score_gap = primary_score
                    gaps.append(score_gap)
                for item, score in scored_pairs[1 : self._settings.top_candidates_per_window]:
                    if score >= self._settings.min_alternate_ranking_score:
                        alternate_ids.append(item.candidate_id)
            else:
                failure_reason = "NO_VALID_CANDIDATES" if not scored_pairs else "ALL_CANDIDATES_BELOW_MINIMUM"

            ambiguous = bool(
                primary_id is not None
                and second_score is not None
                and score_gap <= self._settings.ambiguous_ranking_gap
            )
            clear_winner = bool(
                primary_id is not None
                and primary_score >= 0.60
                and (second_score is None or score_gap > self._settings.ambiguous_ranking_gap * 2.0)
            )
            confidence = 0.0
            if primary_id is not None:
                gap_component = clamp01(score_gap / max(self._settings.ambiguous_ranking_gap * 4.0, 1e-6))
                confidence = normalized_weighted([(primary_score, 0.70), (gap_component, 0.30)])

            for rank, (item, score) in enumerate(scored_pairs, start=1):
                role = SelectionRole.NONE
                if item.candidate_id == primary_id:
                    role = SelectionRole.PRIMARY
                elif item.candidate_id in alternate_ids:
                    role = SelectionRole.ALTERNATE
                ranked.append(
                    RankedCandidate(
                        candidate_id=item.candidate_id,
                        stable_window_id=item.stable_window_id,
                        ranking_score=score,
                        rank_within_window=rank,
                        selection_role=role,
                        score_gap_from_best=clamp01(primary_score - score) if primary_id is not None else 0.0,
                        candidate_type=item.candidate_type,
                        source_type=item.source_type,
                        frame_timestamp_seconds=item.frame_timestamp_seconds or 0.0,
                        relative_position=item.relative_position,
                        visual_quality_score=item.visual_quality_score,
                        completeness_heuristic_score=item.completeness_heuristic_score,
                        transition_safety_score=item.transition_safety_score,
                        local_stability_score=item.local_stability_score,
                        boundary_score=item.boundary_score,
                        content_density_score=item.content_density_score,
                        content_accumulation_score=item.content_accumulation_score,
                    )
                )

            selections.append(
                WindowSelection(
                    stable_window_id=window.window_id,
                    boundary_id=boundary_id,
                    candidate_count=len(all_by_window.get(window.window_id, [])),
                    valid_candidate_count=len(candidates),
                    primary_candidate_id=primary_id,
                    alternate_candidate_ids=alternate_ids,
                    primary_score=primary_score,
                    second_best_score=second_score,
                    score_gap=score_gap,
                    selection_confidence=confidence,
                    is_ambiguous=ambiguous,
                    clear_winner=clear_winner,
                    failure_reason=failure_reason,
                )
            )
            if valid_windows:
                progress_callback(100.0 * offset / len(valid_windows))

        ranking_values = [item.ranking_score for item in ranked]
        ambiguity_count = sum(selection.is_ambiguous for selection in selections)
        clear_count = sum(selection.clear_winner for selection in selections)
        primary_count = sum(selection.primary_candidate_id is not None for selection in selections)
        alternate_count = sum(len(selection.alternate_candidate_ids) for selection in selections)
        warnings: list[str] = []
        if selections and ambiguity_count / len(selections) > self._settings.phase4_high_ambiguity_ratio:
            warnings.append("Candidate ranking is ambiguous for most stable windows")
        if primary_count:
            most_common = winner_types.most_common(1)
            if most_common and most_common[0][1] / primary_count > self._settings.phase4_winner_type_skew_ratio:
                warnings.append(f"Primary selection is heavily skewed toward {most_common[0][0]}")
        if len(ranking_values) > 2 and max(ranking_values) - min(ranking_values) < 0.02:
            warnings.append("Candidate ranking scores are nearly uniform")

        stats = RankingStats(
            windows_considered=len(valid_windows),
            windows_with_primary=primary_count,
            primary_candidates=primary_count,
            alternate_candidates=alternate_count,
            ambiguous_windows=ambiguity_count,
            clear_winner_windows=clear_count,
            windows_without_candidate=len(valid_windows) - primary_count,
            mean_ranking_score=mean(ranking_values),
            median_ranking_score=median(ranking_values),
            p90_ranking_score=percentile(ranking_values, 90),
            mean_score_gap=mean(gaps),
            ambiguity_ratio=(ambiguity_count / len(valid_windows) if valid_windows else 0.0),
            winner_type_distribution=dict(sorted(winner_types.items())),
            warnings=warnings,
        )
        config_fingerprint = ranking_config_fingerprint(self._settings)
        ranked_manifest = RankedCandidatesManifest(
            algorithm_version=CANDIDATE_RANKING_ALGORITHM_VERSION,
            scored_candidates_fingerprint=scored.artifact_fingerprint,
            config_fingerprint=config_fingerprint,
            artifact_fingerprint="pending",
            stats=stats,
            candidates=ranked,
            ranking_seconds=round(time.monotonic() - started, 3),
        )
        ranked_manifest.artifact_fingerprint = ranked_artifact_fingerprint(ranked_manifest)
        selections_manifest = SelectionsManifest(
            algorithm_version=CANDIDATE_RANKING_ALGORITHM_VERSION,
            ranked_candidates_fingerprint=ranked_manifest.artifact_fingerprint,
            scored_candidates_fingerprint=scored.artifact_fingerprint,
            stability_windows_fingerprint=windows.artifact_fingerprint,
            config_fingerprint=config_fingerprint,
            artifact_fingerprint="pending",
            stats=stats,
            windows=selections,
        )
        selections_manifest.artifact_fingerprint = selections_artifact_fingerprint(selections_manifest)
        atomic_write_json(self._workspace.ranked_candidates_path(job_id), ranked_manifest.model_dump(mode="json"))
        atomic_write_json(self._workspace.candidate_selections_path(job_id), selections_manifest.model_dump(mode="json"))
        progress_callback(100.0)
        return ranked_manifest, selections_manifest

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable

from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, SemanticDecisionError
from app.jobs.checkpoints import CheckpointStore
from app.semantic_analysis.models import (
    CandidateDecision,
    CompletionState,
    EducationalUsefulness,
    SEMANTIC_DECISION_ALGORITHM_VERSION,
    SemanticCandidateDecision,
    SemanticSelectionsManifest,
    SemanticSelectionStats,
    SemanticWindowDecision,
    TranscriptVisualConsistency,
    VisualChangeState,
    WindowDecisionType,
)
from app.semantic_analysis.repository import SemanticRepository
from app.video_analysis.fingerprints import stable_hash

CP_SEMANTIC_CANDIDATES_READY = "SEMANTIC_CANDIDATES_READY"
CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[int], None]


class SemanticDecisionEngine:
    def __init__(
        self,
        settings: AppSettings,
        checkpoints: CheckpointStore,
        semantic_repository: SemanticRepository,
        candidate_repository: CandidateAnalysisRepository,
    ) -> None:
        self._settings = settings
        self._checkpoints = checkpoints
        self._semantic = semantic_repository
        self._candidates = candidate_repository

    def config_fingerprint(self) -> str:
        return stable_hash(
            {
                "algorithm_version": SEMANTIC_DECISION_ALGORITHM_VERSION,
                "weights": self._weights(),
                "min_score": self._settings.min_semantic_decision_score,
                "min_confidence": self._settings.min_semantic_confidence,
                "alternate_switch_margin": self._settings.semantic_alternate_switch_margin,
                "primary_tie_bonus": self._settings.semantic_primary_tie_bonus,
                "ambiguity_gap": self._settings.semantic_decision_ambiguity_gap,
                "transition_reject_threshold": self._settings.semantic_transition_reject_threshold,
                "mappings": "decision-mappings-v1",
            }
        )

    def _weights(self) -> dict[str, float]:
        return {
            "completion": self._settings.semantic_decision_completion_weight,
            "usefulness": self._settings.semantic_decision_usefulness_weight,
            "settled": self._settings.semantic_decision_settled_weight,
            "confidence": self._settings.semantic_decision_confidence_weight,
            "transcript": self._settings.semantic_decision_transcript_weight,
            "phase4": self._settings.semantic_decision_phase4_weight,
        }

    @staticmethod
    def _completion(value: CompletionState) -> float:
        return {
            CompletionState.COMPLETE: 1.0,
            CompletionState.MOSTLY_COMPLETE: 0.8,
            CompletionState.UNCERTAIN: 0.5,
            CompletionState.INCOMPLETE: 0.2,
            CompletionState.TRANSITION: 0.0,
        }[value]

    @staticmethod
    def _usefulness(value: EducationalUsefulness) -> float:
        return {
            EducationalUsefulness.HIGH: 1.0,
            EducationalUsefulness.MEDIUM: 0.65,
            EducationalUsefulness.LOW: 0.3,
            EducationalUsefulness.NONE: 0.0,
        }[value]

    @staticmethod
    def _settled(value: VisualChangeState) -> float:
        return {
            VisualChangeState.SETTLED: 1.0,
            VisualChangeState.MOSTLY_SETTLED: 0.75,
            VisualChangeState.UNCERTAIN: 0.45,
            VisualChangeState.STILL_CHANGING: 0.0,
        }[value]

    @staticmethod
    def _transcript(value: TranscriptVisualConsistency) -> float:
        return {
            TranscriptVisualConsistency.CONSISTENT: 1.0,
            TranscriptVisualConsistency.PARTIALLY_CONSISTENT: 0.7,
            TranscriptVisualConsistency.NO_TRANSCRIPT: 0.5,
            TranscriptVisualConsistency.UNCERTAIN: 0.5,
            TranscriptVisualConsistency.UNRELATED: 0.2,
        }[value]

    def _score(self, semantic_result, ranking_score: float) -> float:
        w = self._weights()
        score = (
            w["completion"] * self._completion(semantic_result.completion_state)
            + w["usefulness"] * self._usefulness(semantic_result.educational_usefulness)
            + w["settled"] * self._settled(semantic_result.visual_change_state)
            + w["confidence"] * semantic_result.confidence
            + w["transcript"] * self._transcript(semantic_result.transcript_visual_consistency)
            + w["phase4"] * ranking_score
        )
        return min(1.0, max(0.0, score))

    def _gate(self, result, score: float) -> tuple[bool, CandidateDecision, list[str]]:
        if result.completion_state is CompletionState.TRANSITION or result.transition_probability >= self._settings.semantic_transition_reject_threshold:
            return False, CandidateDecision.REJECT_TRANSITION, ["SEMANTIC_TRANSITION"]
        if not result.has_meaningful_visual_content or result.educational_usefulness is EducationalUsefulness.NONE:
            return False, CandidateDecision.REJECT_LOW_VALUE, ["SEMANTIC_LOW_USEFULNESS"]
        if result.completion_state is CompletionState.INCOMPLETE:
            return False, CandidateDecision.REJECT_INCOMPLETE, ["SEMANTIC_INCOMPLETE"]
        if result.confidence < self._settings.min_semantic_confidence:
            return False, CandidateDecision.REJECT_LOW_CONFIDENCE, ["SEMANTIC_LOW_CONFIDENCE"]
        if score < self._settings.min_semantic_decision_score:
            return False, CandidateDecision.REJECT_LOW_VALUE, ["SEMANTIC_SCORE_BELOW_THRESHOLD"]
        reasons = []
        if result.completion_state is CompletionState.COMPLETE:
            reasons.append("SEMANTIC_COMPLETE")
        elif result.completion_state is CompletionState.MOSTLY_COMPLETE:
            reasons.append("SEMANTIC_MOSTLY_COMPLETE")
        if result.educational_usefulness is EducationalUsefulness.HIGH:
            reasons.append("SEMANTIC_HIGH_USEFULNESS")
        return True, CandidateDecision.KEEP, reasons

    def process(self, job_id: str, *, progress_callback: ProgressCallback | None = None, cancel_check: CancelCheck | None = None) -> SemanticSelectionsManifest:
        results_manifest = self._semantic.load_results(job_id)
        inputs_manifest = self._semantic.load_input_manifest(job_id)
        ranked = self._candidates.load_ranked_candidates(job_id)
        selections = self._candidates.load_selections(job_id)
        results = {x.candidate_id: x for x in results_manifest.results}
        inputs = {x.candidate_id: x for x in inputs_manifest.inputs}
        ranked_by_id = {x.candidate_id: x for x in ranked.candidates}
        candidate_decisions: list[SemanticCandidateDecision] = []
        decision_by_id: dict[int, SemanticCandidateDecision] = {}

        ordered_inputs = sorted(inputs.values(), key=lambda x: (x.stable_window_id, x.candidate_timestamp_seconds, x.candidate_id))
        for ordinal, semantic_input in enumerate(ordered_inputs):
            if cancel_check and cancel_check():
                raise JobCancelledError(f"Job {job_id} was cancelled")
            result = results.get(semantic_input.candidate_id)
            ranking = ranked_by_id.get(semantic_input.candidate_id)
            if result is None or ranking is None:
                raise SemanticDecisionError("Semantic decision references are incomplete.")
            score = self._score(result, ranking.ranking_score)
            eligible, decision, reasons = self._gate(result, score)
            record = SemanticCandidateDecision(
                candidate_id=semantic_input.candidate_id,
                stable_window_id=semantic_input.stable_window_id,
                selection_role=semantic_input.selection_role,
                eligible=eligible,
                semantic_decision_score=score,
                decision=decision,
                reason_codes=reasons,
            )
            candidate_decisions.append(record)
            decision_by_id[record.candidate_id] = record
            if progress_callback:
                progress_callback(int(((ordinal + 1) / max(1, len(ordered_inputs))) * 75))

        window_decisions: list[SemanticWindowDecision] = []
        selected_ids: list[int] = []
        for selection in sorted(selections.windows, key=lambda x: x.stable_window_id):
            ids = []
            if selection.primary_candidate_id is not None and selection.primary_candidate_id in decision_by_id:
                ids.append(selection.primary_candidate_id)
            ids.extend(x for x in selection.alternate_candidate_ids if x in decision_by_id)
            if not ids:
                # Phase-4 window may legitimately have no retained semantic candidate.
                continue
            primary_id = selection.primary_candidate_id if selection.primary_candidate_id in decision_by_id else None
            alt_ids = [x for x in selection.alternate_candidate_ids if x in decision_by_id]
            alternate_id = alt_ids[0] if alt_ids else None
            eligible_ids = [x for x in ids if decision_by_id[x].eligible]
            if not eligible_ids:
                window_decisions.append(
                    SemanticWindowDecision(
                        stable_window_id=selection.stable_window_id,
                        primary_candidate_id=primary_id,
                        alternate_candidate_id=alternate_id,
                        selected_candidate_id=None,
                        decision=WindowDecisionType.REJECT_WINDOW,
                        reason_codes=["BOTH_CANDIDATES_REJECTED"] if len(ids) > 1 else ["CANDIDATE_REJECTED"],
                    )
                )
                continue
            if len(eligible_ids) == 1:
                winner = eligible_ids[0]
                if len(ids) == 1:
                    decision_type = WindowDecisionType.SELECT_SINGLE
                else:
                    decision_type = WindowDecisionType.SELECT_PRIMARY if winner == primary_id else WindowDecisionType.SELECT_ALTERNATE
                decision_by_id[winner].decision = CandidateDecision.KEEP
                decision_by_id[winner].reason_codes.append("ONLY_ELIGIBLE_CANDIDATE")
                selected_ids.append(winner)
                window_decisions.append(
                    SemanticWindowDecision(
                        stable_window_id=selection.stable_window_id,
                        primary_candidate_id=primary_id,
                        alternate_candidate_id=alternate_id,
                        selected_candidate_id=winner,
                        decision=decision_type,
                        reason_codes=["ONLY_ELIGIBLE_CANDIDATE"],
                    )
                )
                continue

            primary_score = decision_by_id[primary_id].semantic_decision_score if primary_id in eligible_ids else -1.0
            best_alt = max((x for x in eligible_ids if x != primary_id), key=lambda cid: (decision_by_id[cid].semantic_decision_score, -cid), default=None)
            alt_score = decision_by_id[best_alt].semantic_decision_score if best_alt is not None else -1.0
            raw_gap = abs(primary_score - alt_score) if primary_id in eligible_ids and best_alt is not None else 1.0
            primary_compare_score = primary_score
            tie_bonus_applied = False
            if (
                primary_id in eligible_ids
                and best_alt is not None
                and raw_gap <= self._settings.semantic_decision_ambiguity_gap
            ):
                primary_compare_score = min(1.0, primary_score + self._settings.semantic_primary_tie_bonus)
                tie_bonus_applied = self._settings.semantic_primary_tie_bonus > 0
            if primary_id in eligible_ids and (
                best_alt is None
                or alt_score < primary_compare_score + self._settings.semantic_alternate_switch_margin
            ):
                winner = primary_id
                decision_type = WindowDecisionType.SELECT_PRIMARY
                reason = "PRIMARY_REMAINS_STRONGER"
            else:
                assert best_alt is not None
                winner = best_alt
                decision_type = WindowDecisionType.SELECT_ALTERNATE
                reason = "ALTERNATE_OUTSCORES_PRIMARY"
            sorted_scores = sorted((decision_by_id[x].semantic_decision_score for x in eligible_ids), reverse=True)
            gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            ambiguous = gap <= self._settings.semantic_decision_ambiguity_gap
            for cid in eligible_ids:
                if cid == winner:
                    decision_by_id[cid].decision = CandidateDecision.KEEP
                    decision_by_id[cid].reason_codes.append(reason)
                else:
                    decision_by_id[cid].decision = CandidateDecision.REVIEW_ALTERNATE
                    decision_by_id[cid].reason_codes.append("SEMANTIC_TIE" if ambiguous else "NOT_SELECTED_AFTER_COMPARISON")
            selected_ids.append(winner)
            window_decisions.append(
                SemanticWindowDecision(
                    stable_window_id=selection.stable_window_id,
                    primary_candidate_id=primary_id,
                    alternate_candidate_id=best_alt,
                    selected_candidate_id=winner,
                    decision=decision_type,
                    score_gap=gap,
                    is_semantically_ambiguous=ambiguous,
                    reason_codes=[reason]
                    + (["PRIMARY_TIE_BONUS"] if tie_bonus_applied and winner == primary_id else [])
                    + (["SEMANTIC_TIE"] if ambiguous else []),
                )
            )

        candidate_decisions.sort(key=lambda x: (x.stable_window_id, x.candidate_id))
        window_decisions.sort(key=lambda x: x.stable_window_id)
        selected_scores = [decision_by_id[x].semantic_decision_score for x in selected_ids]
        windows_with_alternates = sum(bool(x.alternate_candidate_ids) for x in selections.windows)
        primary_count = sum(x.decision is WindowDecisionType.SELECT_PRIMARY for x in window_decisions)
        alternate_count = sum(x.decision is WindowDecisionType.SELECT_ALTERNATE for x in window_decisions)
        rejected_count = sum(x.decision is WindowDecisionType.REJECT_WINDOW for x in window_decisions)
        ambiguous_count = sum(x.is_semantically_ambiguous for x in window_decisions)
        warnings: list[str] = []
        if window_decisions and rejected_count / len(window_decisions) >= 0.8:
            warnings.append("HIGH_SEMANTIC_REJECTION_RATE")
        if window_decisions and rejected_count / len(window_decisions) <= 0.02:
            warnings.append("LOW_SEMANTIC_SELECTIVITY")
        if window_decisions and ambiguous_count / len(window_decisions) >= 0.5:
            warnings.append("HIGH_SEMANTIC_AMBIGUITY_RATE")
        stats = SemanticSelectionStats(
            window_count=len(window_decisions),
            selected_window_count=len(selected_ids),
            rejected_window_count=rejected_count,
            primary_selected_count=primary_count,
            alternate_selected_count=alternate_count,
            candidate_keep_count=sum(x.decision is CandidateDecision.KEEP for x in candidate_decisions),
            candidate_reject_count=sum(x.decision not in {CandidateDecision.KEEP, CandidateDecision.REVIEW_ALTERNATE} for x in candidate_decisions),
            semantic_ambiguous_window_count=ambiguous_count,
            rejected_incomplete_count=sum(x.decision is CandidateDecision.REJECT_INCOMPLETE for x in candidate_decisions),
            rejected_transition_count=sum(x.decision is CandidateDecision.REJECT_TRANSITION for x in candidate_decisions),
            rejected_low_value_count=sum(x.decision is CandidateDecision.REJECT_LOW_VALUE for x in candidate_decisions),
            rejected_low_confidence_count=sum(x.decision is CandidateDecision.REJECT_LOW_CONFIDENCE for x in candidate_decisions),
            mean_selected_semantic_score=statistics.fmean(selected_scores) if selected_scores else 0.0,
            median_selected_semantic_score=statistics.median(selected_scores) if selected_scores else 0.0,
            alternate_switch_rate=(alternate_count / windows_with_alternates) if windows_with_alternates else 0.0,
            warnings=warnings,
        )
        config_fp = self.config_fingerprint()
        payload = {
            "semantic_results_fingerprint": results_manifest.artifact_fingerprint,
            "ranked_candidates_fingerprint": ranked.artifact_fingerprint,
            "candidate_selections_fingerprint": selections.artifact_fingerprint,
            "config_fingerprint": config_fp,
            "candidate_decisions": [x.model_dump(mode="json") for x in candidate_decisions],
            "window_decisions": [x.model_dump(mode="json") for x in window_decisions],
            "selected_candidate_ids": selected_ids,
        }
        manifest = SemanticSelectionsManifest(
            semantic_results_fingerprint=results_manifest.artifact_fingerprint,
            ranked_candidates_fingerprint=ranked.artifact_fingerprint,
            candidate_selections_fingerprint=selections.artifact_fingerprint,
            config_fingerprint=config_fp,
            artifact_fingerprint=stable_hash(payload),
            stats=stats,
            candidate_decisions=candidate_decisions,
            window_decisions=window_decisions,
            selected_candidate_ids=selected_ids,
        )
        self._semantic.save_selections(job_id, manifest)
        if progress_callback:
            progress_callback(100)
        return manifest

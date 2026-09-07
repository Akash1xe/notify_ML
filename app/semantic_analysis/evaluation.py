from __future__ import annotations

from collections import Counter

from app.core.config import AppSettings
from app.semantic_analysis.models import Phase6EvaluationReport, TemporalContextType
from app.semantic_analysis.repository import SemanticRepository


class Phase6Evaluator:
    def __init__(self, settings: AppSettings, repository: SemanticRepository) -> None:
        self._settings = settings
        self._repository = repository

    def evaluate(self, job_id: str, *, persist: bool = True) -> Phase6EvaluationReport:
        inputs = self._repository.load_input_manifest(job_id)
        temporal = self._repository.load_temporal_contexts(job_id)
        results = self._repository.load_results(job_id)
        selections = self._repository.load_selections(job_id)
        completion = Counter(x.completion_state.value for x in results.results)
        usefulness = Counter(x.educational_usefulness.value for x in results.results)
        content_type = Counter(x.content_type.value for x in results.results)
        context_types = Counter(x.context_type for x in temporal.contexts)
        window_count = selections.stats.window_count
        rejected_ratio = selections.stats.rejected_window_count / window_count if window_count else 0.0
        fallback_ratio = results.stats.fallback_candidate_count / results.stats.candidate_count if results.stats.candidate_count else 0.0
        current_only_ratio = context_types[TemporalContextType.CURRENT_ONLY] / len(temporal.contexts) if temporal.contexts else 0.0
        warnings = list(dict.fromkeys(inputs.stats.warnings + temporal.stats.warnings + results.stats.warnings + selections.stats.warnings))
        if window_count and rejected_ratio >= self._settings.phase6_high_rejection_ratio:
            warnings.append("HIGH_SEMANTIC_REJECTION_RATE")
        if window_count and rejected_ratio <= self._settings.phase6_low_rejection_ratio:
            warnings.append("LOW_SEMANTIC_SELECTIVITY")
        if fallback_ratio >= self._settings.phase6_high_fallback_ratio and results.stats.candidate_count:
            warnings.append("HIGH_VLM_FALLBACK_RATE")
        if current_only_ratio >= self._settings.phase6_high_current_only_ratio and temporal.contexts:
            warnings.append("HIGH_CURRENT_ONLY_CONTEXT_RATIO")
        if len(results.stats.model_usage_counts) > 1:
            warnings.append("MIXED_VLM_MODELS")
        report = Phase6EvaluationReport(
            semantic_input_count=inputs.stats.input_count,
            primary_input_count=inputs.stats.primary_input_count,
            alternate_input_count=inputs.stats.alternate_input_count,
            triplet_count=context_types[TemporalContextType.TRIPLET],
            current_only_count=context_types[TemporalContextType.CURRENT_ONLY],
            completion_distribution=dict(completion),
            usefulness_distribution=dict(usefulness),
            content_type_distribution=dict(content_type),
            selected_windows=selections.stats.selected_window_count,
            rejected_windows=selections.stats.rejected_window_count,
            primary_selected=selections.stats.primary_selected_count,
            alternate_selected=selections.stats.alternate_selected_count,
            alternate_switch_rate=selections.stats.alternate_switch_rate,
            mean_confidence=results.stats.mean_confidence,
            mean_inference_seconds=results.stats.mean_inference_seconds,
            fallback_candidate_count=results.stats.fallback_candidate_count,
            cache_hit_ratio=(results.stats.cached_analysis_count / results.stats.candidate_count if results.stats.candidate_count else 0.0),
            warnings=list(dict.fromkeys(warnings)),
        )
        if persist:
            self._repository.save_evaluation(job_id, report)
        return report

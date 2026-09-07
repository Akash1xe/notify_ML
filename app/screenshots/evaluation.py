from __future__ import annotations

import statistics
from collections import Counter

from app.core.config import AppSettings
from app.screenshots.models import Phase7EvaluationReport, PairDecision, ScreenshotQualityState
from app.screenshots.repository import ScreenshotRepository
from app.semantic_analysis.repository import SemanticRepository


class Phase7Evaluator:
    def __init__(self,settings:AppSettings,repository:ScreenshotRepository,semantic_repository:SemanticRepository)->None:
        self._settings=settings; self._repository=repository; self._semantic=semantic_repository

    def evaluate(self,job_id:str,*,persist:bool=True)->Phase7EvaluationReport:
        quality=self._repository.load_quality_manifest(job_id); pairs=self._repository.load_duplicate_pairs(job_id); groups=self._repository.load_duplicate_groups(job_id); final=self._repository.load_final_selections(job_id)
        qdist=Counter(r.quality_state.value for r in quality.screenshots); pdist=Counter(p.decision.value for p in pairs.pairs)
        shifts=[abs(r.timestamp_shift_seconds) for r in quality.screenshots]
        handoff={r.candidate_id:r for r in self._semantic.load_phase7_handoff(job_id)}
        quality_overrides=0
        for group in final.groups:
            semantic_best=max(group.members,key=lambda m:(m.semantic_decision_score,-m.candidate_id)) if group.members else None
            if semantic_best and semantic_best.candidate_id!=group.winner_candidate_id: quality_overrides+=1
        total_candidates=len(quality.screenshots); cache_hits=quality.stats.cached_quality_count
        warnings=list(dict.fromkeys(quality.stats.warnings+groups.stats.warnings+final.stats.warnings))
        if total_candidates and quality.stats.fallback_selected_count/total_candidates>self._settings.phase7_high_fallback_ratio and 'HIGH_SCREENSHOT_FALLBACK_RATE' not in warnings: warnings.append('HIGH_SCREENSHOT_FALLBACK_RATE')
        if groups.stats.duplicate_ratio>self._settings.phase7_high_duplicate_ratio and 'HIGH_DUPLICATE_RATIO' not in warnings: warnings.append('HIGH_DUPLICATE_RATIO')
        report=Phase7EvaluationReport(semantic_candidate_count=len(self._semantic.load_phase7_handoff(job_id)),final_screenshot_count=len(final.final_screenshots),quality_state_distribution=dict(qdist),exact_retained_count=sum(not r.used_timestamp_fallback for r in quality.screenshots),refined_timestamp_count=sum(r.used_timestamp_fallback for r in quality.screenshots),mean_absolute_timestamp_shift=statistics.mean(shifts) if shifts else 0,duplicate_pair_distribution=dict(pdist),duplicate_group_count=len(groups.groups),max_duplicate_group_size=max((g.group_size for g in groups.groups),default=0),dedup_ratio=final.stats.final_dedup_ratio,mean_final_quality_score=final.stats.mean_winner_quality_score,mean_final_semantic_score=final.stats.mean_winner_semantic_score,quality_override_count=quality_overrides,cache_hit_ratio=cache_hits/total_candidates if total_candidates else 0,warnings=warnings)
        if persist: self._repository.save_evaluation(job_id,report)
        return report

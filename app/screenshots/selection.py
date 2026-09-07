from __future__ import annotations

import statistics
from typing import Callable

from app.core.config import AppSettings
from app.core.exceptions import FinalScreenshotSelectionError, JobCancelledError
from app.jobs.checkpoints import CheckpointStore
from app.screenshots.models import (
    FINAL_SCREENSHOT_SELECTION_ALGORITHM_VERSION,
    DuplicateGroupsManifest,
    FinalGroupSelection,
    FinalMemberDecision,
    FinalScreenshotRecord,
    FinalSelectionMember,
    FinalSelectionsManifest,
    FinalSelectionStats,
    QualityManifest,
    ScreenshotQualityState,
)
from app.screenshots.repository import ScreenshotRepository
from app.screenshots.utils import canonical_fingerprint, file_sha256, resolve_job_path
from app.semantic_analysis.models import CompletionState
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager

CP_FINAL_SCREENSHOT_SELECTION_READY='FINAL_SCREENSHOT_SELECTION_READY'


class FinalScreenshotSelector:
    def __init__(self,settings:AppSettings,workspace:WorkspaceManager,checkpoints:CheckpointStore,repository:ScreenshotRepository,semantic_repository:SemanticRepository)->None:
        self._settings=settings; self._workspace=workspace; self._checkpoints=checkpoints; self._repository=repository; self._semantic=semantic_repository

    def config_fingerprint(self)->str:
        s=self._settings
        return canonical_fingerprint({'algorithm':FINAL_SCREENSHOT_SELECTION_ALGORITHM_VERSION,'weights':[s.final_selection_semantic_weight,s.final_selection_quality_weight,s.final_selection_completion_weight,s.final_selection_confidence_weight,s.final_selection_sharpness_weight,s.final_selection_timestamp_weight],'exact_bonus':s.final_selection_exact_timestamp_bonus,'epsilon':s.final_selection_tie_epsilon,'max_semantic_deficit':s.final_selection_max_semantic_deficit_for_quality_override})

    @staticmethod
    def _completion_score(state:CompletionState)->float:
        return {CompletionState.COMPLETE:1.0,CompletionState.MOSTLY_COMPLETE:0.8,CompletionState.UNCERTAIN:0.5,CompletionState.INCOMPLETE:0.2,CompletionState.TRANSITION:0.0}[state]

    def _semantic_dependency(self,job_id:str)->str:
        results=self._semantic.load_results(job_id); selections=self._semantic.load_selections(job_id)
        return canonical_fingerprint({'results':results.artifact_fingerprint,'selections':selections.artifact_fingerprint})

    def process(self,job_id:str,groups_manifest:DuplicateGroupsManifest,quality_manifest:QualityManifest,*,cancel_check:Callable[[],bool]|None=None)->FinalSelectionsManifest:
        qualities={r.candidate_id:r for r in quality_manifest.screenshots}
        handoff={r.candidate_id:r for r in self._semantic.load_phase7_handoff(job_id)}
        semantic_results={r.candidate_id:r for r in self._semantic.load_results(job_id).results}
        group_selections=[]; winner_rows=[]; invalid_members=0; quality_overrides=0
        for group in groups_manifest.groups:
            if cancel_check and cancel_check(): raise JobCancelledError('Final screenshot selection cancelled.')
            members=[]
            for cid in group.candidate_ids:
                q=qualities.get(cid); h=handoff.get(cid); sr=semantic_results.get(cid)
                if q is None or h is None or sr is None:
                    raise FinalScreenshotSelectionError('Duplicate group references incomplete upstream candidate metadata.')
                path=resolve_job_path(self._workspace,job_id,q.selected_relative_path)
                eligible=q.quality_state is not ScreenshotQualityState.INVALID and sr.completion_state is not CompletionState.TRANSITION and file_sha256(path)==q.selected_file_sha256
                if not eligible: invalid_members+=1
                radius=max(self._settings.screenshot_quality_search_radius_seconds,1e-9)
                proximity=max(0.0,1.0-min(1.0,abs(q.timestamp_shift_seconds)/radius)) if self._settings.screenshot_quality_search_radius_seconds>0 else 1.0
                completion=self._completion_score(sr.completion_state)
                score=(h.semantic_decision_score*self._settings.final_selection_semantic_weight+q.selected_quality_score*self._settings.final_selection_quality_weight+completion*self._settings.final_selection_completion_weight+sr.confidence*self._settings.final_selection_confidence_weight+q.metrics.sharpness_score*self._settings.final_selection_sharpness_weight+proximity*self._settings.final_selection_timestamp_weight)
                if abs(q.timestamp_shift_seconds)<1e-7: score+=self._settings.final_selection_exact_timestamp_bonus
                score=max(0.0,min(1.0,score)) if eligible else 0.0
                reasons=[]
                if len(group.candidate_ids)==1: reasons.append('SINGLETON_GROUP')
                if sr.completion_state is CompletionState.COMPLETE: reasons.append('SEMANTICALLY_COMPLETE')
                if q.metrics.sharpness_score>=0.7: reasons.append('HIGHER_SHARPNESS')
                if not eligible: reasons.append('INVALID_SCREENSHOT')
                members.append(FinalSelectionMember(candidate_id=cid,eligible=eligible,final_selection_score=score,semantic_decision_score=h.semantic_decision_score,quality_score=q.selected_quality_score,completion_score=completion,semantic_confidence=sr.confidence,sharpness_score=q.metrics.sharpness_score,timestamp_proximity_score=proximity,decision=FinalMemberDecision.INELIGIBLE if not eligible else FinalMemberDecision.DUPLICATE_SUPPRESSED,reason_codes=reasons))
            eligible=[m for m in members if m.eligible]
            if not eligible: raise FinalScreenshotSelectionError(f'Duplicate group {group.group_id} has no valid screenshot member.')
            # Semantic dominance guard: quality cannot override a materially stronger semantic candidate.
            highest_sem=max(eligible,key=lambda m:(m.semantic_decision_score,m.quality_score,-m.candidate_id))
            pool=[m for m in eligible if highest_sem.semantic_decision_score-m.semantic_decision_score<=self._settings.final_selection_max_semantic_deficit_for_quality_override+self._settings.final_selection_tie_epsilon]
            eps=self._settings.final_selection_tie_epsilon
            def key(m:FinalSelectionMember):
                q=qualities[m.candidate_id]
                exact=1 if abs(q.timestamp_shift_seconds)<1e-7 else 0
                return (round(m.final_selection_score/max(eps,1e-12))*eps,m.semantic_decision_score,m.quality_score,m.completion_score,m.sharpness_score,m.timestamp_proximity_score,exact,m.semantic_confidence,-q.selected_timestamp_seconds,-m.candidate_id)
            winner=max(pool,key=key)
            if winner.candidate_id!=highest_sem.candidate_id: quality_overrides+=1
            for m in members:
                if m.candidate_id==winner.candidate_id:
                    m.decision=FinalMemberDecision.WINNER; m.reason_codes=list(dict.fromkeys(m.reason_codes+['HIGHEST_FINAL_SELECTION_SCORE']))
                elif m.eligible:
                    m.decision=FinalMemberDecision.DUPLICATE_SUPPRESSED
            q=qualities[winner.candidate_id]; h=handoff[winner.candidate_id]; sr=semantic_results[winner.candidate_id]
            gs=FinalGroupSelection(group_id=group.group_id,winner_candidate_id=winner.candidate_id,winner_relative_path=q.selected_relative_path,winner_selected_timestamp_seconds=q.selected_timestamp_seconds,final_selection_score=winner.final_selection_score,group_size=group.group_size,reason_codes=['SINGLETON_GROUP'] if group.group_size==1 else ['HIGHEST_FINAL_SELECTION_SCORE'],members=members)
            group_selections.append(gs); winner_rows.append((gs,q,h,sr))
        winner_rows.sort(key=lambda x:(x[1].selected_timestamp_seconds,x[0].winner_candidate_id))
        final=[]
        for idx,(gs,q,h,sr) in enumerate(winner_rows,1):
            final.append(FinalScreenshotRecord(order=idx,candidate_id=gs.winner_candidate_id,stable_window_id=h.stable_window_id,duplicate_group_id=gs.group_id,timestamp_seconds=q.selected_timestamp_seconds,semantic_timestamp_seconds=q.semantic_timestamp_seconds,image_relative_path=q.selected_relative_path,file_sha256=q.selected_file_sha256,width=q.width,height=q.height,content_type=h.content_type,completion_state=sr.completion_state,semantic_decision_score=h.semantic_decision_score,quality_score=q.selected_quality_score,final_selection_score=gs.final_selection_score))
        sem=[r.semantic_decision_score for r in final]; qual=[r.quality_score for r in final]; fs=[r.final_selection_score for r in final]
        suppressed=max(0,len(quality_manifest.screenshots)-len(final)); ratio=suppressed/len(quality_manifest.screenshots) if quality_manifest.screenshots else 0
        warnings=[]
        if final and sum(r.quality_score<self._settings.screenshot_min_quality_score for r in final)/len(final)>self._settings.phase7_poor_final_quality_ratio: warnings.append('LOW_FINAL_SCREENSHOT_QUALITY')
        if winner_rows and quality_overrides/len(winner_rows)>self._settings.phase7_high_quality_override_ratio: warnings.append('HIGH_QUALITY_OVERRIDE_RATE')
        stats=FinalSelectionStats(duplicate_group_count=len(groups_manifest.groups),final_screenshot_count=len(final),singleton_selected_count=sum(g.group_size==1 for g in groups_manifest.groups),multi_member_group_count=sum(g.group_size>1 for g in groups_manifest.groups),suppressed_duplicate_count=suppressed,invalid_member_count=invalid_members,final_dedup_ratio=ratio,mean_winner_semantic_score=statistics.mean(sem) if sem else 0,median_winner_semantic_score=statistics.median(sem) if sem else 0,mean_winner_quality_score=statistics.mean(qual) if qual else 0,median_winner_quality_score=statistics.median(qual) if qual else 0,mean_final_selection_score=statistics.mean(fs) if fs else 0,median_final_selection_score=statistics.median(fs) if fs else 0,p10_final_selection_score=float(__import__('numpy').percentile(fs,10)) if fs else 0,warnings=warnings)
        sem_dep=self._semantic_dependency(job_id)
        artifact=canonical_fingerprint({'groups':groups_manifest.artifact_fingerprint,'semantic':sem_dep,'quality':quality_manifest.artifact_fingerprint,'config':self.config_fingerprint(),'winners':[(r.candidate_id,r.file_sha256,round(r.final_selection_score,8)) for r in final]})
        manifest=FinalSelectionsManifest(duplicate_groups_fingerprint=groups_manifest.artifact_fingerprint,semantic_dependency_fingerprint=sem_dep,quality_manifest_fingerprint=quality_manifest.artifact_fingerprint,config_fingerprint=self.config_fingerprint(),artifact_fingerprint=artifact,stats=stats,groups=group_selections,final_screenshots=final)
        self._repository.save_final_selections(job_id,manifest); self._checkpoints.mark_completed(job_id,CP_FINAL_SCREENSHOT_SELECTION_READY); return manifest

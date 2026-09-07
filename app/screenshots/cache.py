from __future__ import annotations

import re
from pathlib import Path

from app.jobs.checkpoints import CheckpointStore
from app.screenshots.duplicates import CP_DUPLICATE_GROUPS_READY, CrossWindowDuplicateDetector
from app.screenshots.extraction import CP_SOURCE_SCREENSHOTS_EXTRACTED, SourceScreenshotExtractor
from app.screenshots.fingerprints import CP_VISUAL_FINGERPRINTS_READY, VisualFingerprintGenerator
from app.screenshots.models import (
    Phase7CacheSnapshot,
    Phase7CacheState,
    Phase7CandidateCacheCheck,
    Phase7ResumePlan,
    Phase7ResumeStage,
)
from app.screenshots.quality import CP_QUALITY_SCREENSHOTS_READY, ScreenshotQualityValidator
from app.screenshots.repository import ScreenshotRepository
from app.screenshots.selection import CP_FINAL_SCREENSHOT_SELECTION_READY, FinalScreenshotSelector
from app.screenshots.utils import canonical_fingerprint, decode_rgb, file_sha256, resolve_job_path
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager
from app.video_analysis.models import AnalysisArtifactCheck, AnalysisArtifactState

CP_FINAL_SCREENSHOTS_READY='FINAL_SCREENSHOTS_READY'


class Phase7CacheCoordinator:
    def __init__(self,workspace:WorkspaceManager,checkpoints:CheckpointStore,repository:ScreenshotRepository,semantic_repository:SemanticRepository,extractor:SourceScreenshotExtractor,quality:ScreenshotQualityValidator,fingerprints:VisualFingerprintGenerator,duplicates:CrossWindowDuplicateDetector,selector:FinalScreenshotSelector)->None:
        self._workspace=workspace; self._checkpoints=checkpoints; self._repository=repository; self._semantic=semantic_repository
        self._extractor=extractor; self._quality=quality; self._fingerprints=fingerprints; self._duplicates=duplicates; self._selector=selector

    @staticmethod
    def _check(state:AnalysisArtifactState,reason:str|None=None)->AnalysisArtifactCheck:
        return AnalysisArtifactCheck(state=state,reason=reason)

    @staticmethod
    def _candidate(candidate_id:int,state:Phase7CacheState,reason:str|None=None)->Phase7CandidateCacheCheck:
        return Phase7CandidateCacheCheck(candidate_id=candidate_id,state=state,reason=reason)

    def handoff(self,job_id:str):
        return self._semantic.load_phase7_handoff(job_id)

    def cleanup_partial_artifacts(self,job_id:str)->list[str]:
        root=self._workspace.screenshots_dir(job_id); removed=[]
        if not root.exists(): return removed
        for p in root.rglob('*'):
            if p.is_file() and ('.tmp.' in p.name or p.name.endswith('.tmp') or p.name.endswith('.tmp.json')):
                try: p.unlink(); removed.append(self._workspace.relative_to_workspace(job_id,p))
                except OSError: pass
        return removed

    def extraction_candidate_checks(self,job_id:str,handoff=None)->list[Phase7CandidateCacheCheck]:
        handoff=handoff or self.handoff(job_id)
        # Resolve source fingerprint once. If source is unavailable all candidates are stale/failed.
        try: _,source_fp,_=self._extractor._source(job_id)
        except Exception as exc: return [self._candidate(h.candidate_id,Phase7CacheState.STALE,'source_dependency_invalid') for h in handoff]
        checks=[]
        for h in handoff:
            path=self._workspace.screenshot_extraction_record_path(job_id,h.candidate_id)
            if not path.exists(): checks.append(self._candidate(h.candidate_id,Phase7CacheState.MISSING,'record_missing')); continue
            try:
                r=self._repository.load_extraction_record(job_id,h.candidate_id)
                if r.extraction_fingerprint!=self._extractor.expected_fingerprint(h,source_fp) or r.extraction_config_fingerprint!=self._extractor.config_fingerprint():
                    checks.append(self._candidate(h.candidate_id,Phase7CacheState.STALE,'dependency_mismatch')); continue
                img=resolve_job_path(self._workspace,job_id,r.screenshot_relative_path)
                if file_sha256(img)!=r.file_sha256: checks.append(self._candidate(h.candidate_id,Phase7CacheState.CORRUPT,'file_sha_mismatch')); continue
                decode_rgb(img)
                checks.append(self._candidate(h.candidate_id,Phase7CacheState.VALID))
            except Exception: checks.append(self._candidate(h.candidate_id,Phase7CacheState.CORRUPT,'invalid_record_or_image'))
        return checks

    def validate_extraction(self,job_id:str,handoff=None):
        handoff=handoff or self.handoff(job_id); checks=self.extraction_candidate_checks(job_id,handoff)
        if any(x.state is not Phase7CacheState.VALID for x in checks):
            return self._check(AnalysisArtifactState.PARTIAL if any(x.state is Phase7CacheState.VALID for x in checks) else AnalysisArtifactState.STALE,'candidate_extractions_incomplete'),None,checks
        try:
            manifest=self._repository.load_extraction_manifest(job_id)
            ids=[r.candidate_id for r in manifest.screenshots]
            if ids!=[r.candidate_id for r in sorted(manifest.screenshots,key=lambda r:(r.requested_timestamp_seconds,r.candidate_id))] or set(ids)!={h.candidate_id for h in handoff}: raise ValueError
            if manifest.config_fingerprint!=self._extractor.config_fingerprint(): return self._check(AnalysisArtifactState.STALE,'config_mismatch'),manifest,checks
            dependency = canonical_fingerprint([(x.candidate_id, round(x.candidate_timestamp_seconds,6)) for x in handoff])
            expected_artifact = canonical_fingerprint({'algorithm': manifest.algorithm_version, 'semantic':dependency, 'source':manifest.source_fingerprint, 'config':manifest.config_fingerprint, 'records':[r.extraction_fingerprint for r in manifest.screenshots]})
            if manifest.semantic_selections_fingerprint != dependency or manifest.artifact_fingerprint != expected_artifact:
                return self._check(AnalysisArtifactState.CORRUPT,'artifact_fingerprint_mismatch'),manifest,checks
            if not self._checkpoints.is_completed(job_id,CP_SOURCE_SCREENSHOTS_EXTRACTED): return self._check(AnalysisArtifactState.PARTIAL,'checkpoint_missing'),manifest,checks
            return self._check(AnalysisArtifactState.VALID),manifest,checks
        except Exception:
            return self._check(AnalysisArtifactState.PARTIAL,'manifest_missing_or_invalid'),None,checks

    def quality_candidate_checks(self,job_id:str,handoff=None)->list[Phase7CandidateCacheCheck]:
        handoff=handoff or self.handoff(job_id); extraction_checks={x.candidate_id:x for x in self.extraction_candidate_checks(job_id,handoff)}; out=[]
        for h in handoff:
            if extraction_checks[h.candidate_id].state is not Phase7CacheState.VALID: out.append(self._candidate(h.candidate_id,Phase7CacheState.STALE,'extraction_invalid')); continue
            try:
                ex=self._repository.load_extraction_record(job_id,h.candidate_id); r=self._repository.load_quality_record(job_id,h.candidate_id)
                if r.extraction_fingerprint!=ex.extraction_fingerprint or r.quality_config_fingerprint!=self._quality.config_fingerprint(): out.append(self._candidate(h.candidate_id,Phase7CacheState.STALE,'dependency_mismatch')); continue
                img=resolve_job_path(self._workspace,job_id,r.selected_relative_path)
                if file_sha256(img)!=r.selected_file_sha256: out.append(self._candidate(h.candidate_id,Phase7CacheState.CORRUPT,'file_sha_mismatch')); continue
                decode_rgb(img); out.append(self._candidate(h.candidate_id,Phase7CacheState.VALID))
            except Exception as exc:
                state=Phase7CacheState.MISSING if not self._workspace.screenshot_quality_record_path(job_id,h.candidate_id).exists() else Phase7CacheState.CORRUPT
                out.append(self._candidate(h.candidate_id,state,'quality_record_missing_or_invalid'))
        return out

    def validate_quality(self,job_id:str,handoff=None):
        handoff=handoff or self.handoff(job_id); checks=self.quality_candidate_checks(job_id,handoff)
        if any(x.state is not Phase7CacheState.VALID for x in checks): return self._check(AnalysisArtifactState.PARTIAL if any(x.state is Phase7CacheState.VALID for x in checks) else AnalysisArtifactState.STALE,'quality_candidates_incomplete'),None,checks
        try:
            m=self._repository.load_quality_manifest(job_id); ex=self._repository.load_extraction_manifest(job_id)
            if m.extraction_manifest_fingerprint!=ex.artifact_fingerprint or m.config_fingerprint!=self._quality.config_fingerprint(): return self._check(AnalysisArtifactState.STALE,'dependency_mismatch'),m,checks
            if set(x.candidate_id for x in m.screenshots)!={h.candidate_id for h in handoff}: raise ValueError
            expected_artifact = canonical_fingerprint({'extraction':m.extraction_manifest_fingerprint,'config':m.config_fingerprint,'records':[r.quality_fingerprint for r in m.screenshots]})
            if expected_artifact != m.artifact_fingerprint: return self._check(AnalysisArtifactState.CORRUPT,'artifact_fingerprint_mismatch'),m,checks
            if not self._checkpoints.is_completed(job_id,CP_QUALITY_SCREENSHOTS_READY): return self._check(AnalysisArtifactState.PARTIAL,'checkpoint_missing'),m,checks
            return self._check(AnalysisArtifactState.VALID),m,checks
        except Exception: return self._check(AnalysisArtifactState.PARTIAL,'manifest_missing_or_invalid'),None,checks

    def fingerprint_candidate_checks(self,job_id:str,handoff=None)->list[Phase7CandidateCacheCheck]:
        handoff=handoff or self.handoff(job_id); qchecks={x.candidate_id:x for x in self.quality_candidate_checks(job_id,handoff)}; out=[]
        for h in handoff:
            if qchecks[h.candidate_id].state is not Phase7CacheState.VALID: out.append(self._candidate(h.candidate_id,Phase7CacheState.STALE,'quality_invalid')); continue
            try:
                q=self._repository.load_quality_record(job_id,h.candidate_id); r=self._repository.load_fingerprint_record(job_id,h.candidate_id)
                if r.visual_fingerprint!=self._fingerprints.expected_fingerprint(q) or r.fingerprint_config_fingerprint!=self._fingerprints.config_fingerprint() or r.source_file_sha256!=q.selected_file_sha256: out.append(self._candidate(h.candidate_id,Phase7CacheState.STALE,'dependency_mismatch')); continue
                for rel,sha in ((r.thumbnail_relative_path,r.thumbnail_file_sha256),(r.edge_map_relative_path,r.edge_map_file_sha256)):
                    p=resolve_job_path(self._workspace,job_id,rel); decode_rgb(p)
                    if file_sha256(p)!=sha: raise ValueError
                out.append(self._candidate(h.candidate_id,Phase7CacheState.VALID))
            except Exception:
                state=Phase7CacheState.MISSING if not self._workspace.screenshot_fingerprint_record_path(job_id,h.candidate_id).exists() else Phase7CacheState.CORRUPT
                out.append(self._candidate(h.candidate_id,state,'fingerprint_record_missing_or_invalid'))
        return out

    def validate_fingerprints(self,job_id:str,handoff=None):
        handoff=handoff or self.handoff(job_id); checks=self.fingerprint_candidate_checks(job_id,handoff)
        if any(x.state is not Phase7CacheState.VALID for x in checks): return self._check(AnalysisArtifactState.PARTIAL if any(x.state is Phase7CacheState.VALID for x in checks) else AnalysisArtifactState.STALE,'fingerprints_incomplete'),None,checks
        try:
            m=self._repository.load_fingerprint_manifest(job_id); q=self._repository.load_quality_manifest(job_id)
            if m.quality_manifest_fingerprint!=q.artifact_fingerprint or m.config_fingerprint!=self._fingerprints.config_fingerprint(): return self._check(AnalysisArtifactState.STALE,'dependency_mismatch'),m,checks
            if set(x.candidate_id for x in m.fingerprints)!={h.candidate_id for h in handoff}: raise ValueError
            expected_artifact = canonical_fingerprint({'quality':m.quality_manifest_fingerprint,'config':m.config_fingerprint,'records':[r.visual_fingerprint for r in m.fingerprints]})
            if expected_artifact != m.artifact_fingerprint: return self._check(AnalysisArtifactState.CORRUPT,'artifact_fingerprint_mismatch'),m,checks
            if not self._checkpoints.is_completed(job_id,CP_VISUAL_FINGERPRINTS_READY): return self._check(AnalysisArtifactState.PARTIAL,'checkpoint_missing'),m,checks
            return self._check(AnalysisArtifactState.VALID),m,checks
        except Exception: return self._check(AnalysisArtifactState.PARTIAL,'manifest_missing_or_invalid'),None,checks

    def validate_duplicates(self,job_id:str,fingerprint_manifest=None):
        if fingerprint_manifest is None:
            check,fingerprint_manifest,_=self.validate_fingerprints(job_id)
            if not check.valid or fingerprint_manifest is None: return self._check(AnalysisArtifactState.STALE,'fingerprints_invalid'),None
        try:
            p=self._repository.load_duplicate_pairs(job_id); g=self._repository.load_duplicate_groups(job_id)
            current_content_fp = self._duplicates.content_types_fingerprint(job_id)
            if p.fingerprint_manifest_fingerprint!=fingerprint_manifest.artifact_fingerprint or g.fingerprint_manifest_fingerprint!=fingerprint_manifest.artifact_fingerprint or p.config_fingerprint!=self._duplicates.config_fingerprint() or g.config_fingerprint!=self._duplicates.config_fingerprint() or p.content_types_fingerprint != current_content_fp: return self._check(AnalysisArtifactState.STALE,'dependency_mismatch'),g
            expected_pair = canonical_fingerprint({'fingerprints':p.fingerprint_manifest_fingerprint,'content':p.content_types_fingerprint,'config':p.config_fingerprint,'pairs':[x.model_dump(mode='json') for x in p.pairs]})
            expected_group = canonical_fingerprint({'pairs':p.artifact_fingerprint,'fingerprints':g.fingerprint_manifest_fingerprint,'config':g.config_fingerprint,'groups':[x.model_dump(mode='json') for x in g.groups]})
            if expected_pair != p.artifact_fingerprint or expected_group != g.artifact_fingerprint: return self._check(AnalysisArtifactState.CORRUPT,'artifact_fingerprint_mismatch'),g
            current={x.candidate_id for x in fingerprint_manifest.fingerprints}; members=[c for group in g.groups for c in group.candidate_ids]
            if set(members)!=current or len(members)!=len(set(members)): return self._check(AnalysisArtifactState.CORRUPT,'group_coverage_invalid'),g
            if not self._checkpoints.is_completed(job_id,CP_DUPLICATE_GROUPS_READY): return self._check(AnalysisArtifactState.PARTIAL,'checkpoint_missing'),g
            return self._check(AnalysisArtifactState.VALID),g
        except Exception: return self._check(AnalysisArtifactState.MISSING,'duplicate_artifact_missing_or_invalid'),None

    def validate_final_selection(self,job_id:str,groups=None,quality=None):
        if groups is None:
            check,groups=self.validate_duplicates(job_id)
            if not check.valid or groups is None: return self._check(AnalysisArtifactState.STALE,'duplicate_groups_invalid'),None
        try:
            quality=quality or self._repository.load_quality_manifest(job_id); m=self._repository.load_final_selections(job_id)
            if m.duplicate_groups_fingerprint!=groups.artifact_fingerprint or m.quality_manifest_fingerprint!=quality.artifact_fingerprint or m.config_fingerprint!=self._selector.config_fingerprint(): return self._check(AnalysisArtifactState.STALE,'dependency_mismatch'),m
            if m.semantic_dependency_fingerprint!=self._selector._semantic_dependency(job_id): return self._check(AnalysisArtifactState.STALE,'semantic_dependency_mismatch'),m
            group_map={g.group_id:set(g.candidate_ids) for g in groups.groups}; seen=[]
            if len(m.groups)!=len(groups.groups) or len(m.final_screenshots)!=len(groups.groups): return self._check(AnalysisArtifactState.CORRUPT,'winner_count_mismatch'),m
            for gs in m.groups:
                if gs.group_id not in group_map or gs.winner_candidate_id not in group_map[gs.group_id]: return self._check(AnalysisArtifactState.CORRUPT,'winner_reference_broken'),m
                seen.append(gs.winner_candidate_id)
            if len(seen)!=len(set(seen)): return self._check(AnalysisArtifactState.CORRUPT,'duplicate_winner'),m
            for idx,r in enumerate(m.final_screenshots,1):
                if r.order!=idx: return self._check(AnalysisArtifactState.CORRUPT,'order_invalid'),m
                p=resolve_job_path(self._workspace,job_id,r.image_relative_path); decode_rgb(p)
                if file_sha256(p)!=r.file_sha256: return self._check(AnalysisArtifactState.CORRUPT,'winner_file_sha_mismatch'),m
            expected_artifact = canonical_fingerprint({'groups':m.duplicate_groups_fingerprint,'semantic':m.semantic_dependency_fingerprint,'quality':m.quality_manifest_fingerprint,'config':m.config_fingerprint,'winners':[(r.candidate_id,r.file_sha256,round(r.final_selection_score,8)) for r in m.final_screenshots]})
            if expected_artifact != m.artifact_fingerprint: return self._check(AnalysisArtifactState.CORRUPT,'artifact_fingerprint_mismatch'),m
            if not self._checkpoints.is_completed(job_id,CP_FINAL_SCREENSHOT_SELECTION_READY): return self._check(AnalysisArtifactState.PARTIAL,'checkpoint_missing'),m
            return self._check(AnalysisArtifactState.VALID),m
        except Exception: return self._check(AnalysisArtifactState.MISSING,'final_selection_missing_or_invalid'),None

    def reconcile(self,job_id:str)->Phase7CacheSnapshot:
        handoff=self.handoff(job_id)
        echeck,em,ec=self.validate_extraction(job_id,handoff)
        qcheck,qm,qc=self.validate_quality(job_id,handoff)
        fcheck,fm,fc=self.validate_fingerprints(job_id,handoff)
        dcheck,dm=self.validate_duplicates(job_id,fm if fcheck.valid else None)
        scheck,sm=self.validate_final_selection(job_id,dm if dcheck.valid else None,qm if qcheck.valid else None)
        eids=[x.candidate_id for x in ec if x.state is not Phase7CacheState.VALID]
        qids=[x.candidate_id for x in qc if x.state is not Phase7CacheState.VALID]
        fids=[x.candidate_id for x in fc if x.state is not Phase7CacheState.VALID]
        if eids: stage=Phase7ResumeStage.SOURCE_EXTRACTION
        elif qids: stage=Phase7ResumeStage.QUALITY_VALIDATION
        elif fids: stage=Phase7ResumeStage.FINGERPRINT_GENERATION
        elif not dcheck.valid: stage=Phase7ResumeStage.DUPLICATE_DETECTION
        elif not scheck.valid: stage=Phase7ResumeStage.FINAL_SCREENSHOT_SELECTION
        else: stage=Phase7ResumeStage.PHASE7_READY
        current_ids={h.candidate_id for h in handoff}; stored=set()
        for d in (self._workspace.screenshot_extraction_records_dir(job_id),self._workspace.screenshot_quality_records_dir(job_id),self._workspace.screenshot_fingerprint_records_dir(job_id)):
            if d.exists():
                for p in d.glob('candidate_*.json'):
                    m=re.match(r'candidate_(\d+)\.json$',p.name)
                    if m: stored.add(int(m.group(1)))
        plan=Phase7ResumePlan(resume_stage=stage,candidate_ids_to_extract=eids,candidate_ids_to_quality_check=sorted(set(qids)|set(eids)),candidate_ids_to_fingerprint=sorted(set(fids)|set(qids)|set(eids)),rebuild_extraction_manifest=not echeck.valid,rebuild_quality_manifest=not qcheck.valid,rebuild_fingerprint_manifest=not fcheck.valid,rerun_duplicate_detection=not dcheck.valid or bool(fids or qids or eids),rerun_final_selection=not scheck.valid or not dcheck.valid or bool(fids or qids or eids),cleanup_orphan_candidate_ids=sorted(stored-current_ids))
        ready=self._check(AnalysisArtifactState.VALID) if stage is Phase7ResumeStage.PHASE7_READY and self._checkpoints.is_completed(job_id,CP_FINAL_SCREENSHOTS_READY) else self._check(AnalysisArtifactState.PARTIAL if stage is Phase7ResumeStage.PHASE7_READY else AnalysisArtifactState.STALE,'terminal_checkpoint_missing' if stage is Phase7ResumeStage.PHASE7_READY else 'phase7_incomplete')
        return Phase7CacheSnapshot(extraction=echeck,quality=qcheck,fingerprints=fcheck,duplicates=dcheck,final_selection=scheck,final_screenshots_ready=ready,extraction_candidates=ec,quality_candidates=qc,fingerprint_candidates=fc,resume_plan=plan)

    inspect=reconcile

    def repair_checkpoints(self,job_id:str)->None:
        snap=self.reconcile(job_id)
        mapping=[(snap.extraction,CP_SOURCE_SCREENSHOTS_EXTRACTED),(snap.quality,CP_QUALITY_SCREENSHOTS_READY),(snap.fingerprints,CP_VISUAL_FINGERPRINTS_READY),(snap.duplicates,CP_DUPLICATE_GROUPS_READY),(snap.final_selection,CP_FINAL_SCREENSHOT_SELECTION_READY)]
        for check,cp in mapping:
            if check.valid and not self._checkpoints.is_completed(job_id,cp): self._checkpoints.mark_completed(job_id,cp)
            elif not check.valid and self._checkpoints.is_completed(job_id,cp): self._checkpoints.invalidate(job_id,cp)
        if not snap.final_selection.valid and self._checkpoints.is_completed(job_id,CP_FINAL_SCREENSHOTS_READY): self._checkpoints.invalidate(job_id,CP_FINAL_SCREENSHOTS_READY)

from __future__ import annotations

import asyncio
import statistics
import time
from collections import Counter, defaultdict

from app.core.exceptions import JobCancelledError, Phase7ValidationError
from app.core.logging import JobEventLogger
from app.jobs.checkpoints import CheckpointStore
from app.jobs.models import JobStage, JobStatus
from app.jobs.service import JobService
from app.screenshots.cache import CP_FINAL_SCREENSHOTS_READY, Phase7CacheCoordinator
from app.screenshots.duplicates import CrossWindowDuplicateDetector
from app.screenshots.evaluation import Phase7Evaluator
from app.screenshots.extraction import SourceScreenshotExtractor
from app.screenshots.fingerprints import VisualFingerprintGenerator
from app.screenshots.models import Phase7Summary, Phase7Timings
from app.screenshots.quality import ScreenshotQualityValidator
from app.screenshots.repository import ScreenshotRepository
from app.screenshots.selection import FinalScreenshotSelector
from app.semantic_analysis.repository import SemanticRepository


class ScreenshotPipeline:
    def __init__(self,*,jobs:JobService,checkpoints:CheckpointStore,events:JobEventLogger,semantic_repository:SemanticRepository,repository:ScreenshotRepository,extractor:SourceScreenshotExtractor,quality:ScreenshotQualityValidator,fingerprints:VisualFingerprintGenerator,duplicates:CrossWindowDuplicateDetector,selector:FinalScreenshotSelector,cache:Phase7CacheCoordinator,evaluator:Phase7Evaluator)->None:
        self._jobs=jobs; self._checkpoints=checkpoints; self._events=events; self._semantic=semantic_repository; self._repository=repository
        self._extractor=extractor; self._quality=quality; self._fingerprints=fingerprints; self._duplicates=duplicates; self._selector=selector; self._cache=cache; self._evaluator=evaluator
        self._locks:dict[str,asyncio.Lock]=defaultdict(asyncio.Lock)

    def _cancelled(self,job_id:str)->bool: return self._jobs.get_job(job_id).status is JobStatus.CANCELLED
    def _guard(self,job_id:str)->None:
        if self._cancelled(job_id): raise JobCancelledError(f'Job {job_id} was cancelled')
    def _stage(self,job_id:str,stage:JobStage,msg:str)->None:
        self._guard(job_id); self._jobs.update_stage(job_id,stage,msg); self._events.write(job_id,level='INFO',stage=stage.value,message=msg)
    def _progress(self,job_id:str,value:int,msg:str)->None:
        if self._cancelled(job_id): return
        current=self._jobs.get_job(job_id); self._jobs.update_progress(job_id,max(current.progress,value),msg)

    async def process(self,job_id:str,*,finalize_job:bool=True)->Phase7Summary:
        async with self._locks[job_id]: return await self._process_locked(job_id,finalize_job=finalize_job)

    async def _process_locked(self,job_id:str,*,finalize_job:bool)->Phase7Summary:
        if not self._checkpoints.is_completed(job_id,'SEMANTIC_CANDIDATES_READY'):
            raise Phase7ValidationError('SEMANTIC_CANDIDATES_READY is required before Phase 7.')
        started=time.monotonic(); removed=self._cache.cleanup_partial_artifacts(job_id); handoff=self._semantic.load_phase7_handoff(job_id)
        self._events.write(job_id,level='INFO',stage='SCREENSHOT_PIPELINE',message=f'Phase-7 cache inspection started; removed {len(removed)} temporary artifacts')
        timings={'extract':0.0,'quality':0.0,'fp':0.0,'dup':0.0,'select':0.0}

        # 7.1 exact full-resolution extraction
        self._stage(job_id,JobStage.EXTRACTING_SOURCE_SCREENSHOTS,'Extracting high-resolution source screenshots')
        echeck,em,ec=self._cache.validate_extraction(job_id,handoff); valid_e={x.candidate_id for x in ec if x.state.value=='VALID'}
        records=[]; t=time.monotonic()
        for i,h in enumerate(handoff,1):
            self._guard(job_id)
            if h.candidate_id in valid_e: r=self._repository.load_extraction_record(job_id,h.candidate_id)
            else: r=await self._extractor.extract_candidate(job_id,h,cancel_check=lambda:self._cancelled(job_id))
            records.append(r); self._progress(job_id,82+int(3*i/max(1,len(handoff))),'Extracting high-resolution source screenshots')
        timings['extract']=time.monotonic()-t
        em=self._extractor.build_manifest(job_id,handoff,records,cached_count=len(valid_e))

        # 7.2 quality validation / bounded local refinement
        self._stage(job_id,JobStage.VALIDATING_SCREENSHOT_QUALITY,'Validating screenshot quality')
        qchecks=self._cache.quality_candidate_checks(job_id,handoff); valid_q={x.candidate_id for x in qchecks if x.state.value=='VALID'}
        qrecords=[]; t=time.monotonic(); by_ex={r.candidate_id:r for r in records}
        for i,h in enumerate(handoff,1):
            self._guard(job_id)
            if h.candidate_id in valid_q: q=self._repository.load_quality_record(job_id,h.candidate_id)
            else: q=await self._quality.validate_candidate(job_id,by_ex[h.candidate_id],cancel_check=lambda:self._cancelled(job_id))
            qrecords.append(q); self._progress(job_id,85+int(3*i/max(1,len(handoff))),'Validating screenshot quality')
        timings['quality']=time.monotonic()-t
        qm=self._quality.build_manifest(job_id,em,qrecords,cached_count=len(valid_q))

        # 7.3 final screenshot fingerprints
        self._stage(job_id,JobStage.GENERATING_VISUAL_FINGERPRINTS,'Generating screenshot comparison fingerprints')
        fchecks=self._cache.fingerprint_candidate_checks(job_id,handoff); valid_f={x.candidate_id for x in fchecks if x.state.value=='VALID'}
        frecords=[]; t=time.monotonic(); by_q={r.candidate_id:r for r in qrecords}
        for i,h in enumerate(handoff,1):
            self._guard(job_id)
            if h.candidate_id in valid_f: f=self._repository.load_fingerprint_record(job_id,h.candidate_id)
            else: f=await asyncio.to_thread(self._fingerprints.generate_candidate,job_id,by_q[h.candidate_id],cancel_check=lambda:self._cancelled(job_id))
            frecords.append(f); self._progress(job_id,88+int(2*i/max(1,len(handoff))),'Generating screenshot comparison fingerprints')
        timings['fp']=time.monotonic()-t
        fm=self._fingerprints.build_manifest(job_id,qm,frecords,cached_count=len(valid_f))

        # 7.4 cross-window duplicate groups. Reuse only if exact dependencies validate now.
        self._stage(job_id,JobStage.DETECTING_SCREENSHOT_DUPLICATES,'Detecting duplicate teaching screenshots')
        dcheck,gm=self._cache.validate_duplicates(job_id,fm); t=time.monotonic()
        if not dcheck.valid or gm is None:
            _,gm=await asyncio.to_thread(self._duplicates.process,job_id,fm,cancel_check=lambda:self._cancelled(job_id))
        timings['dup']=time.monotonic()-t; self._progress(job_id,92,'Duplicate screenshot groups ready')

        # 7.5 deterministic best member per group.
        self._stage(job_id,JobStage.SELECTING_FINAL_SCREENSHOTS,'Selecting the best screenshot from duplicate groups')
        scheck,sm=self._cache.validate_final_selection(job_id,gm,qm); t=time.monotonic()
        if not scheck.valid or sm is None:
            sm=await asyncio.to_thread(self._selector.process,job_id,gm,qm,cancel_check=lambda:self._cancelled(job_id))
        timings['select']=time.monotonic()-t; self._progress(job_id,94,'Final screenshot selections ready')

        # Final integrity validation before terminal checkpoint.
        dcheck,gm2=self._cache.validate_duplicates(job_id,fm); scheck,sm2=self._cache.validate_final_selection(job_id,gm2 if dcheck.valid else None,qm)
        if not dcheck.valid or not scheck.valid or sm2 is None or gm2 is None:
            raise Phase7ValidationError('Phase-7 artifacts failed final consistency validation.')
        if len(sm2.final_screenshots)!=len(gm2.groups): raise Phase7ValidationError('Final screenshot count does not match duplicate group count.')
        if [r.order for r in sm2.final_screenshots]!=list(range(1,len(sm2.final_screenshots)+1)): raise Phase7ValidationError('Final screenshot order is not contiguous.')
        if [r.timestamp_seconds for r in sm2.final_screenshots]!=sorted(r.timestamp_seconds for r in sm2.final_screenshots): raise Phase7ValidationError('Final screenshots are not chronologically ordered.')

        finals=sm2.final_screenshots; warnings=list(dict.fromkeys(qm.stats.warnings+gm2.stats.warnings+sm2.stats.warnings)); dist=Counter(r.content_type.value for r in finals)
        timing_model=Phase7Timings(source_extraction_seconds=round(timings['extract'],3),quality_validation_seconds=round(timings['quality'],3),fingerprint_seconds=round(timings['fp'],3),duplicate_detection_seconds=round(timings['dup'],3),final_selection_seconds=round(timings['select'],3),total_phase7_seconds=round(time.monotonic()-started,3))
        summary=Phase7Summary(semantic_candidate_count=len(handoff),source_extraction_count=len(em.screenshots),quality_selected_count=len(qm.screenshots),fingerprint_count=len(fm.fingerprints),duplicate_group_count=len(gm2.groups),final_screenshot_count=len(finals),suppressed_duplicate_count=sm2.stats.suppressed_duplicate_count,exact_timestamp_winner_count=sum(abs(r.timestamp_seconds-r.semantic_timestamp_seconds)<1e-7 for r in finals),refined_timestamp_winner_count=sum(abs(r.timestamp_seconds-r.semantic_timestamp_seconds)>=1e-7 for r in finals),mean_final_quality_score=statistics.mean([r.quality_score for r in finals]) if finals else 0,mean_final_semantic_score=statistics.mean([r.semantic_decision_score for r in finals]) if finals else 0,mean_final_selection_score=statistics.mean([r.final_selection_score for r in finals]) if finals else 0,final_dedup_ratio=sm2.stats.final_dedup_ratio,content_type_distribution=dict(dist),cache_stats={'extraction_cache_hits':len(valid_e),'quality_cache_hits':len(valid_q),'fingerprint_cache_hits':len(valid_f),'duplicate_cache_hit':dcheck.valid,'final_selection_cache_hit':scheck.valid},extraction_manifest_fingerprint=em.artifact_fingerprint,quality_manifest_fingerprint=qm.artifact_fingerprint,fingerprint_manifest_fingerprint=fm.artifact_fingerprint,duplicate_groups_fingerprint=gm2.artifact_fingerprint,final_selections_fingerprint=sm2.artifact_fingerprint,timings=timing_model,warnings=warnings)
        self._repository.save_summary(job_id,summary)
        try: self._evaluator.evaluate(job_id,persist=True)
        except Exception as exc: self._events.write(job_id,level='WARNING',stage='SCREENSHOT_EVALUATION',message=f'Phase-7 evaluation unavailable: {type(exc).__name__}')
        self._checkpoints.mark_completed(job_id,CP_FINAL_SCREENSHOTS_READY); self._progress(job_id,96,'Final screenshots ready for document generation')
        self._events.write(job_id,level='INFO',stage=CP_FINAL_SCREENSHOTS_READY,message='Phase 7 final screenshots ready')
        if finalize_job: self._jobs.mark_completed(job_id,message='Final screenshots ready for document generation')
        return summary

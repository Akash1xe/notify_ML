from __future__ import annotations

import asyncio

from app.screenshots.models import DuplicateGroup, DuplicateGroupsManifest, DuplicateGroupStats
from app.screenshots.selection import FinalScreenshotSelector
from app.screenshots.utils import canonical_fingerprint
from tests.phase7_helpers import build_phase7_context, build_phase7_pipeline


class ExpandedSemanticRepo:
    def __init__(self,base,new_id:int): self.base=base; self.new_id=new_id
    def load_phase7_handoff(self,job_id):
        h=self.base.load_phase7_handoff(job_id)[0]
        return [h,h.model_copy(update={'candidate_id':self.new_id,'stable_window_id':2,'semantic_decision_score':max(0,h.semantic_decision_score-0.01)})]
    def load_results(self,job_id):
        m=self.base.load_results(job_id); r=m.results[0]
        return m.model_copy(update={'results':[r,r.model_copy(update={'candidate_id':self.new_id})]})
    def load_selections(self,job_id): return self.base.load_selections(job_id)


def test_duplicate_group_selector_can_prefer_much_cleaner_near_semantic_equal_member(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    pipeline,cache,repo,extractor,quality,*_=build_phase7_pipeline(ctx,sem,backend); job=ctx[4].id
    h=sem.load_phase7_handoff(job)[0]
    ex=asyncio.run(extractor.extract_candidate(job,h)); em=extractor.build_manifest(job,[h],[ex])
    q=asyncio.run(quality.validate_candidate(job,ex))
    new_id=102
    better_metrics=q.metrics.model_copy(update={'quality_score':0.99,'sharpness_score':0.99})
    q2=q.model_copy(update={'candidate_id':new_id,'stable_window_id':2,'selected_quality_score':0.99,'metrics':better_metrics,'quality_fingerprint':'synthetic-better'})
    qm=quality.build_manifest(job,em,[q,q2])
    group=DuplicateGroup(group_id='dup_test',candidate_ids=[h.candidate_id,new_id],representative_candidate_id=h.candidate_id,group_size=2,earliest_timestamp_seconds=q.selected_timestamp_seconds,latest_timestamp_seconds=q2.selected_timestamp_seconds,minimum_internal_duplicate_score=.95)
    gm=DuplicateGroupsManifest(duplicate_pairs_fingerprint='pairs',fingerprint_manifest_fingerprint='fps',config_fingerprint='cfg',artifact_fingerprint='groups',stats=DuplicateGroupStats(candidate_count=2,duplicate_group_count=1,multi_member_group_count=1,potential_duplicate_removals=1,duplicate_ratio=.5,mean_group_size=2,max_group_size=2),groups=[group])
    expanded=ExpandedSemanticRepo(sem,new_id)
    selector=FinalScreenshotSelector(ctx[0],ctx[1],ctx[3],repo,expanded)
    result=selector.process(job,gm,qm)
    assert result.groups[0].winner_candidate_id==new_id
    assert result.final_screenshots[0].candidate_id==new_id
    assert result.final_screenshots[0].order==1

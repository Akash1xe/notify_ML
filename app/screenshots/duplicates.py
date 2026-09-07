from __future__ import annotations

import itertools
import math
import statistics
from collections import defaultdict
from typing import Callable

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from app.core.config import AppSettings
from app.core.exceptions import DuplicateDetectionError, JobCancelledError
from app.jobs.checkpoints import CheckpointStore
from app.screenshots.fingerprints import normalized_hamming_distance
from app.screenshots.models import (
    CROSS_WINDOW_DUPLICATE_ALGORITHM_VERSION,
    DuplicateGroup,
    DuplicateGroupsManifest,
    DuplicateGroupStats,
    DuplicatePairsManifest,
    DuplicatePairStats,
    FingerprintManifest,
    PairDecision,
    ScreenshotPairSimilarity,
    VisualFingerprintRecord,
)
from app.screenshots.repository import ScreenshotRepository
from app.screenshots.utils import canonical_fingerprint, decode_rgb, resolve_job_path
from app.semantic_analysis.models import SemanticContentType
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager

CP_DUPLICATE_GROUPS_READY='DUPLICATE_GROUPS_READY'


class CrossWindowDuplicateDetector:
    def __init__(self,settings:AppSettings,workspace:WorkspaceManager,checkpoints:CheckpointStore,repository:ScreenshotRepository,semantic_repository:SemanticRepository)->None:
        self._settings=settings; self._workspace=workspace; self._checkpoints=checkpoints; self._repository=repository; self._semantic=semantic_repository

    def config_fingerprint(self)->str:
        s=self._settings
        return canonical_fingerprint({'algorithm':CROSS_WINDOW_DUPLICATE_ALGORITHM_VERSION,'window':s.duplicate_primary_temporal_window_seconds,'cutoff':s.duplicate_full_pairwise_cutoff,'bucket_slices':s.duplicate_hash_bucket_slices,'weights':[s.duplicate_structural_weight,s.duplicate_edge_weight,s.duplicate_phash_weight,s.duplicate_dhash_weight,s.duplicate_content_weight,s.duplicate_temporal_weight],'threshold':s.duplicate_score_threshold,'possible':s.possible_duplicate_score_threshold,'min_structural':s.duplicate_min_structural_similarity,'min_edge':s.duplicate_min_edge_similarity,'max_addition':s.duplicate_max_structural_addition_ratio,'changed_pixel':s.duplicate_changed_pixel_threshold,'long_hash':s.duplicate_long_range_phash_distance})

    def _content_map(self,job_id:str)->dict[int,SemanticContentType]:
        return {x.candidate_id:x.content_type for x in self._semantic.load_phase7_handoff(job_id)}

    def content_types_fingerprint(self, job_id: str) -> str:
        content = self._content_map(job_id)
        return canonical_fingerprint(sorted((cid, typ.value) for cid, typ in content.items()))

    @staticmethod
    def _compatibility(a:SemanticContentType,b:SemanticContentType)->float:
        if a==b: return 1.0
        if SemanticContentType.UNKNOWN in {a,b}: return 0.65
        if SemanticContentType.MIXED in {a,b}: return 0.8
        related=[{SemanticContentType.WHITEBOARD,SemanticContentType.BLACKBOARD,SemanticContentType.EQUATION,SemanticContentType.DIAGRAM},{SemanticContentType.SLIDE,SemanticContentType.DOCUMENT,SemanticContentType.DIAGRAM},{SemanticContentType.CODE,SemanticContentType.UI_DEMO}]
        return 0.75 if any(a in g and b in g for g in related) else 0.4

    def _pair_keys(self,records:list[VisualFingerprintRecord])->list[tuple[int,int]]:
        records=sorted(records,key=lambda r:(r.selected_timestamp_seconds,r.candidate_id)); n=len(records)
        if n<=self._settings.duplicate_full_pairwise_cutoff:
            return [(a.candidate_id,b.candidate_id) for a,b in itertools.combinations(records,2)]
        keys:set[tuple[int,int]]=set()
        # temporal sliding candidates
        left=0
        for right,b in enumerate(records):
            while left<right and b.selected_timestamp_seconds-records[left].selected_timestamp_seconds>self._settings.duplicate_primary_temporal_window_seconds: left+=1
            for i in range(left,right): keys.add(tuple(sorted((records[i].candidate_id,b.candidate_id))))
        # long-range hash buckets: each pHash slice is an independent lookup key.
        slices=max(1,self._settings.duplicate_hash_bucket_slices)
        buckets:dict[tuple[int,str],list[int]]=defaultdict(list)
        for r in records:
            chunk=max(1,len(r.phash)//slices)
            for i in range(slices): buckets[(i,r.phash[i*chunk:(i+1)*chunk])].append(r.candidate_id)
        for ids in buckets.values():
            if len(ids)<=30:
                for a,b in itertools.combinations(sorted(set(ids)),2): keys.add((a,b))
        return sorted(keys)

    def _arrays(self,job_id:str,r:VisualFingerprintRecord):
        gray=np.asarray(decode_rgb(resolve_job_path(self._workspace,job_id,r.thumbnail_relative_path)).convert('L'),dtype=np.uint8)
        edge=np.asarray(decode_rgb(resolve_job_path(self._workspace,job_id,r.edge_map_relative_path)).convert('L'),dtype=np.uint8)>0
        return gray,edge

    def compare(self,job_id:str,a:VisualFingerprintRecord,b:VisualFingerprintRecord,content:dict[int,SemanticContentType])->ScreenshotPairSimilarity:
        if a.candidate_id>b.candidate_id: a,b=b,a
        gap=abs(a.selected_timestamp_seconds-b.selected_timestamp_seconds)
        if a.source_file_sha256==b.source_file_sha256:
            return ScreenshotPairSimilarity(candidate_a_id=a.candidate_id,candidate_b_id=b.candidate_id,timestamp_gap_seconds=gap,phash_distance=0,dhash_distance=0,thumbnail_ssim=1,edge_similarity=1,changed_pixel_ratio=0,a_to_b_edge_addition_ratio=0,b_to_a_edge_addition_ratio=0,content_type_compatible=True,content_compatibility_score=1,duplicate_score=1,decision=PairDecision.DUPLICATE,reason_codes=['IDENTICAL_FILE_SHA'])
        ph=normalized_hamming_distance(a.phash,b.phash); dh=normalized_hamming_distance(a.dhash,b.dhash)
        ga,ea=self._arrays(job_id,a); gb,eb=self._arrays(job_id,b)
        if gb.shape!=ga.shape: gb=np.asarray(Image.fromarray(gb).resize((ga.shape[1],ga.shape[0]), Image.Resampling.BILINEAR),dtype=np.uint8)
        if eb.shape!=ea.shape: eb=np.asarray(Image.fromarray(eb.astype(np.uint8)*255).resize((ea.shape[1],ea.shape[0]), Image.Resampling.NEAREST),dtype=np.uint8)>127
        ssim=float(max(0.0,min(1.0,structural_similarity(ga,gb,data_range=255))))
        inter=float(np.logical_and(ea,eb).sum()); union=float(np.logical_or(ea,eb).sum()); edge_sim=1.0 if union==0 else inter/union
        changed=float(np.mean(np.abs(ga.astype(np.int16)-gb.astype(np.int16))/255.0>self._settings.duplicate_changed_pixel_threshold))
        add_ab=float(np.logical_and(eb,~ea).sum()/max(1,eb.sum())); add_ba=float(np.logical_and(ea,~eb).sum()/max(1,ea.sum()))
        compat=self._compatibility(content.get(a.candidate_id,SemanticContentType.UNKNOWN),content.get(b.candidate_id,SemanticContentType.UNKNOWN))
        temporal=max(0.0,1.0-gap/max(self._settings.duplicate_primary_temporal_window_seconds,1e-9)) if gap<=self._settings.duplicate_primary_temporal_window_seconds else 0.15
        score=(ssim*self._settings.duplicate_structural_weight+edge_sim*self._settings.duplicate_edge_weight+(1-ph)*self._settings.duplicate_phash_weight+(1-dh)*self._settings.duplicate_dhash_weight+compat*self._settings.duplicate_content_weight+temporal*self._settings.duplicate_temporal_weight)
        score=float(max(0,min(1,score)))
        meaningful_add=max(add_ab,add_ba)>self._settings.duplicate_max_structural_addition_ratio
        structural_guard=ssim>=self._settings.duplicate_min_structural_similarity and edge_sim>=self._settings.duplicate_min_edge_similarity and not meaningful_add
        if score>=self._settings.duplicate_score_threshold and structural_guard: decision=PairDecision.DUPLICATE
        elif score>=self._settings.possible_duplicate_score_threshold or (ph<=self._settings.duplicate_long_range_phash_distance and ssim>=0.80): decision=PairDecision.POSSIBLE_DUPLICATE
        else: decision=PairDecision.DISTINCT
        reasons=[]
        if ph<=0.08: reasons.append('VERY_LOW_PHASH_DISTANCE')
        if dh<=0.10: reasons.append('VERY_LOW_DHASH_DISTANCE')
        if ssim>=self._settings.duplicate_min_structural_similarity: reasons.append('HIGH_STRUCTURAL_SIMILARITY')
        if edge_sim>=self._settings.duplicate_min_edge_similarity: reasons.append('HIGH_EDGE_SIMILARITY')
        if meaningful_add: reasons.append('MEANINGFUL_STRUCTURAL_ADDITION')
        if compat<0.5: reasons.append('CONTENT_TYPE_MISMATCH')
        if gap<=self._settings.duplicate_primary_temporal_window_seconds: reasons.append('TEMPORALLY_ADJACENT')
        else: reasons.append('LONG_RANGE_HASH_MATCH')
        return ScreenshotPairSimilarity(candidate_a_id=a.candidate_id,candidate_b_id=b.candidate_id,timestamp_gap_seconds=gap,phash_distance=ph,dhash_distance=dh,thumbnail_ssim=ssim,edge_similarity=edge_sim,changed_pixel_ratio=changed,a_to_b_edge_addition_ratio=add_ab,b_to_a_edge_addition_ratio=add_ba,content_type_compatible=compat>=0.5,content_compatibility_score=compat,duplicate_score=score,decision=decision,reason_codes=reasons)

    def _groups(self,records:list[VisualFingerprintRecord],pairs:list[ScreenshotPairSimilarity])->list[DuplicateGroup]:
        record={r.candidate_id:r for r in records}; dup={(p.candidate_a_id,p.candidate_b_id):p for p in pairs if p.decision is PairDecision.DUPLICATE}
        groups:list[list[int]]=[]
        # Conservative complete-linkage greedy grouping prevents A≈B≈C chaining overmerge.
        for cid in [r.candidate_id for r in sorted(records,key=lambda r:(r.selected_timestamp_seconds,r.candidate_id))]:
            placed=False
            for group in groups:
                if all(tuple(sorted((cid,m))) in dup for m in group):
                    group.append(cid); placed=True; break
            if not placed: groups.append([cid])
        result=[]
        for members in groups:
            members=sorted(members,key=lambda c:(record[c].selected_timestamp_seconds,c)); scores=[]
            for a,b in itertools.combinations(members,2): scores.append(dup[(min(a,b),max(a,b))].duplicate_score)
            gid='dup_'+canonical_fingerprint(members)[:12]
            result.append(DuplicateGroup(group_id=gid,candidate_ids=members,representative_candidate_id=members[0],group_size=len(members),earliest_timestamp_seconds=record[members[0]].selected_timestamp_seconds,latest_timestamp_seconds=record[members[-1]].selected_timestamp_seconds,minimum_internal_duplicate_score=min(scores) if scores else 1.0))
        return sorted(result,key=lambda g:(g.earliest_timestamp_seconds,min(g.candidate_ids)))

    def process(self,job_id:str,fingerprint_manifest:FingerprintManifest,*,cancel_check:Callable[[],bool]|None=None)->tuple[DuplicatePairsManifest,DuplicateGroupsManifest]:
        records=fingerprint_manifest.fingerprints; byid={r.candidate_id:r for r in records}; content=self._content_map(job_id)
        stable_windows=[r.stable_window_id for r in records]
        if len(stable_windows) != len(set(stable_windows)):
            raise DuplicateDetectionError('Phase-7 received multiple semantic winners for the same stable window.')
        pairs=[]
        for a,b in self._pair_keys(records):
            if cancel_check and cancel_check(): raise JobCancelledError('Duplicate detection cancelled.')
            pairs.append(self.compare(job_id,byid[a],byid[b],content))
        pair_stats=DuplicatePairStats(candidate_count=len(records),comparison_pair_count=len(pairs),duplicate_pair_count=sum(p.decision is PairDecision.DUPLICATE for p in pairs),possible_duplicate_pair_count=sum(p.decision is PairDecision.POSSIBLE_DUPLICATE for p in pairs),distinct_pair_count=sum(p.decision is PairDecision.DISTINCT for p in pairs))
        content_fp=canonical_fingerprint(sorted((cid,typ.value) for cid,typ in content.items()))
        pair_art=canonical_fingerprint({'fingerprints':fingerprint_manifest.artifact_fingerprint,'content':content_fp,'config':self.config_fingerprint(),'pairs':[p.model_dump(mode='json') for p in pairs]})
        pm=DuplicatePairsManifest(fingerprint_manifest_fingerprint=fingerprint_manifest.artifact_fingerprint,content_types_fingerprint=content_fp,config_fingerprint=self.config_fingerprint(),artifact_fingerprint=pair_art,stats=pair_stats,pairs=pairs)
        self._repository.save_duplicate_pairs(job_id,pm)
        groups=self._groups(records,pairs); sizes=[g.group_size for g in groups]; spans=[g.latest_timestamp_seconds-g.earliest_timestamp_seconds for g in groups if g.group_size>1]; removals=max(0,len(records)-len(groups)); ratio=removals/len(records) if records else 0
        warnings=[]
        if ratio>self._settings.phase7_high_duplicate_ratio: warnings.append('HIGH_DUPLICATE_RATIO')
        if any(g.group_size>=6 for g in groups): warnings.append('LARGE_DUPLICATE_GROUP')
        if any(g.latest_timestamp_seconds-g.earliest_timestamp_seconds>self._settings.duplicate_primary_temporal_window_seconds*2 for g in groups): warnings.append('LONG_RANGE_DUPLICATE_GROUP')
        if any(g.group_size>=3 and g.minimum_internal_duplicate_score<self._settings.duplicate_score_threshold+0.03 for g in groups): warnings.append('DUPLICATE_GROUP_CHAINING_RISK')
        gs=DuplicateGroupStats(candidate_count=len(records),duplicate_group_count=len(groups),singleton_group_count=sum(g.group_size==1 for g in groups),multi_member_group_count=sum(g.group_size>1 for g in groups),potential_duplicate_removals=removals,duplicate_ratio=ratio,mean_group_size=statistics.mean(sizes) if sizes else 0,max_group_size=max(sizes) if sizes else 0,mean_duplicate_group_time_span=statistics.mean(spans) if spans else 0,max_duplicate_group_time_span=max(spans) if spans else 0,warnings=warnings)
        possible=[p for p in pairs if p.decision is PairDecision.POSSIBLE_DUPLICATE]
        group_art=canonical_fingerprint({'pairs':pair_art,'fingerprints':fingerprint_manifest.artifact_fingerprint,'config':self.config_fingerprint(),'groups':[g.model_dump(mode='json') for g in groups]})
        gm=DuplicateGroupsManifest(duplicate_pairs_fingerprint=pair_art,fingerprint_manifest_fingerprint=fingerprint_manifest.artifact_fingerprint,config_fingerprint=self.config_fingerprint(),artifact_fingerprint=group_art,stats=gs,groups=groups,possible_duplicate_pairs=possible)
        self._repository.save_duplicate_groups(job_id,gm); self._checkpoints.mark_completed(job_id,CP_DUPLICATE_GROUPS_READY); return pm,gm

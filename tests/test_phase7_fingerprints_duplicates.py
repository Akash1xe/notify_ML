from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image, ImageDraw

from app.screenshots.fingerprints import compute_dhash, compute_phash, normalized_hamming_distance
from app.screenshots.models import PairDecision
from tests.phase7_helpers import build_phase7_context, build_phase7_pipeline


def test_perceptual_hashes_are_deterministic_and_distances_bounded():
    a=Image.new('RGB',(200,100),'white'); d=ImageDraw.Draw(a); d.text((20,30),'diagram A',fill='black')
    b=a.copy(); ImageDraw.Draw(b).ellipse((150,20,160,30),fill='black')
    assert compute_phash(a)==compute_phash(a)
    assert compute_dhash(a)==compute_dhash(a)
    dist=normalized_hamming_distance(compute_phash(a),compute_phash(b))
    assert 0<=dist<=1


def _single_records(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    _,_,repo,extractor,quality,fingerprints,duplicates,*_=build_phase7_pipeline(ctx,sem,backend)
    job=ctx[4].id; h=sem.load_phase7_handoff(job)[0]
    ex=asyncio.run(extractor.extract_candidate(job,h)); em=extractor.build_manifest(job,[h],[ex])
    q=asyncio.run(quality.validate_candidate(job,ex)); qm=quality.build_manifest(job,em,[q])
    f=fingerprints.generate_candidate(job,q); fm=fingerprints.build_manifest(job,qm,[f])
    return ctx,sem,repo,duplicates,fm,f


def test_fingerprint_stage_preserves_original_quality_image(tmp_path):
    ctx,sem,repo,duplicates,fm,f=_single_records(tmp_path)
    q=repo.load_quality_record(ctx[4].id,f.candidate_id)
    from app.screenshots.utils import file_sha256, resolve_job_path
    assert file_sha256(resolve_job_path(ctx[1],ctx[4].id,q.selected_relative_path))==q.selected_file_sha256
    assert f.phash and f.dhash and f.edge_hash


def test_duplicate_identical_sha_is_immediate_duplicate(tmp_path):
    ctx,sem,repo,detector,fm,f=_single_records(tmp_path)
    b=f.model_copy(update={'candidate_id':f.candidate_id+100,'stable_window_id':2,'selected_timestamp_seconds':f.selected_timestamp_seconds+20})
    content={f.candidate_id:sem.load_phase7_handoff(ctx[4].id)[0].content_type,b.candidate_id:sem.load_phase7_handoff(ctx[4].id)[0].content_type}
    pair=detector.compare(ctx[4].id,f,b,content)
    assert pair.decision is PairDecision.DUPLICATE
    assert pair.duplicate_score==1


def test_duplicate_grouping_prevents_transitive_chaining(tmp_path):
    ctx,sem,repo,detector,fm,f=_single_records(tmp_path)
    from app.screenshots.models import ScreenshotPairSimilarity
    def pair(a,b,score,decision):
        return ScreenshotPairSimilarity(candidate_a_id=a,candidate_b_id=b,timestamp_gap_seconds=1,phash_distance=.01,dhash_distance=.01,thumbnail_ssim=.95,edge_similarity=.9,changed_pixel_ratio=.01,a_to_b_edge_addition_ratio=.01,b_to_a_edge_addition_ratio=.01,content_type_compatible=True,content_compatibility_score=1,duplicate_score=score,decision=decision)
    r1=f.model_copy(update={'candidate_id':1,'selected_timestamp_seconds':1})
    r2=f.model_copy(update={'candidate_id':2,'selected_timestamp_seconds':2})
    r3=f.model_copy(update={'candidate_id':3,'selected_timestamp_seconds':3})
    groups=detector._groups([r1,r2,r3],[pair(1,2,.94,PairDecision.DUPLICATE),pair(2,3,.94,PairDecision.DUPLICATE),pair(1,3,.60,PairDecision.DISTINCT)])
    assert max(g.group_size for g in groups)==2
    assert len(groups)==2


def test_single_candidate_duplicate_stage_creates_singleton_group(tmp_path):
    ctx,sem,repo,detector,fm,f=_single_records(tmp_path)
    pairs,groups=detector.process(ctx[4].id,fm)
    assert pairs.stats.comparison_pair_count==0
    assert len(groups.groups)==1
    assert groups.groups[0].candidate_ids==[f.candidate_id]

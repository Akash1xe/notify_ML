from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from app.screenshots.models import ScreenshotQualityState
from tests.phase7_helpers import FakeFrameBackend, build_phase7_context, build_phase7_pipeline


def test_exact_source_extraction_preserves_timestamp_and_resolution(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    _,_,repo,extractor,*_=build_phase7_pipeline(ctx,sem,backend)
    job=ctx[4].id; handoff=sem.load_phase7_handoff(job)[0]
    rec=asyncio.run(extractor.extract_candidate(job,handoff))
    assert backend.calls==[handoff.candidate_timestamp_seconds]
    assert (rec.width,rec.height)==(640,360)
    assert rec.requested_timestamp_seconds==handoff.candidate_timestamp_seconds
    assert rec.file_sha256
    assert ctx[1].screenshot_extraction_path(job,handoff.candidate_id).exists()


def test_extraction_cache_fingerprint_ignores_semantic_score(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    _,_,_,extractor,*_=build_phase7_pipeline(ctx,sem,backend)
    h=sem.load_phase7_handoff(ctx[4].id)[0]
    fp=extractor.expected_fingerprint(h,extractor._source(ctx[4].id)[1])
    changed=h.model_copy(update={'semantic_decision_score':0.01})
    assert extractor.expected_fingerprint(changed,extractor._source(ctx[4].id)[1])==fp


def test_quality_good_exact_frame_does_not_search(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    _,_,_,extractor,quality,*_=build_phase7_pipeline(ctx,sem,backend)
    job=ctx[4].id; h=sem.load_phase7_handoff(job)[0]
    ex=asyncio.run(extractor.extract_candidate(job,h)); before=len(backend.calls)
    q=asyncio.run(quality.validate_candidate(job,ex))
    assert q.quality_state in {ScreenshotQualityState.GOOD, ScreenshotQualityState.ACCEPTABLE}
    assert not q.used_timestamp_fallback
    assert len(backend.calls)==before


def test_quality_search_chooses_sharper_nearby_frame(tmp_path):
    def factory(ts:float):
        img=FakeFrameBackend.default_image(ts)
        if abs(ts-12.0)<1e-6:
            img=img.filter(ImageFilter.GaussianBlur(radius=8))
        return img
    backend=FakeFrameBackend(factory)
    ctx,sem,backend,_=build_phase7_context(tmp_path,settings_overrides={'screenshot_blur_score_threshold':0.55,'screenshot_min_quality_score':0.65,'screenshot_good_quality_score':0.75,'screenshot_quality_search_radius_seconds':0.3,'screenshot_quality_search_step_seconds':0.1},backend=backend)
    _,_,_,extractor,quality,*_=build_phase7_pipeline(ctx,sem,backend)
    job=ctx[4].id; h=sem.load_phase7_handoff(job)[0]
    ex=asyncio.run(extractor.extract_candidate(job,h)); q=asyncio.run(quality.validate_candidate(job,ex))
    assert len(q.attempts)>1
    assert q.used_timestamp_fallback
    assert abs(q.timestamp_shift_seconds)<=0.3+1e-9
    assert q.selected_quality_score>q.exact_quality_score


def test_whiteboard_is_not_misclassified_as_blank(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    _,_,_,_,quality,*_=build_phase7_pipeline(ctx,sem,backend)
    p=tmp_path/'whiteboard.png'; FakeFrameBackend.default_image(1).save(p)
    metrics=quality.measure(p)
    assert not metrics.is_black_or_blank
    assert metrics.edge_density>0


def test_dark_code_like_screen_is_not_automatically_invalid(tmp_path):
    ctx,sem,backend,_=build_phase7_context(tmp_path)
    _,_,_,_,quality,*_=build_phase7_pipeline(ctx,sem,backend)
    p=tmp_path/'dark.png'; img=Image.new('RGB',(640,360),(18,18,18)); d=ImageDraw.Draw(img)
    for i in range(15): d.text((30,20+i*20),f'const value{i} = fn({i});',fill=(220,220,220))
    img.save(p)
    metrics=quality.measure(p)
    assert not metrics.is_black_or_blank
    assert metrics.quality_state is not ScreenshotQualityState.INVALID

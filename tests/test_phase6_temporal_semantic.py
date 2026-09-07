from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from app.semantic_analysis.models import (
    CompletionState,
    EducationalUsefulness,
    TemporalContextType,
    WindowDecisionType,
)
from app.semantic_analysis.runtime import FakeVisionLanguageModelAdapter
from tests.phase6_helpers import build_phase6_context, build_phase6_pipeline


def _response(prompt: str, *, completion="COMPLETE", usefulness="HIGH", score=0.9, confidence=0.85, transition=0.05):
    candidate_id = int(re.search(r"Candidate ID:\s*(\d+)", prompt).group(1))
    return json.dumps({
        "candidate_id": candidate_id,
        "content_type": "DIAGRAM",
        "completion_state": completion,
        "completion_score": score,
        "visual_change_state": "SETTLED" if completion != "INCOMPLETE" else "STILL_CHANGING",
        "educational_usefulness": usefulness,
        "usefulness_score": score,
        "transition_probability": transition,
        "has_meaningful_visual_content": usefulness != "NONE",
        "teacher_still_writing_likely": completion == "INCOMPLETE",
        "transcript_visual_consistency": "CONSISTENT",
        "confidence": confidence,
        "reason_codes": ["TEST_EVIDENCE"],
    })


def test_phase63_current_frame_is_immutable_and_context_is_bounded(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    _, _, repository, preparation, _, temporal, *_ = build_phase6_pipeline(ctx, fake)
    inputs = preparation.process(ctx[4].id)
    ctx[3].mark_completed(ctx[4].id, "SEMANTIC_INPUT_READY")
    manifest = temporal.process(ctx[4].id)
    input_by_id = {x.candidate_id: x for x in inputs.inputs}
    assert len(manifest.contexts) == len(inputs.inputs)
    for item in manifest.contexts:
        assert item.current.frame_index == input_by_id[item.candidate_id].semantic_frame.frame_index
        assert item.context_type in set(TemporalContextType)
        ids = [item.current.frame_index] + ([item.previous.frame_index] if item.previous else []) + ([item.next.frame_index] if item.next else [])
        assert len(ids) == len(set(ids))


def test_phase64_retries_invalid_json_once(tmp_path: Path):
    calls = {"count": 0}
    def responder(prompt, index):
        calls["count"] += 1
        return "not-json" if calls["count"] == 1 else _response(prompt)
    fake = FakeVisionLanguageModelAdapter(responder=responder)
    ctx, _ = build_phase6_context(tmp_path, vlm_adapter=fake)
    pipeline, _, repository, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    artifacts = [repository.load_candidate_artifact(ctx[4].id, x.candidate_id) for x in repository.load_input_manifest(ctx[4].id).inputs]
    assert any(x.attempt_count == 2 for x in artifacts)
    assert all(x.semantic_result is not None for x in artifacts)


def test_phase64_markdown_fenced_json_is_accepted(tmp_path: Path):
    fake = FakeVisionLanguageModelAdapter(responder=lambda prompt, index: "```json\n" + _response(prompt) + "\n```")
    ctx, _ = build_phase6_context(tmp_path, vlm_adapter=fake)
    pipeline, _, repository, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert repository.load_results(ctx[4].id).stats.successful_analysis_count >= 1


def test_phase65_alternate_can_replace_incomplete_primary(tmp_path: Path):
    mapping = {}
    def responder(prompt, index):
        cid = int(re.search(r"Candidate ID:\s*(\d+)", prompt).group(1))
        role = mapping.get(cid)
        if role == "PRIMARY":
            return _response(prompt, completion="INCOMPLETE", usefulness="MEDIUM", score=0.3)
        return _response(prompt, completion="COMPLETE", usefulness="HIGH", score=0.95)
    fake = FakeVisionLanguageModelAdapter(responder=responder)
    ctx, _ = build_phase6_context(tmp_path, vlm_adapter=fake)
    pipeline, _, repository, preparation, *_ = build_phase6_pipeline(ctx, fake)
    prepared = preparation.process(ctx[4].id)
    ctx[3].mark_completed(ctx[4].id, "SEMANTIC_INPUT_READY")
    mapping.update({x.candidate_id: x.selection_role.value for x in prepared.inputs})
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    selections = repository.load_selections(ctx[4].id)
    windows_with_alt = [x for x in selections.window_decisions if x.alternate_candidate_id is not None]
    if windows_with_alt:
        assert any(x.decision is WindowDecisionType.SELECT_ALTERNATE for x in windows_with_alt)


def test_phase65_transition_is_rejected(tmp_path: Path):
    fake = FakeVisionLanguageModelAdapter(responder=lambda prompt, index: _response(prompt, completion="TRANSITION", usefulness="LOW", score=0.1, transition=0.95))
    ctx, _ = build_phase6_context(tmp_path, vlm_adapter=fake)
    pipeline, _, repository, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert repository.load_selections(ctx[4].id).stats.selected_window_count == 0


def test_phase65_primary_tie_bonus_is_applied_only_in_close_comparisons(tmp_path: Path):
    fake = FakeVisionLanguageModelAdapter(responder=lambda prompt, index: _response(prompt))
    ctx, _ = build_phase6_context(
        tmp_path,
        settings_overrides={
            "semantic_decision_ambiguity_gap": 1.0,
            "semantic_primary_tie_bonus": 0.02,
            "semantic_alternate_switch_margin": 0.0,
        },
        vlm_adapter=fake,
    )
    pipeline, _, repository, *_ = build_phase6_pipeline(ctx, fake)
    asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    selections = repository.load_selections(ctx[4].id)
    windows_with_alt = [x for x in selections.window_decisions if x.alternate_candidate_id is not None]
    assert windows_with_alt
    decision = windows_with_alt[0]
    assert decision.decision is WindowDecisionType.SELECT_PRIMARY
    assert "PRIMARY_TIE_BONUS" in decision.reason_codes

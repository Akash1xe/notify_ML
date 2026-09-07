from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import AppSettings
from app.core.exceptions import VLMOutOfMemoryError, VLMUnsupportedConfigurationError
from app.semantic_analysis.hardware import HardwareCapability
from app.semantic_analysis.models import Phase6ResumeStage
from app.semantic_analysis.runtime import FakeVisionLanguageModelAdapter, QwenRuntimeManager
from tests.phase6_helpers import build_phase6_context, build_phase6_pipeline


def test_phase61_semantic_inputs_join_phase4_and_phase5(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    _, _, repository, preparation, *_ = build_phase6_pipeline(ctx, fake)
    manifest = preparation.process(ctx[4].id)
    assert manifest.stats.input_count >= 1
    assert manifest.stats.input_count == manifest.stats.primary_input_count + manifest.stats.alternate_input_count
    assert all(item.semantic_frame.relative_path == item.processed_frame_path for item in manifest.inputs)
    assert all(item.comparison_group_id.startswith("window:") for item in manifest.inputs)


def test_phase61_missing_frame_fails_safely(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    _, _, _, preparation, *_ = build_phase6_pipeline(ctx, fake)
    handoff = preparation._candidates.load_handoff(ctx[4].id)
    assert handoff
    (ctx[1].workspace(ctx[4].id) / handoff[0].processed_frame_path).unlink()
    with pytest.raises(Exception):
        preparation.process(ctx[4].id)


def test_runtime_is_lazy_and_reuses_loaded_fake_model():
    settings = AppSettings(qwen_vl_device="cpu")
    fake = FakeVisionLanguageModelAdapter()
    runtime = QwenRuntimeManager(
        settings,
        hardware=HardwareCapability(torch_available=False, cuda_available=False),
        adapter_factory=lambda *_: fake,
    )
    assert not runtime.is_loaded()
    assert runtime.inspect_runtime().model_loaded is False
    image = Image.new("RGB", (100, 100))
    runtime.generate([image], "Candidate ID: 1")
    runtime.generate([image], "Candidate ID: 2")
    assert fake.load_calls == 1
    assert fake.generate_calls == 2


def test_runtime_explicit_cuda_requires_cuda():
    settings = AppSettings(qwen_vl_device="cuda")
    with pytest.raises(VLMUnsupportedConfigurationError):
        QwenRuntimeManager(settings, hardware=HardwareCapability(torch_available=False, cuda_available=False))


def test_runtime_falls_back_after_primary_oom():
    settings = AppSettings(qwen_vl_device="cpu", qwen_vl_enable_fallback=True)
    primary = FakeVisionLanguageModelAdapter(fail_load=VLMOutOfMemoryError("oom"))
    fallback = FakeVisionLanguageModelAdapter()
    def factory(model, *_):
        return primary if model.endswith("4B-Instruct") else fallback
    runtime = QwenRuntimeManager(
        settings,
        hardware=HardwareCapability(torch_available=False, cuda_available=False),
        adapter_factory=factory,
    )
    result = runtime.generate([Image.new("RGB", (64, 64))], "Candidate ID: 1")
    assert result.fallback_used is True
    assert result.model_name.endswith("2B-Instruct")
    assert primary.load_calls == 1 and fallback.load_calls == 1


def test_phase6_pipeline_reaches_ready_with_fake_vlm(tmp_path: Path):
    ctx, fake = build_phase6_context(tmp_path)
    pipeline, cache, repository, *_ = build_phase6_pipeline(ctx, fake)
    summary = asyncio.run(pipeline.process(ctx[4].id, finalize_job=False))
    assert summary.semantic_input_count >= 1
    assert repository.load_selections(ctx[4].id).stats.window_count >= 1
    assert cache.inspect(ctx[4].id).resume_plan.resume_stage is Phase6ResumeStage.PHASE6_READY


def test_runtime_falls_back_after_inference_oom():
    settings = AppSettings(qwen_vl_device="cpu", qwen_vl_enable_fallback=True)

    class OOMGenerateAdapter(FakeVisionLanguageModelAdapter):
        def generate(self, *args, **kwargs):
            self.generate_calls += 1
            raise VLMOutOfMemoryError("inference oom")

    primary = OOMGenerateAdapter()
    fallback = FakeVisionLanguageModelAdapter()

    def factory(model, *_):
        return primary if model.endswith("4B-Instruct") else fallback

    runtime = QwenRuntimeManager(
        settings,
        hardware=HardwareCapability(torch_available=False, cuda_available=False),
        adapter_factory=factory,
    )
    result = runtime.generate([Image.new("RGB", (64, 64))], "Candidate ID: 1")
    assert result.fallback_used is True
    assert result.fallback_reason == "INFERENCE_OOM"
    assert result.model_name.endswith("2B-Instruct")

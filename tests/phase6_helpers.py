from __future__ import annotations

import asyncio
from pathlib import Path

from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.logging import JobEventLogger
from app.semantic_analysis.analysis import SemanticCandidateAnalyzer
from app.semantic_analysis.cache import SemanticCacheCoordinator
from app.semantic_analysis.decision import SemanticDecisionEngine
from app.semantic_analysis.evaluation import Phase6Evaluator
from app.semantic_analysis.hardware import HardwareCapability
from app.semantic_analysis.pipeline import SemanticPipeline
from app.semantic_analysis.preparation import SemanticInputPreparationService
from app.semantic_analysis.prompts import SemanticPromptBuilder
from app.semantic_analysis.repository import SemanticRepository
from app.semantic_analysis.runtime import FakeVisionLanguageModelAdapter, QwenRuntimeManager
from app.semantic_analysis.temporal_context import TemporalVisualContextService
from app.transcription.repository import TranscriptionRepository
from app.video_analysis.repository import FrameAnalysisRepository
from tests.phase5_helpers import build_phase5_context, build_phase5_pipeline


def build_phase6_context(tmp_path: Path, *, settings_overrides=None, vlm_adapter=None, frame_kinds=None):
    overrides = {"processor_mode": "semantic", "qwen_vl_device": "cpu"}
    overrides.update(settings_overrides or {})
    ctx, whisper = build_phase5_context(tmp_path, settings_overrides=overrides)
    phase5_pipeline, *_ = build_phase5_pipeline(ctx, whisper)
    asyncio.run(phase5_pipeline.process(ctx[4].id, finalize_job=False))
    fake = vlm_adapter or FakeVisionLanguageModelAdapter()
    return ctx, fake


def build_phase6_pipeline(ctx, fake: FakeVisionLanguageModelAdapter):
    settings, workspace, jobs, checkpoints, *_ = ctx
    candidate_repo = CandidateAnalysisRepository(workspace)
    transcript_repo = TranscriptionRepository(workspace)
    frame_repo = FrameAnalysisRepository(workspace)
    semantic_repo = SemanticRepository(workspace)
    preparation = SemanticInputPreparationService(
        settings, workspace, checkpoints, semantic_repo, candidate_repo, transcript_repo, frame_repo
    )
    runtime = QwenRuntimeManager(
        settings,
        hardware=HardwareCapability(torch_available=False, cuda_available=False, system_ram_total_mb=8192, system_ram_available_mb=4096, cpu_count=4),
        adapter_factory=lambda model, device, dtype: fake,
    )
    prompt_builder = SemanticPromptBuilder(settings)
    temporal = TemporalVisualContextService(
        settings, workspace, checkpoints, semantic_repo, frame_repo, candidate_repo
    )
    analyzer = SemanticCandidateAnalyzer(
        settings, workspace, checkpoints, semantic_repo, runtime, prompt_builder
    )
    decision = SemanticDecisionEngine(settings, checkpoints, semantic_repo, candidate_repo)
    cache = SemanticCacheCoordinator(
        workspace,
        checkpoints,
        semantic_repo,
        candidate_repo,
        transcript_repo,
        frame_repo,
        preparation,
        temporal,
        analyzer,
        decision,
    )
    pipeline = SemanticPipeline(
        jobs=jobs,
        checkpoints=checkpoints,
        events=JobEventLogger(workspace),
        preparation=preparation,
        temporal_context=temporal,
        analyzer=analyzer,
        decision=decision,
        cache=cache,
        repository=semantic_repo,
    )
    evaluator = Phase6Evaluator(settings, semantic_repo)
    return pipeline, cache, semantic_repo, preparation, runtime, temporal, analyzer, decision, evaluator

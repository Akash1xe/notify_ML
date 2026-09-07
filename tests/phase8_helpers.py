from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.logging import JobEventLogger
from app.document.cache import DocumentCacheCoordinator
from app.document.input import DocumentInputBuilder
from app.document.layout import DocumentLayoutEngine
from app.document.pdf import DocumentPdfGenerator
from app.document.pipeline import DocumentPipeline
from app.document.render import DocumentRenderPlanner
from app.document.repository import DocumentRepository
from app.document.results import DocumentResultService
from app.screenshots.repository import ScreenshotRepository
from tests.phase7_helpers import build_phase7_context, build_phase7_pipeline


def build_phase8_context(tmp_path: Path, *, settings_overrides=None):
    ctx, semantic_repo, backend, fake_vlm = build_phase7_context(tmp_path, settings_overrides=settings_overrides)
    phase7, *_ = build_phase7_pipeline(ctx, semantic_repo, backend)
    asyncio.run(phase7.process(ctx[4].id, finalize_job=False))
    settings, workspace, jobs, checkpoints, *_ = ctx
    screenshots = ScreenshotRepository(workspace)
    repository = DocumentRepository(workspace)
    input_builder = DocumentInputBuilder(settings, workspace, checkpoints, screenshots, repository)
    layout = DocumentLayoutEngine(settings, checkpoints, repository)
    render = DocumentRenderPlanner(settings, checkpoints, repository)
    pdf = DocumentPdfGenerator(settings, workspace, checkpoints, repository)
    cache = DocumentCacheCoordinator(workspace, checkpoints, screenshots, repository, input_builder, layout, render, pdf)
    pipeline = DocumentPipeline(
        jobs=jobs, checkpoints=checkpoints, events=JobEventLogger(workspace), screenshots=screenshots,
        repository=repository, cache=cache, input_builder=input_builder, layout=layout, render=render, pdf=pdf,
    )
    results = DocumentResultService(settings, jobs, workspace, checkpoints, repository, screenshots, cache)
    return ctx, repository, input_builder, layout, render, pdf, cache, pipeline, results, backend, fake_vlm

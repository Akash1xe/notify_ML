from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from app.core.logging import JobEventLogger
from app.screenshots.cache import Phase7CacheCoordinator
from app.screenshots.duplicates import CrossWindowDuplicateDetector
from app.screenshots.evaluation import Phase7Evaluator
from app.screenshots.extraction import ExtractedFrameInfo, SourceScreenshotExtractor
from app.screenshots.fingerprints import VisualFingerprintGenerator
from app.screenshots.pipeline import ScreenshotPipeline
from app.screenshots.quality import ScreenshotQualityValidator
from app.screenshots.repository import ScreenshotRepository
from app.screenshots.selection import FinalScreenshotSelector
from tests.phase6_helpers import build_phase6_context, build_phase6_pipeline


class FakeFrameBackend:
    def __init__(self, image_factory=None):
        self.calls: list[float] = []
        self.image_factory = image_factory or self.default_image

    @staticmethod
    def default_image(timestamp: float) -> Image.Image:
        image = Image.new('RGB', (640, 360), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 600, 320), outline='black', width=4)
        draw.text((70, 80), f'Lecture state {timestamp:.2f}', fill='black')
        for y in range(120, 290, 35):
            draw.line((70, y, 550, y), fill='black', width=3)
        return image

    def extract(self, *, source, timestamp_seconds, output, timeout_seconds, cancel_check=None):
        if cancel_check and cancel_check():
            from app.core.exceptions import JobCancelledError
            raise JobCancelledError('cancelled')
        self.calls.append(timestamp_seconds)
        output.parent.mkdir(parents=True, exist_ok=True)
        image = self.image_factory(timestamp_seconds)
        image.save(output, 'PNG')
        return ExtractedFrameInfo(width=image.width, height=image.height, resolved_timestamp_seconds=timestamp_seconds)


def build_phase7_context(tmp_path: Path, *, settings_overrides=None, backend=None):
    overrides={'processor_mode':'screenshots','screenshot_min_quality_score':0.30,'screenshot_good_quality_score':0.45}
    overrides.update(settings_overrides or {})
    ctx, fake_vlm = build_phase6_context(tmp_path, settings_overrides=overrides)
    semantic_pipeline, _, semantic_repo, *_ = build_phase6_pipeline(ctx, fake_vlm)
    asyncio.run(semantic_pipeline.process(ctx[4].id, finalize_job=False))
    return ctx, semantic_repo, backend or FakeFrameBackend(), fake_vlm


def build_phase7_pipeline(ctx, semantic_repo, backend):
    settings, workspace, jobs, checkpoints, *_ = ctx
    repo=ScreenshotRepository(workspace)
    extractor=SourceScreenshotExtractor(settings,workspace,checkpoints,repo,semantic_repo,backend=backend)
    quality=ScreenshotQualityValidator(settings,workspace,checkpoints,repo,semantic_repo,extraction_backend=backend)
    fingerprints=VisualFingerprintGenerator(settings,workspace,checkpoints,repo)
    duplicates=CrossWindowDuplicateDetector(settings,workspace,checkpoints,repo,semantic_repo)
    selector=FinalScreenshotSelector(settings,workspace,checkpoints,repo,semantic_repo)
    cache=Phase7CacheCoordinator(workspace,checkpoints,repo,semantic_repo,extractor,quality,fingerprints,duplicates,selector)
    evaluator=Phase7Evaluator(settings,repo,semantic_repo)
    pipeline=ScreenshotPipeline(jobs=jobs,checkpoints=checkpoints,events=JobEventLogger(workspace),semantic_repository=semantic_repo,repository=repo,extractor=extractor,quality=quality,fingerprints=fingerprints,duplicates=duplicates,selector=selector,cache=cache,evaluator=evaluator)
    return pipeline,cache,repo,extractor,quality,fingerprints,duplicates,selector,evaluator

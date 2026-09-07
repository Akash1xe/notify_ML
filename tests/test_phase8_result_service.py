from __future__ import annotations

import asyncio

from app.document.models import DocumentApiStatus
from tests.phase8_helpers import build_phase8_context


def test_result_service_returns_summary_screenshots_download_and_preview(tmp_path):
    ctx, _, _, _, _, _, _, pipeline, results, *_ = build_phase8_context(tmp_path)
    job_id = ctx[4].id
    asyncio.run(pipeline.process(job_id, finalize_job=False))
    status = results.status(job_id)
    assert status.status is DocumentApiStatus.READY
    summary = results.summary(job_id)
    shots = results.screenshots(job_id, offset=0, limit=100)
    descriptor = results.download(job_id)
    preview, mime = results.preview_path(job_id, shots.items[0].candidate_id)
    assert summary.page_count == 1
    assert shots.total == 1 and shots.items[0].order == 1
    assert descriptor.filename.endswith(".pdf") and descriptor.etag.startswith('"sha256-')
    assert preview.exists() and mime.startswith("image/")

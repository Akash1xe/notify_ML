from __future__ import annotations

import asyncio
import json

import pytest
from pypdf import PdfReader

from app.core.exceptions import DocumentInputIntegrityError, PdfEmptyDocumentError
from app.document.models import DocumentApiStatus, DocumentResumeStage
from app.document.utils import format_timestamp, safe_download_filename
from tests.phase8_helpers import build_phase8_context


def test_phase8_end_to_end_generates_valid_pdf_and_reuses_cache(tmp_path):
    ctx, repo, _, _, _, _, cache, pipeline, results, *_ = build_phase8_context(tmp_path)
    job_id = ctx[4].id
    first = asyncio.run(pipeline.process(job_id, finalize_job=False))
    assert first.document_ready is True
    assert first.page_count == first.final_screenshot_count == 1
    reader = PdfReader(str(ctx[1].document_pdf_path(job_id)))
    assert len(reader.pages) == 1
    snapshot = cache.inspect(job_id, deep_pdf=True)
    assert snapshot.resume_plan.resume_stage is DocumentResumeStage.DOCUMENT_READY
    second = asyncio.run(pipeline.process(job_id, finalize_job=False))
    assert second.cache_hits == {"document_input": True, "layout": True, "render_plan": True, "pdf": True}
    status = results.status(job_id)
    # FINAL_DOCUMENT_READY is set by pipeline even when job finalization is disabled.
    assert status.status is DocumentApiStatus.READY
    assert status.download_available is True


def test_phase8_invalidation_boundaries(tmp_path):
    ctx, _, _, _, _, _, cache, pipeline, *_ = build_phase8_context(tmp_path)
    job_id = ctx[4].id
    asyncio.run(pipeline.process(job_id, finalize_job=False))
    settings = ctx[0]

    settings.pdf_jpeg_quality = 88
    snap = cache.inspect(job_id, deep_pdf=False)
    assert snap.resume_plan.resume_stage is DocumentResumeStage.PDF_GENERATION
    settings.pdf_jpeg_quality = 92

    settings.pdf_caption_max_characters = 120
    snap = cache.inspect(job_id, deep_pdf=False)
    assert snap.resume_plan.resume_stage is DocumentResumeStage.DOCUMENT_RENDER_PLAN
    settings.pdf_caption_max_characters = 300

    settings.pdf_margin_left_mm = 15
    snap = cache.inspect(job_id, deep_pdf=False)
    assert snap.resume_plan.resume_stage is DocumentResumeStage.DOCUMENT_LAYOUT


def test_source_title_change_invalidates_document_input(tmp_path):
    ctx, _, _, _, _, _, cache, pipeline, *_ = build_phase8_context(tmp_path)
    job_id = ctx[4].id
    asyncio.run(pipeline.process(job_id, finalize_job=False))
    path = ctx[1].ingestion_path(job_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["title"] = "Changed Lecture Title"
    path.write_text(json.dumps(payload), encoding="utf-8")
    snap = cache.inspect(job_id, deep_pdf=False)
    assert snap.resume_plan.resume_stage is DocumentResumeStage.DOCUMENT_INPUT


def test_corrupt_pdf_only_invalidates_pdf_stage(tmp_path):
    ctx, _, _, _, _, _, cache, pipeline, *_ = build_phase8_context(tmp_path)
    job_id = ctx[4].id
    asyncio.run(pipeline.process(job_id, finalize_job=False))
    pdf = ctx[1].document_pdf_path(job_id)
    data = bytearray(pdf.read_bytes())
    data[-10] ^= 1
    pdf.write_bytes(data)
    # lightweight status intentionally avoids hashing; deep integrity catches tampering.
    snap = cache.inspect(job_id, deep_pdf=True)
    assert snap.resume_plan.resume_stage is DocumentResumeStage.PDF_GENERATION


def test_document_helpers_are_deterministic_and_filename_safe():
    assert format_timestamp(641.2) == "10:41"
    assert format_timestamp(3661) == "01:01:01"
    name = safe_download_filename("../../evil\r\nContent-Type: text/html")
    assert name.endswith(".pdf")
    assert "/" not in name and "\\" not in name and "\r" not in name and "\n" not in name


def test_missing_final_image_is_rejected_by_document_input(tmp_path):
    ctx, repo, input_builder, *_ = build_phase8_context(tmp_path)
    job_id = ctx[4].id
    selection = ctx[1].screenshot_final_selections_path(job_id)
    manifest = json.loads(selection.read_text(encoding="utf-8"))
    image = ctx[1].workspace(job_id) / manifest["final_screenshots"][0]["image_relative_path"]
    image.unlink()
    with pytest.raises(DocumentInputIntegrityError):
        input_builder.build(job_id)


def test_zero_final_screenshots_builds_empty_manifests_then_pdf_fails_explicitly(tmp_path):
    ctx, repo, input_builder, layout, render, pdf, *_ = build_phase8_context(tmp_path)
    job_id = ctx[4].id
    selection_path = ctx[1].screenshot_final_selections_path(job_id)
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    payload["final_screenshots"] = []
    payload["stats"]["final_screenshot_count"] = 0
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    input_manifest = input_builder.build(job_id)
    assert input_manifest.items == []
    layout_manifest = layout.build(job_id, input_manifest)
    assert layout_manifest.pages == []
    render_manifest = render.build(job_id, input_manifest, layout_manifest)
    assert render_manifest.pages == []
    with pytest.raises(PdfEmptyDocumentError):
        pdf.generate(job_id, input_manifest, render_manifest)

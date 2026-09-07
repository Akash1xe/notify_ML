from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import DocumentPipelineError
from app.document.models import (
    DocumentInputManifest,
    DocumentLayoutManifest,
    DocumentPdfManifest,
    DocumentRenderManifest,
    FinalDocumentSummary,
)
from app.storage.workspace import WorkspaceManager, atomic_write_json

T = TypeVar("T", bound=BaseModel)


class DocumentRepository:
    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    @staticmethod
    def _load(path: Path, model: type[T]) -> T:
        try:
            return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise DocumentPipelineError(f"Unable to load {path.name}.") from exc

    @staticmethod
    def _save(path: Path, value: BaseModel) -> None:
        atomic_write_json(path, value.model_dump(mode="json"))

    def load_input_manifest(self, job_id: str) -> DocumentInputManifest:
        return self._load(self._workspace.document_input_manifest_path(job_id), DocumentInputManifest)

    def save_input_manifest(self, job_id: str, value: DocumentInputManifest) -> None:
        self._save(self._workspace.document_input_manifest_path(job_id), value)

    def load_document_items(self, job_id: str):
        return list(self.load_input_manifest(job_id).items)

    def get_document_item(self, job_id: str, candidate_id: int):
        for item in self.load_input_manifest(job_id).items:
            if item.candidate_id == candidate_id:
                return item
        raise DocumentPipelineError("Document candidate is not present in the input manifest.")

    def load_layout_manifest(self, job_id: str) -> DocumentLayoutManifest:
        return self._load(self._workspace.document_layout_path(job_id), DocumentLayoutManifest)

    def save_layout_manifest(self, job_id: str, value: DocumentLayoutManifest) -> None:
        self._save(self._workspace.document_layout_path(job_id), value)

    def get_page(self, job_id: str, page_number: int):
        for page in self.load_layout_manifest(job_id).pages:
            if page.page_number == page_number:
                return page
        raise DocumentPipelineError("Document page is not available.")

    def load_render_manifest(self, job_id: str) -> DocumentRenderManifest:
        return self._load(self._workspace.document_render_plan_path(job_id), DocumentRenderManifest)

    def save_render_manifest(self, job_id: str, value: DocumentRenderManifest) -> None:
        self._save(self._workspace.document_render_plan_path(job_id), value)

    def load_pdf_manifest(self, job_id: str) -> DocumentPdfManifest:
        return self._load(self._workspace.document_pdf_manifest_path(job_id), DocumentPdfManifest)

    def save_pdf_manifest(self, job_id: str, value: DocumentPdfManifest) -> None:
        self._save(self._workspace.document_pdf_manifest_path(job_id), value)

    def load_summary(self, job_id: str) -> FinalDocumentSummary:
        return self._load(self._workspace.document_summary_path(job_id), FinalDocumentSummary)

    def save_summary(self, job_id: str, value: FinalDocumentSummary) -> None:
        self._save(self._workspace.document_summary_path(job_id), value)

    def get_pdf_path(self, job_id: str) -> Path:
        return self._workspace.document_pdf_path(job_id)

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import Phase7ValidationError
from app.screenshots.models import (
    DuplicateGroupsManifest,
    DuplicatePairsManifest,
    ExtractionManifest,
    FinalSelectionsManifest,
    FingerprintManifest,
    Phase7EvaluationReport,
    Phase7Summary,
    QualityManifest,
    QualitySelectionRecord,
    SourceScreenshotRecord,
    VisualFingerprintRecord,
)
from app.storage.workspace import WorkspaceManager, atomic_write_json

T = TypeVar('T', bound=BaseModel)


class ScreenshotRepository:
    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    @staticmethod
    def _load(path: Path, model: type[T]) -> T:
        try:
            return model.model_validate(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise Phase7ValidationError(f'Unable to load {path.name}.') from exc

    @staticmethod
    def _save(path: Path, value: BaseModel) -> None:
        atomic_write_json(path, value.model_dump(mode='json'))

    def load_extraction_record(self, job_id: str, candidate_id: int) -> SourceScreenshotRecord:
        return self._load(self._workspace.screenshot_extraction_record_path(job_id, candidate_id), SourceScreenshotRecord)

    def save_extraction_record(self, job_id: str, value: SourceScreenshotRecord) -> None:
        self._save(self._workspace.screenshot_extraction_record_path(job_id, value.candidate_id), value)

    def load_extraction_manifest(self, job_id: str) -> ExtractionManifest:
        return self._load(self._workspace.screenshot_extraction_manifest_path(job_id), ExtractionManifest)

    def save_extraction_manifest(self, job_id: str, value: ExtractionManifest) -> None:
        self._save(self._workspace.screenshot_extraction_manifest_path(job_id), value)

    def load_quality_record(self, job_id: str, candidate_id: int) -> QualitySelectionRecord:
        return self._load(self._workspace.screenshot_quality_record_path(job_id, candidate_id), QualitySelectionRecord)

    def save_quality_record(self, job_id: str, value: QualitySelectionRecord) -> None:
        self._save(self._workspace.screenshot_quality_record_path(job_id, value.candidate_id), value)

    def load_quality_manifest(self, job_id: str) -> QualityManifest:
        return self._load(self._workspace.screenshot_quality_manifest_path(job_id), QualityManifest)

    def save_quality_manifest(self, job_id: str, value: QualityManifest) -> None:
        self._save(self._workspace.screenshot_quality_manifest_path(job_id), value)

    def load_fingerprint_record(self, job_id: str, candidate_id: int) -> VisualFingerprintRecord:
        return self._load(self._workspace.screenshot_fingerprint_record_path(job_id, candidate_id), VisualFingerprintRecord)

    def save_fingerprint_record(self, job_id: str, value: VisualFingerprintRecord) -> None:
        self._save(self._workspace.screenshot_fingerprint_record_path(job_id, value.candidate_id), value)

    def load_fingerprint_manifest(self, job_id: str) -> FingerprintManifest:
        return self._load(self._workspace.screenshot_fingerprint_manifest_path(job_id), FingerprintManifest)

    def save_fingerprint_manifest(self, job_id: str, value: FingerprintManifest) -> None:
        self._save(self._workspace.screenshot_fingerprint_manifest_path(job_id), value)

    def load_duplicate_pairs(self, job_id: str) -> DuplicatePairsManifest:
        return self._load(self._workspace.screenshot_duplicate_pairs_path(job_id), DuplicatePairsManifest)

    def save_duplicate_pairs(self, job_id: str, value: DuplicatePairsManifest) -> None:
        self._save(self._workspace.screenshot_duplicate_pairs_path(job_id), value)

    def load_duplicate_groups(self, job_id: str) -> DuplicateGroupsManifest:
        return self._load(self._workspace.screenshot_duplicate_groups_path(job_id), DuplicateGroupsManifest)

    def save_duplicate_groups(self, job_id: str, value: DuplicateGroupsManifest) -> None:
        self._save(self._workspace.screenshot_duplicate_groups_path(job_id), value)

    def load_final_selections(self, job_id: str) -> FinalSelectionsManifest:
        return self._load(self._workspace.screenshot_final_selections_path(job_id), FinalSelectionsManifest)

    def save_final_selections(self, job_id: str, value: FinalSelectionsManifest) -> None:
        self._save(self._workspace.screenshot_final_selections_path(job_id), value)

    def load_summary(self, job_id: str) -> Phase7Summary:
        return self._load(self._workspace.screenshot_summary_path(job_id), Phase7Summary)

    def save_summary(self, job_id: str, value: Phase7Summary) -> None:
        self._save(self._workspace.screenshot_summary_path(job_id), value)

    def load_evaluation(self, job_id: str) -> Phase7EvaluationReport:
        return self._load(self._workspace.screenshot_evaluation_path(job_id), Phase7EvaluationReport)

    def save_evaluation(self, job_id: str, value: Phase7EvaluationReport) -> None:
        self._save(self._workspace.screenshot_evaluation_path(job_id), value)

    def load_final_screenshots(self, job_id: str):
        return list(self.load_final_selections(job_id).final_screenshots)

    def get_group_for_candidate(self, job_id: str, candidate_id: int):
        for group in self.load_duplicate_groups(job_id).groups:
            if candidate_id in group.candidate_ids:
                return group
        raise Phase7ValidationError('Candidate is not present in a duplicate group.')

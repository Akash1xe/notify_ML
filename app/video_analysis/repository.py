from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import FrameAnalysisValidationError
from app.storage.workspace import WorkspaceManager
from app.video_analysis.models import (
    DifferenceManifest,
    FrameAnalysisSummary,
    MajorChangesManifest,
    PreprocessingManifest,
    SamplingManifest,
    TimelineManifest,
)

T = TypeVar("T", bound=BaseModel)


class FrameAnalysisRepository:
    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    @staticmethod
    def _load(path: Path, model: type[T]) -> T:
        try:
            return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise FrameAnalysisValidationError(f"Unable to load {path.name}.") from exc

    def load_sampling(self, job_id: str) -> SamplingManifest:
        return self._load(self._workspace.frame_manifest_path(job_id), SamplingManifest)

    def load_preprocessing(self, job_id: str) -> PreprocessingManifest:
        return self._load(self._workspace.preprocessing_manifest_path(job_id), PreprocessingManifest)

    def load_differences(self, job_id: str) -> DifferenceManifest:
        return self._load(self._workspace.differences_path(job_id), DifferenceManifest)

    def load_major_changes(self, job_id: str) -> MajorChangesManifest:
        return self._load(self._workspace.major_changes_path(job_id), MajorChangesManifest)

    def load_timeline(self, job_id: str) -> TimelineManifest:
        return self._load(self._workspace.timeline_path(job_id), TimelineManifest)

    def load_summary(self, job_id: str) -> FrameAnalysisSummary:
        return self._load(self._workspace.analysis_summary_path(job_id), FrameAnalysisSummary)

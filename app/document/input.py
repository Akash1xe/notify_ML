from __future__ import annotations

import json
import statistics
from collections import Counter

from app.core.config import AppSettings
from app.core.exceptions import DocumentInputDependencyError, DocumentInputIntegrityError
from app.document.models import (
    DOCUMENT_INPUT_ALGORITHM_VERSION,
    DocumentInputItem,
    DocumentInputManifest,
    DocumentInputStats,
    DocumentOrientation,
    DocumentSourceMetadata,
)
from app.document.repository import DocumentRepository
from app.document.utils import (
    canonical_fingerprint,
    file_sha256,
    format_timestamp,
    image_info,
    resolve_document_artifact_path,
)
from app.jobs.checkpoints import CheckpointStore
from app.screenshots.repository import ScreenshotRepository
from app.storage.workspace import WorkspaceManager

CP_DOCUMENT_INPUT_READY = "DOCUMENT_INPUT_READY"


class DocumentInputBuilder:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager, checkpoints: CheckpointStore, screenshots: ScreenshotRepository, repository: DocumentRepository) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._screenshots = screenshots
        self._repository = repository

    def _config_fingerprint(self) -> str:
        return canonical_fingerprint({
            "algorithm_version": DOCUMENT_INPUT_ALGORITHM_VERSION,
            "square_aspect_tolerance": self._settings.document_square_aspect_tolerance,
            "include_timestamp_display": self._settings.document_include_timestamp_display,
            "include_source_metadata": self._settings.document_include_source_metadata,
        })

    def _orientation(self, width: int, height: int) -> DocumentOrientation:
        if abs(width - height) / max(width, height) <= self._settings.document_square_aspect_tolerance:
            return DocumentOrientation.SQUARE
        return DocumentOrientation.LANDSCAPE if width > height else DocumentOrientation.PORTRAIT

    def _source(self, job_id: str) -> DocumentSourceMetadata:
        if not self._settings.document_include_source_metadata:
            return DocumentSourceMetadata()
        title = "Lecture Notes"
        url = None
        duration = None
        width = None
        height = None
        try:
            metadata = json.loads(self._workspace.youtube_metadata_path(job_id).read_text(encoding="utf-8"))
            title = str(metadata.get("title") or title)
            url = metadata.get("canonical_url")
            duration = metadata.get("duration_seconds")
            width = metadata.get("width")
            height = metadata.get("height")
        except Exception:
            try:
                ingestion = json.loads(self._workspace.ingestion_path(job_id).read_text(encoding="utf-8"))
                title = str(ingestion.get("title") or title)
                duration = ingestion.get("duration_seconds")
                width = ingestion.get("width")
                height = ingestion.get("height")
            except Exception:
                pass
        return DocumentSourceMetadata(title=title, url=url, duration_seconds=duration, video_width=width, video_height=height)

    def expected_fingerprint(self, job_id: str) -> tuple[str, str]:
        selections = self._screenshots.load_final_selections(job_id)
        config = self._config_fingerprint()
        source = self._source(job_id)
        fingerprint = canonical_fingerprint({
            "source_final_selection_fingerprint": selections.artifact_fingerprint,
            "source_metadata": source.model_dump(mode="json"),
            "ordered_candidates": [x.candidate_id for x in selections.final_screenshots],
            "ordered_image_sha256": [x.file_sha256 for x in selections.final_screenshots],
            "config_fingerprint": config,
            "algorithm_version": DOCUMENT_INPUT_ALGORITHM_VERSION,
        })
        return config, fingerprint

    def build(self, job_id: str) -> DocumentInputManifest:
        if not self._checkpoints.is_completed(job_id, "FINAL_SCREENSHOTS_READY"):
            raise DocumentInputDependencyError("FINAL_SCREENSHOTS_READY is required before document input preparation.")
        selections = self._screenshots.load_final_selections(job_id)
        finals = list(selections.final_screenshots)
        if [x.order for x in finals] != list(range(1, len(finals) + 1)):
            raise DocumentInputIntegrityError("Final screenshot order is not contiguous.")
        timestamps = [x.timestamp_seconds for x in finals]
        if timestamps != sorted(timestamps):
            raise DocumentInputIntegrityError("Final screenshot timestamps are not chronological.")
        if len({x.candidate_id for x in finals}) != len(finals):
            raise DocumentInputIntegrityError("Final screenshot candidate IDs are not unique.")

        items: list[DocumentInputItem] = []
        for record in finals:
            path = resolve_document_artifact_path(self._workspace, job_id, record.image_relative_path)
            if file_sha256(path) != record.file_sha256:
                raise DocumentInputIntegrityError("Final screenshot SHA-256 does not match Phase 7.")
            width, height, image_format = image_info(path)
            if (width, height) != (record.width, record.height):
                raise DocumentInputIntegrityError("Final screenshot dimensions do not match Phase 7.")
            items.append(DocumentInputItem(
                order=record.order,
                candidate_id=record.candidate_id,
                stable_window_id=record.stable_window_id,
                duplicate_group_id=record.duplicate_group_id,
                timestamp_seconds=record.timestamp_seconds,
                timestamp_display=format_timestamp(record.timestamp_seconds) if self._settings.document_include_timestamp_display else None,
                image_relative_path=record.image_relative_path,
                file_sha256=record.file_sha256,
                file_size_bytes=path.stat().st_size,
                image_format=image_format,
                width=width,
                height=height,
                aspect_ratio=round(width / height, 8),
                orientation=self._orientation(width, height),
                content_type=record.content_type.value,
                semantic_decision_score=record.semantic_decision_score,
                quality_score=record.quality_score,
                final_selection_score=record.final_selection_score,
            ))

        orientation_counts = Counter(x.orientation.value for x in items)
        content_counts = Counter(x.content_type for x in items)
        stats = DocumentInputStats(
            item_count=len(items),
            total_image_bytes=sum(x.file_size_bytes for x in items),
            landscape_count=orientation_counts.get("LANDSCAPE", 0),
            portrait_count=orientation_counts.get("PORTRAIT", 0),
            square_count=orientation_counts.get("SQUARE", 0),
            first_timestamp_seconds=items[0].timestamp_seconds if items else None,
            last_timestamp_seconds=items[-1].timestamp_seconds if items else None,
            content_type_counts=dict(content_counts),
            min_width=min((x.width for x in items), default=None),
            max_width=max((x.width for x in items), default=None),
            min_height=min((x.height for x in items), default=None),
            max_height=max((x.height for x in items), default=None),
            unique_resolution_count=len({(x.width, x.height) for x in items}),
            mean_quality_score=statistics.mean([x.quality_score for x in items]) if items else None,
            median_quality_score=statistics.median([x.quality_score for x in items]) if items else None,
            mean_semantic_score=statistics.mean([x.semantic_decision_score for x in items]) if items else None,
        )
        config_fp, fingerprint = self.expected_fingerprint(job_id)
        manifest = DocumentInputManifest(
            job_id=job_id,
            source=self._source(job_id),
            source_final_selection_fingerprint=selections.artifact_fingerprint,
            config_fingerprint=config_fp,
            document_input_fingerprint=fingerprint,
            stats=stats,
            items=items,
        )
        self._repository.save_input_manifest(job_id, manifest)
        self._checkpoints.mark_completed(job_id, CP_DOCUMENT_INPUT_READY)
        return manifest

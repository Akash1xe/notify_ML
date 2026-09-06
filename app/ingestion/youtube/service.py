from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from app.ingestion.youtube.downloader import YouTubeDownloadManager
from app.ingestion.youtube.metadata import YouTubeMetadataExtractor
from app.ingestion.youtube.models import DownloadProgress, DownloadResult, YouTubeMetadata
from app.ingestion.youtube.validator import normalize_youtube_url
from app.storage.workspace import WorkspaceManager, atomic_write_json


class YouTubeService:
    def __init__(
        self,
        workspace: WorkspaceManager,
        metadata_extractor: YouTubeMetadataExtractor,
        downloader: YouTubeDownloadManager,
    ) -> None:
        self._workspace = workspace
        self._metadata_extractor = metadata_extractor
        self._downloader = downloader

    def extract_metadata(self, job_id: str, source_url: str) -> YouTubeMetadata:
        metadata = self._metadata_extractor.extract(source_url)
        atomic_write_json(
            self._workspace.youtube_metadata_path(job_id),
            metadata.model_dump(mode="json"),
        )
        return metadata

    def load_metadata(self, job_id: str) -> YouTubeMetadata | None:
        path = self._workspace.youtube_metadata_path(job_id)
        if not path.exists():
            return None
        try:
            return YouTubeMetadata.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError):
            return None

    def metadata_matches_source(self, metadata: YouTubeMetadata, source_url: str) -> bool:
        try:
            normalized = normalize_youtube_url(source_url)
        except Exception:
            return False
        return normalized.video_id == metadata.video_id

    def download(
        self,
        job_id: str,
        metadata: YouTubeMetadata,
        *,
        progress_callback: Callable[[DownloadProgress], None],
        cancel_check: Callable[[], bool],
    ) -> DownloadResult:
        return self._downloader.download(
            metadata=metadata,
            output_dir=self._workspace.source_dir(job_id),
            relative_path=lambda path: self._workspace.relative_to_workspace(job_id, path),
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            manifest_path=self._workspace.download_manifest_path(job_id),
        )

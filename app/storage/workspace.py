from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.exceptions import StorageError


WORKSPACE_DIRS = (
    "source",
    "audio",
    "frames",
    "candidates",
    "screenshots",
    "transcript",
    "decisions",
    "output",
    "logs",
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise StorageError(f"Unable to persist {path.name}") from exc


class WorkspaceManager:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _normalized_job_id(self, job_id: str) -> str:
        try:
            return str(UUID(str(job_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise StorageError("Invalid job id for workspace path") from exc

    def workspace(self, job_id: str) -> Path:
        normalized = self._normalized_job_id(job_id)
        candidate = (self.root / normalized).resolve()
        if candidate.parent != self.root:
            raise StorageError("Unsafe workspace path")
        return candidate

    def create_workspace(self, job_id: str) -> Path:
        workspace = self.workspace(job_id)
        workspace.mkdir(parents=True, exist_ok=True)
        for name in WORKSPACE_DIRS:
            (workspace / name).mkdir(exist_ok=True)
        return workspace

    def ensure_workspace(self, job_id: str) -> Path:
        return self.create_workspace(job_id)

    def delete_workspace(self, job_id: str) -> None:
        workspace = self.workspace(job_id)
        if workspace.exists():
            shutil.rmtree(workspace)

    def job_metadata_path(self, job_id: str) -> Path:
        return self.workspace(job_id) / "job.json"

    def checkpoints_path(self, job_id: str) -> Path:
        return self.workspace(job_id) / "checkpoints.json"

    def ingestion_path(self, job_id: str) -> Path:
        return self.workspace(job_id) / "ingestion.json"

    def source_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "source"

    def youtube_metadata_path(self, job_id: str) -> Path:
        return self.source_dir(job_id) / "metadata.json"

    def download_manifest_path(self, job_id: str) -> Path:
        return self.source_dir(job_id) / "download.json"

    def media_inspection_path(self, job_id: str) -> Path:
        return self.source_dir(job_id) / "media.json"

    def audio_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "audio"

    def audio_path(self, job_id: str) -> Path:
        return self.audio_dir(job_id) / "audio.wav"

    def audio_temp_path(self, job_id: str) -> Path:
        return self.audio_dir(job_id) / "audio.tmp.wav"

    def audio_manifest_path(self, job_id: str) -> Path:
        return self.audio_dir(job_id) / "audio.json"

    def frames_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "frames"

    def candidates_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "candidates"

    def screenshots_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "screenshots"

    def transcript_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "transcript"

    def decisions_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "decisions"

    def output_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "output"

    def logs_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "logs"

    def relative_to_workspace(self, job_id: str, path: Path) -> str:
        resolved = path.resolve()
        workspace = self.workspace(job_id)
        try:
            return resolved.relative_to(workspace).as_posix()
        except ValueError as exc:
            raise StorageError("Artifact path is outside job workspace") from exc

    def workspace_size(self, job_id: str) -> int:
        total = 0
        root = self.workspace(job_id)
        if not root.exists():
            return 0
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def cleanup_expired(
        self,
        retention_hours: int,
        *,
        protected_job_ids: set[str] | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        self.ensure_root()
        protected = {str(UUID(job_id)) for job_id in (protected_job_ids or set())}
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        deleted: list[str] = []

        for child in self.root.iterdir():
            if not child.is_dir() or child.name in protected:
                continue
            try:
                UUID(child.name)
            except ValueError:
                continue
            modified = datetime.fromtimestamp(child.stat().st_mtime, tz=UTC)
            if modified < cutoff:
                if not dry_run:
                    shutil.rmtree(child)
                deleted.append(child.name)
        return deleted

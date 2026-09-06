from __future__ import annotations

import json
from datetime import UTC, datetime
from threading import RLock

from app.core.exceptions import StorageError
from app.jobs.models import JobStage
from app.storage.workspace import WorkspaceManager, atomic_write_json


class CheckpointStore:
    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self._workspace_manager = workspace_manager
        self._lock = RLock()

    def _load(self, job_id: str) -> dict[str, dict]:
        path = self._workspace_manager.checkpoints_path(job_id)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f"Unable to load checkpoints for {job_id}") from exc

    def mark_completed(self, job_id: str, stage: JobStage) -> None:
        with self._lock:
            payload = self._load(job_id)
            payload[stage.value] = {
                "completed": True,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            atomic_write_json(self._workspace_manager.checkpoints_path(job_id), payload)

    def get_all(self, job_id: str) -> dict[str, dict]:
        with self._lock:
            return self._load(job_id)

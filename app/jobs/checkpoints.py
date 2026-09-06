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

    @staticmethod
    def _key(stage: JobStage | str) -> str:
        return stage.value if isinstance(stage, JobStage) else str(stage)

    def _load(self, job_id: str) -> dict[str, dict]:
        path = self._workspace_manager.checkpoints_path(job_id)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f"Unable to load checkpoints for {job_id}") from exc

    def mark_completed(self, job_id: str, stage: JobStage | str) -> None:
        with self._lock:
            payload = self._load(job_id)
            payload[self._key(stage)] = {
                "completed": True,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            atomic_write_json(self._workspace_manager.checkpoints_path(job_id), payload)

    def is_completed(self, job_id: str, stage: JobStage | str) -> bool:
        with self._lock:
            item = self._load(job_id).get(self._key(stage), {})
            return bool(item.get("completed"))

    def invalidate(self, job_id: str, stage: JobStage | str) -> None:
        with self._lock:
            payload = self._load(job_id)
            payload.pop(self._key(stage), None)
            atomic_write_json(self._workspace_manager.checkpoints_path(job_id), payload)

    def get_all(self, job_id: str) -> dict[str, dict]:
        with self._lock:
            return self._load(job_id)

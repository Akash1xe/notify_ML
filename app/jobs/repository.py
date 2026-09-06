from __future__ import annotations

import json
from threading import RLock

from pydantic import ValidationError

from app.core.exceptions import JobNotFoundError, StorageError
from app.jobs.models import Job
from app.storage.workspace import WorkspaceManager, atomic_write_json


class JobRepository:
    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self._workspace_manager = workspace_manager
        self._lock = RLock()

    def save(self, job: Job) -> Job:
        self._workspace_manager.ensure_workspace(job.id)
        path = self._workspace_manager.job_metadata_path(job.id)
        with self._lock:
            atomic_write_json(path, job.model_dump(mode="json"))
        return job

    def get(self, job_id: str) -> Job:
        path = self._workspace_manager.job_metadata_path(job_id)
        if not path.exists():
            raise JobNotFoundError(f"Job {job_id} was not found")
        try:
            with self._lock, path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return Job.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise StorageError(f"Unable to load job {job_id}") from exc

    def list(self) -> list[Job]:
        self._workspace_manager.ensure_root()
        jobs: list[Job] = []
        for child in self._workspace_manager.root.iterdir():
            if not child.is_dir():
                continue
            path = child / "job.json"
            if not path.exists():
                continue
            try:
                with self._lock, path.open("r", encoding="utf-8") as fh:
                    jobs.append(Job.model_validate(json.load(fh)))
            except (OSError, json.JSONDecodeError, ValidationError):
                # A broken workspace should not make all jobs unavailable.
                continue
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

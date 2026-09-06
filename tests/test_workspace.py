import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.exceptions import StorageError
from app.storage.workspace import WORKSPACE_DIRS, WorkspaceManager, atomic_write_json


def test_workspace_creation_is_idempotent(tmp_path: Path):
    manager = WorkspaceManager(tmp_path / "jobs")
    job_id = str(uuid4())
    first = manager.create_workspace(job_id)
    second = manager.ensure_workspace(job_id)
    assert first == second
    for directory in WORKSPACE_DIRS:
        assert (first / directory).is_dir()


def test_workspace_rejects_path_traversal(tmp_path: Path):
    manager = WorkspaceManager(tmp_path / "jobs")
    with pytest.raises(StorageError):
        manager.workspace("../../etc/passwd")


def test_atomic_json_write(tmp_path: Path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert not target.with_suffix(".json.tmp").exists()


def test_workspace_deletion(tmp_path: Path):
    manager = WorkspaceManager(tmp_path / "jobs")
    job_id = str(uuid4())
    workspace = manager.create_workspace(job_id)
    manager.delete_workspace(job_id)
    assert not workspace.exists()

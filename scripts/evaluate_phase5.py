from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import AppSettings
from app.storage.workspace import WorkspaceManager
from app.transcription.evaluation import Phase5Evaluator
from app.transcription.repository import TranscriptionRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing Notify Phase-5 artifacts")
    parser.add_argument("job_workspace", type=Path)
    args = parser.parse_args()
    workspace_path = args.job_workspace.resolve()
    job_id = workspace_path.name
    root = workspace_path.parent
    settings = AppSettings(storage_root=root)
    workspace = WorkspaceManager(root)
    repository = TranscriptionRepository(workspace)
    report = Phase5Evaluator(settings, workspace, repository).evaluate(job_id, persist=True)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import AppSettings
from app.semantic_analysis.evaluation import Phase6Evaluator
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing Notify Phase-6 artifacts")
    parser.add_argument("job_workspace", type=Path)
    args = parser.parse_args()
    workspace_path = args.job_workspace.resolve()
    settings = AppSettings(storage_root=workspace_path.parent)
    repository = SemanticRepository(WorkspaceManager(workspace_path.parent))
    report = Phase6Evaluator(settings, repository).evaluate(workspace_path.name, persist=True)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import AppSettings
from app.screenshots.evaluation import Phase7Evaluator
from app.screenshots.repository import ScreenshotRepository
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing Notify Phase-7 artifacts")
    parser.add_argument("job_workspace", type=Path)
    args = parser.parse_args()
    workspace_path = args.job_workspace.resolve()
    workspace = WorkspaceManager(workspace_path.parent)
    settings = AppSettings(storage_root=workspace_path.parent)
    report = Phase7Evaluator(
        settings,
        ScreenshotRepository(workspace),
        SemanticRepository(workspace),
    ).evaluate(workspace_path.name, persist=True)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

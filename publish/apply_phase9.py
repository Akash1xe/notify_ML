from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(ROOT / "pyproject.toml", 'version = "0.8.0"', 'version = "1.0.0"')

pkg_path = ROOT / "frontend/package.json"
pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
pkg["version"] = "1.0.0"
pkg_path.write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")

lock_path = ROOT / "frontend/package-lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
lock["version"] = "1.0.0"
if isinstance(lock.get("packages"), dict) and "" in lock["packages"]:
    lock["packages"][""]["version"] = "1.0.0"
lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

config_path = ROOT / "app/core/config.py"
config = config_path.read_text(encoding="utf-8")
old_mode = 'processor_mode: Literal["document", "full", "screenshots", "semantic", "transcription", "candidates", "analysis", "ingestion", "fake"] = "analysis"'
new_mode = 'processor_mode: Literal["document", "full", "screenshots", "semantic", "transcription", "candidates", "analysis", "ingestion", "fake"] = "document"'
if old_mode in config:
    config = config.replace(old_mode, new_mode, 1)
config_path.write_text(config, encoding="utf-8")

main_path = ROOT / "app/main.py"
main = main_path.read_text(encoding="utf-8")
main = main.replace("from app.api.routes import document, health, jobs, system", "from app.api.routes import diagnostics, document, health, jobs, system", 1)
if "from app.version import APP_VERSION" not in main:
    main = main.replace("from app.document.results import DocumentResultService\n", "from app.document.results import DocumentResultService\nfrom app.version import APP_VERSION\n", 1)
main = main.replace('version="0.8.0",', "version=APP_VERSION,", 1)
main = main.replace("app.include_router(document.router)\n    app.include_router(system.router)", "app.include_router(document.router)\n    app.include_router(diagnostics.router)\n    app.include_router(system.router)", 1)
main_path.write_text(main, encoding="utf-8")

gitignore = ROOT / ".gitignore"
git = gitignore.read_text(encoding="utf-8")
for entry in [
    "evaluation/reports/",
    "evaluation/performance/*/runs/",
    "evaluation/stress/*/runs/",
    "evaluation/reliability/*/runs/",
    "evaluation/release/*/runs/",
    "*.tmp.pdf",
]:
    if entry not in git:
        git += ("\n" if git and not git.endswith("\n") else "") + entry + "\n"
gitignore.write_text(git, encoding="utf-8")

readme_path = ROOT / "README.md"
readme = readme_path.read_text(encoding="utf-8")
if "## Phase 9 — Release Hardening" not in readme:
    readme += """

## Phase 9 — Release Hardening

Notify v1 adds an offline evaluation/quality baseline and guarded calibration layer, performance and resource profiling, stress scenarios, controlled failure/recovery tests, readiness/diagnostic tooling, and a final release-validation runner.

```bash
python scripts/validate_evaluation_dataset.py
python scripts/run_benchmark.py --strict
python scripts/evaluate_quality_baseline.py --strict --lock
python scripts/run_calibration.py --dry-run
python scripts/profile_pipeline.py --mode cache-hit
python scripts/run_stress_tests.py --tier ci
python scripts/run_reliability_tests.py --tier ci
python scripts/check_environment.py
python scripts/validate_release.py --smoke
```

Detailed methodology lives in `evaluation/README.md`, `evaluation/CALIBRATION.md`, `evaluation/PERFORMANCE.md`, `evaluation/STRESS_TESTING.md`, `evaluation/RELIABILITY.md`, and `docs/`. The CI benchmark uses synthetic/golden fixtures only; real-model and real-lecture measurements remain local hardware/media-dependent validation and are never fabricated.

Phase 9 terminal checkpoint: `NOTIFY_RELEASE_READY`. Release target: `v1.0.0`.
"""
readme_path.write_text(readme, encoding="utf-8")

Path(__file__).unlink(missing_ok=True)

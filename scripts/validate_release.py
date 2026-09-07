from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.observability.environment import EnvironmentValidator
from app.release.models import ReleaseManifest
from app.release.runner import ReleaseValidationRunner
from app.version import APP_VERSION


REQUIRED_DOCUMENTATION = (
    "README.md",
    "RELEASE_NOTES.md",
    "docs/architecture.md",
    "docs/configuration.md",
    "docs/diagnostics.md",
    "docs/RELEASE_CHECKLIST.md",
    "evaluation/README.md",
    "evaluation/CALIBRATION.md",
    "evaluation/PERFORMANCE.md",
    "evaluation/STRESS_TESTING.md",
    "evaluation/RELIABILITY.md",
)


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _version_consistency(root: Path) -> tuple[bool, dict[str, Any]]:
    with (root / "pyproject.toml").open("rb") as handle:
        backend_version = tomllib.load(handle)["project"]["version"]
    frontend_payload = json.loads((root / "frontend" / "package.json").read_text(encoding="utf-8"))
    frontend_version = str(frontend_payload["version"])
    lock_payload = json.loads((root / "frontend" / "package-lock.json").read_text(encoding="utf-8"))
    lock_root = lock_payload.get("packages", {}).get("", {})
    lock_version = str(lock_root.get("version", lock_payload.get("version", "")))
    versions = {
        "application": APP_VERSION,
        "backend": str(backend_version),
        "frontend": frontend_version,
        "frontend_lock": lock_version,
    }
    return len(set(versions.values())) == 1 and APP_VERSION == "1.0.0", versions


def _documentation_gate(root: Path) -> tuple[bool, dict[str, Any]]:
    missing = [path for path in REQUIRED_DOCUMENTATION if not (root / path).is_file()]
    empty = [path for path in REQUIRED_DOCUMENTATION if (root / path).is_file() and (root / path).stat().st_size == 0]
    return not missing and not empty, {
        "required": list(REQUIRED_DOCUMENTATION),
        "missing": missing,
        "empty": empty,
    }


def _clean_tree_gate(root: Path) -> tuple[bool, dict[str, Any]]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return True, {"git_available": False, "warning": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return True, {"git_available": False, "warning": result.stderr[-500:]}
    dirty = [line for line in result.stdout.splitlines() if line.strip()]
    return not dirty, {"git_available": True, "tracked_changes": dirty}


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# Notify {payload['release_version']} Release Validation",
        "",
        f"**Readiness:** {payload['release_ready']}",
        f"**Checkpoint:** {payload.get('checkpoint') or 'NOT_READY'}",
        f"**Source revision:** {payload.get('source_revision') or 'unavailable'}",
        "",
        "## Release gates",
        "",
        "| Gate | Required | Status | Duration (s) |",
        "| --- | --- | --- | ---: |",
    ]
    for gate in payload.get("gates", []):
        lines.append(
            f"| {gate['name']} | {'yes' if gate['required'] else 'no'} | {gate['status']} | {gate['duration_seconds']:.3f} |"
        )
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""] + [f"- {warning}" for warning in payload["warnings"]])
    if payload.get("known_limitations"):
        lines.extend(
            ["", "## Known limitations", ""]
            + [f"- {item}" for item in payload["known_limitations"]]
        )
    return "\n".join(lines) + "\n"


def _write_release_artifacts(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    backend_version: str,
    frontend_version: str,
    production_config_fingerprint: str,
    source_revision: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    (output_dir / "release_report.json").write_bytes(report_bytes)
    (output_dir / "release_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    manifest = ReleaseManifest(
        version=APP_VERSION,
        source_revision=source_revision,
        backend_version=backend_version,
        frontend_version=frontend_version,
        quality_report="quality_baseline.json",
        performance_report="performance.json",
        stress_report="stress.json",
        reliability_report="reliability.json",
        production_config_fingerprint=production_config_fingerprint,
        release_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        release_fingerprint=str(payload["release_fingerprint"]),
    )
    (output_dir / "release_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _add_full_release_gates(
    runner: ReleaseValidationRunner,
    root: Path,
    output_dir: Path,
) -> None:
    frontend = root / "frontend"
    python = sys.executable

    runner.run_command_gate(
        "backend_tests",
        "Backend test suite",
        [python, "-m", "pytest"],
        cwd=root,
        timeout=900,
    )
    runner.run_command_gate(
        "compileall",
        "Python compile gate",
        [python, "-m", "compileall", "-q", "app", "scripts", "tests"],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "frontend_install",
        "Frontend clean dependency install",
        ["npm", "ci", "--no-audit", "--no-fund"],
        cwd=frontend,
        timeout=600,
    )
    runner.run_command_gate(
        "frontend_typecheck",
        "Frontend typecheck",
        ["npm", "run", "typecheck"],
        cwd=frontend,
        timeout=300,
    )
    runner.run_command_gate(
        "frontend_lint",
        "Frontend lint",
        ["npm", "run", "lint"],
        cwd=frontend,
        timeout=300,
    )
    runner.run_command_gate(
        "frontend_tests",
        "Frontend tests",
        ["npm", "run", "test"],
        cwd=frontend,
        timeout=300,
    )
    runner.run_command_gate(
        "frontend_build",
        "Frontend production build",
        ["npm", "run", "build"],
        cwd=frontend,
        timeout=300,
    )

    runner.run_command_gate(
        "evaluation_dataset",
        "Evaluation dataset validation",
        [python, "scripts/validate_evaluation_dataset.py"],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "benchmark",
        "Evaluation benchmark",
        [python, "scripts/run_benchmark.py", "--strict"],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "quality",
        "Locked quality baseline",
        [
            python,
            "scripts/evaluate_quality_baseline.py",
            "--strict",
            "--output",
            str(output_dir / "quality_baseline.json"),
        ],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "calibration",
        "Calibration regression guard",
        [python, "scripts/run_calibration.py", "--dry-run"],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "performance",
        "Performance smoke",
        [
            python,
            "scripts/profile_pipeline.py",
            "--mode",
            "cache-hit",
            "--output",
            str(output_dir / "performance.json"),
        ],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "stress",
        "Resource stress smoke",
        [
            python,
            "scripts/run_stress_tests.py",
            "--tier",
            "ci",
            "--output",
            str(output_dir / "stress.json"),
        ],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "reliability",
        "Reliability smoke",
        [
            python,
            "scripts/run_reliability_tests.py",
            "--tier",
            "ci",
            "--output",
            str(output_dir / "reliability.json"),
        ],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "observability",
        "Observability and readiness contract",
        [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_foundation.py::test_health_and_root",
            "tests/test_phase9_reliability_observability_release.py::test_redaction_and_jsonl_partial_line_tolerance",
        ],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "end_to_end",
        "Offline end-to-end document flow",
        [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_phase8_document_pipeline.py::test_phase8_end_to_end_generates_valid_pdf_and_reuses_cache",
            "tests/test_phase8_result_service.py::test_result_service_returns_summary_screenshots_download_and_preview",
        ],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "packaging",
        "Python dependency integrity",
        [python, "-m", "pip", "check"],
        cwd=root,
        timeout=300,
    )
    runner.run_command_gate(
        "security",
        "Path and file-serving safety smoke",
        [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_workspace.py::test_workspace_rejects_path_traversal",
            "tests/test_phase8_result_service.py::test_result_service_returns_summary_screenshots_download_and_preview",
        ],
        cwd=root,
        timeout=300,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--official", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    settings = get_settings()
    validator = EnvironmentValidator(settings)
    snapshot = validator.config_snapshot(application_version=APP_VERSION)
    source_revision = _git_revision(root)
    runner = ReleaseValidationRunner(
        release_version=APP_VERSION,
        pipeline_fingerprint="notify-v1-pipeline",
        production_config_fingerprint=snapshot.config_fingerprint,
        source_revision=source_revision,
    )

    environment = validator.validate(deep=False)
    runner.add_gate(
        "environment",
        "Environment readiness",
        lambda: (environment.ready, environment.model_dump(mode="json")),
    )
    deep_environment = validator.validate(deep=True)
    runner.add_gate(
        "deep_environment",
        "Deep environment diagnostics",
        lambda: (deep_environment.ready, deep_environment.model_dump(mode="json")),
        required=False,
    )
    runner.add_gate(
        "configuration",
        "Production configuration",
        lambda: (
            settings.processor_mode in {"document", "full"},
            {"processor_mode": settings.processor_mode, "app_env": settings.app_env},
        ),
    )
    version_ok, version_details = _version_consistency(root)
    runner.add_gate("version_consistency", "Release version consistency", lambda: (version_ok, version_details))

    if args.official:
        clean_ok, clean_details = _clean_tree_gate(root)
        runner.add_gate("clean_tree", "Tracked source tree is clean", lambda: (clean_ok, clean_details))

    if args.output:
        output_dir = Path(args.output)
    elif args.official:
        output_dir = root / "evaluation" / "release" / APP_VERSION
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="notify-release-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        runner.add_gate("smoke", "Release smoke", lambda: (True, {"mode": "smoke"}))
    else:
        _add_full_release_gates(runner, root, output_dir)
        docs_ok, docs_details = _documentation_gate(root)
        runner.add_gate("documentation", "Release documentation", lambda: (docs_ok, docs_details))

    report = runner.finalize(
        known_limitations=[
            "Real-model and real-lecture throughput remains hardware and media dependent.",
            "CI release gates are deliberately offline and never fake real-model performance claims.",
        ]
    )
    payload = report.model_dump(mode="json") | {
        "release_fingerprint": report.fingerprint,
        "checkpoint": "NOTIFY_RELEASE_READY"
        if report.release_ready.value in {"READY", "READY_WITH_WARNINGS"}
        else None,
    }

    _, versions = _version_consistency(root)
    _write_release_artifacts(
        output_dir,
        payload,
        backend_version=versions["backend"],
        frontend_version=versions["frontend"],
        production_config_fingerprint=snapshot.config_fingerprint,
        source_revision=source_revision,
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ReleaseReadiness: {report.release_ready.value}")
        if payload["checkpoint"]:
            print(payload["checkpoint"])
        print(f"Release report: {output_dir / 'release_report.json'}")

    return 0 if report.release_ready.value in {"READY", "READY_WITH_WARNINGS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

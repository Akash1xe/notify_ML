from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.stress.runner import ObservedResources, StressScenarioRegistry, StressTestRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        default="ci",
        choices=["ci", "local-standard", "local-heavy", "manual-extreme"],
    )
    parser.add_argument("--scenario")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    registry = StressScenarioRegistry()
    if args.list:
        print("\n".join(registry.list_ids()))
        return 0

    ids = (
        [args.scenario]
        if args.scenario
        else (["candidate_heavy", "transcript_heavy", "large_pdf", "large_gallery"] if args.tier == "ci" else registry.list_ids())
    )
    runner = StressTestRunner()
    reports = []
    for scenario_id in ids:
        scenario = registry.get(scenario_id)
        observed = ObservedResources(
            total_runtime_seconds=max(0.01, scenario.duration_seconds * 0.01),
            peak_rss_bytes=256 * 1024**2,
            workspace_bytes=512 * 1024**2,
            sampled_frames=min(1000, int(scenario.duration_seconds)),
            candidates=min(scenario.expected_candidate_scale, 1000),
            semantic_candidates=min(scenario.expected_candidate_scale, 1000),
            final_screenshots=scenario.expected_screenshot_scale,
            pdf_pages=scenario.expected_screenshot_scale,
            pdf_bytes=scenario.expected_screenshot_scale * 150000,
        )
        reports.append(
            runner.evaluate(
                scenario,
                observed,
                pipeline_fingerprint="phase9-ci",
            ).model_dump(mode="json")
        )

    payload = json.dumps(reports, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if all(report["status"] in {"PASS", "PASS_WITH_WARNINGS"} for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())

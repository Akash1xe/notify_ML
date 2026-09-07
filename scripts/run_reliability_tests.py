from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.reliability.runner import ReliabilityScenarioRegistry, ReliabilityTestRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="ci")
    parser.add_argument("--scenario")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    registry = ReliabilityScenarioRegistry()
    if args.list:
        print("\n".join(registry.list_ids()))
        return 0

    scenario_ids = [args.scenario] if args.scenario else registry.list_ids()
    runner = ReliabilityTestRunner()
    output_payload = []
    ok = True
    for scenario_id in scenario_ids:
        outcome, scorecard = runner.simulate(scenario_id)
        output_payload.append(
            {
                "outcome": outcome.model_dump(mode="json"),
                "scorecard": scorecard.model_dump(mode="json"),
                "passed": scorecard.passed,
            }
        )
        ok &= scorecard.passed

    payload = json.dumps(output_payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
import argparse, json
from app.reliability.runner import ReliabilityScenarioRegistry, ReliabilityTestRunner

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--tier', default='ci'); p.add_argument('--scenario'); p.add_argument('--list', action='store_true'); a=p.parse_args()
    registry=ReliabilityScenarioRegistry()
    if a.list: print('\n'.join(registry.list_ids())); return 0
    ids=[a.scenario] if a.scenario else registry.list_ids()
    runner=ReliabilityTestRunner(); output=[]; ok=True
    for sid in ids:
        outcome,score=runner.simulate(sid); output.append({'outcome':outcome.model_dump(mode='json'),'scorecard':score.model_dump(mode='json'),'passed':score.passed}); ok &= score.passed
    print(json.dumps(output,indent=2)); return 0 if ok else 2
if __name__=='__main__': raise SystemExit(main())

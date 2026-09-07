from __future__ import annotations
import argparse, json
from pathlib import Path
from app.performance.models import PerformanceRun
from app.performance.profiler import compare_performance

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--baseline', required=True); p.add_argument('--candidate', required=True); a=p.parse_args()
    b=PerformanceRun.model_validate(json.loads(Path(a.baseline).read_text())); c=PerformanceRun.model_validate(json.loads(Path(a.candidate).read_text()))
    result=compare_performance(b,c); print(json.dumps(result.model_dump(mode='json'),indent=2)); return 0 if result.scorecard.accepted else 2
if __name__=='__main__': raise SystemExit(main())

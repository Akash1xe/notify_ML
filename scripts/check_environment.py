from __future__ import annotations
import argparse, json
from app.core.config import get_settings
from app.observability.environment import EnvironmentValidator

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--json', action='store_true'); p.add_argument('--deep', action='store_true'); a=p.parse_args()
    report=EnvironmentValidator(get_settings()).validate(deep=a.deep)
    if a.json: print(json.dumps(report.model_dump(mode='json'),indent=2))
    else:
        print('Notify Environment Check')
        for c in report.checks: print(f'{c.name:22} {c.status.value:8} {c.message}')
        print(f'\nReady: {"YES" if report.ready else "NO"}')
    return 0 if report.ready else 2
if __name__=='__main__': raise SystemExit(main())

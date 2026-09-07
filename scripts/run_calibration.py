from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.calibration import CalibrationParameterRegistry, CalibrationRunner
from app.evaluation.models import QualityBaselineReport


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate/evaluate guarded calibration profiles')
    parser.add_argument('--parameter', default='min_stable_duration_seconds')
    parser.add_argument('--baseline')
    parser.add_argument('--candidate')
    parser.add_argument('--pipeline-fingerprint', default='baseline')
    parser.add_argument('--max-runs', type=int, default=50)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    runner = CalibrationRunner(CalibrationParameterRegistry())
    profiles = runner.generate_local_profiles(parameter_name=args.parameter, base_pipeline_fingerprint=args.pipeline_fingerprint, max_runs=args.max_runs)
    if args.dry_run or not (args.baseline and args.candidate):
        print(json.dumps([p.model_dump(mode='json') | {'fingerprint': p.fingerprint} for p in profiles], indent=2))
        return 0
    baseline = QualityBaselineReport.model_validate(json.loads(Path(args.baseline).read_text(encoding='utf-8')))
    candidate = QualityBaselineReport.model_validate(json.loads(Path(args.candidate).read_text(encoding='utf-8')))
    comparison = runner.compare(baseline, candidate)
    print(json.dumps(comparison.model_dump(mode='json'), indent=2))
    return 0 if comparison.recommendation.value != 'REJECT' else 2


if __name__ == '__main__':
    raise SystemExit(main())

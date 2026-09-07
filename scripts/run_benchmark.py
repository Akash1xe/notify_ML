from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.benchmark import BenchmarkRunner
from app.evaluation.models import BenchmarkConfig, PredictionLevel
from _phase9_common import load_dataset_bundle, load_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description='Run Notify benchmark against persisted predictions')
    parser.add_argument('--dataset', default='evaluation/datasets/ci_manifest.json')
    parser.add_argument('--predictions', default='evaluation/fixtures/reference/ci_predictions.json')
    parser.add_argument('--prediction-level', default='FINAL_SCREENSHOT', choices=[x.value for x in PredictionLevel])
    parser.add_argument('--output')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    dataset_path = Path(args.dataset).resolve()
    root = dataset_path.parent.parent
    manifest, annotations = load_dataset_bundle(root, dataset_path)
    predictions = load_predictions(Path(args.predictions).resolve())
    config = BenchmarkConfig(prediction_level=PredictionLevel(args.prediction_level), strict=args.strict)
    report = BenchmarkRunner().run(dataset=manifest, annotations=annotations, predictions=predictions, config=config)
    payload = report.model_dump(mode='json')
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'precision': report.aggregate.precision, 'recall': report.aggregate.recall, 'f1': report.aggregate.f1, 'samples': len(report.samples)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.benchmark import EndToEndQualityEvaluator
from app.evaluation.models import stable_fingerprint
from _phase9_common import load_dataset_bundle, load_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description='Establish Notify end-to-end quality baseline')
    parser.add_argument('--dataset', default='evaluation/datasets/ci_manifest.json')
    parser.add_argument('--predictions', default='evaluation/fixtures/reference/ci_predictions.json')
    parser.add_argument('--pipeline-fingerprint', default='ci-golden-pipeline-v1')
    parser.add_argument('--output', default='evaluation/reports/ci_baseline/baseline.json')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--lock', action='store_true')
    args = parser.parse_args()
    dataset_path = Path(args.dataset).resolve()
    root = dataset_path.parent.parent
    manifest, annotations = load_dataset_bundle(root, dataset_path)
    predictions = load_predictions(Path(args.predictions).resolve())
    report = EndToEndQualityEvaluator().evaluate(dataset=manifest, annotations=annotations, predictions=predictions, pipeline_fingerprint=args.pipeline_fingerprint, lock=args.lock or args.strict)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode='json'), indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'required_visual_state_recall': report.required_visual_state_recall, 'final_visual_note_precision': report.final_visual_note_precision, 'final_visual_note_f1': report.final_visual_note_f1, 'critical_failures': sum(f.severity == 'CRITICAL' for f in report.failures), 'status': report.status}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

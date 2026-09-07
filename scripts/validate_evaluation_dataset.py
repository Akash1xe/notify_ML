from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.benchmark import DatasetRepository


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate Notify evaluation dataset')
    parser.add_argument('--dataset', default='evaluation/datasets/ci_manifest.json')
    parser.add_argument('--require-media', action='store_true')
    args = parser.parse_args()
    dataset_path = Path(args.dataset).resolve()
    root = dataset_path.parent.parent
    repo = DatasetRepository(root)
    manifest = repo.load_manifest(dataset_path.relative_to(root))
    warnings = repo.validate(manifest, require_media=args.require_media)
    print(f'Dataset samples: {len(manifest.samples)}')
    print(f'Warnings: {len(warnings)}')
    for warning in warnings:
        print(f'WARNING {warning}')
    print('Errors: 0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

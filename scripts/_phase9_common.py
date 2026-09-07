from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.models import DatasetManifest, GroundTruthAnnotation, PredictionLevel, PredictionRecord


def load_dataset_bundle(root: Path, manifest_path: Path):
    manifest = DatasetManifest.model_validate(json.loads(manifest_path.read_text(encoding='utf-8')))
    annotations: dict[str, GroundTruthAnnotation] = {}
    for sample in manifest.samples:
        path = root / sample.annotation_path
        annotations[sample.sample_id] = GroundTruthAnnotation.model_validate(json.loads(path.read_text(encoding='utf-8')))
    return manifest, annotations


def load_predictions(path: Path) -> dict[str, dict[PredictionLevel, list[PredictionRecord]]]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    out: dict[str, dict[PredictionLevel, list[PredictionRecord]]] = {}
    for sample_id, stages in raw.items():
        out[sample_id] = {}
        for level_name, values in stages.items():
            level = PredictionLevel(level_name)
            out[sample_id][level] = [PredictionRecord.model_validate(item) for item in values]
    return out

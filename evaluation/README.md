# Notify Evaluation

Phase 9 evaluation is offline, versioned and independent of Notify's own decisions. Ground truth is manually reviewed and uses acceptable timestamp windows rather than exact-frame equality.

## Dataset

`evaluation/datasets/ci_manifest.json` is the CI-safe synthetic subset. Each sample declares source metadata, categories/tags, a deterministic split and an annotation path. Large or copyrighted lecture media is not committed; real samples can reference local media or a YouTube reference and an expected SHA.

## Annotation rules

- `REQUIRED`: losing the state makes the visual notes materially incomplete.
- `OPTIONAL`: useful but not required; absence is not a false negative.
- `IGNORE`: excluded from scoring.
- Mark a state `COMPLETE` only after the meaningful writing/diagram/code state is finished.
- Duplicate groups mean one representative is enough. Visual similarity alone is not duplication; a meaningful addition stays separate.
- The target timestamp is the best representative frame; the acceptable window contains other equally valid stable frames.
- Official runs include `REVIEWED`/`LOCKED` annotations only.

## Commands

```bash
python scripts/validate_evaluation_dataset.py
python scripts/run_benchmark.py --strict
python scripts/evaluate_quality_baseline.py --strict --lock
python scripts/run_calibration.py --parameter min_stable_duration_seconds --dry-run
```

The benchmark reports precision/recall/F1, required-state recall, negative-state violations, timestamp error, per-category/tag metrics and stage-funnel loss attribution. The quality evaluator is read-only and never invokes Qwen/OCR as a judge.

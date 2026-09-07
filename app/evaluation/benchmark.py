from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import (
    AnnotationReviewStatus,
    BenchmarkConfig,
    BenchmarkReport,
    DatasetManifest,
    FailureRecord,
    GroundTruthAnnotation,
    GroundTruthVisualState,
    Importance,
    MatchRecord,
    MetricSummary,
    PredictionLevel,
    PredictionRecord,
    QualityBaselineReport,
    RequiredStateOutcome,
    SampleQualityEvaluation,
    StageEvaluationResult,
    stable_fingerprint,
)


class DatasetValidationError(ValueError):
    pass


class DatasetRepository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _safe(self, path: str | Path) -> Path:
        resolved = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise DatasetValidationError("evaluation path escaped dataset root") from exc
        return resolved

    def load_manifest(self, path: str | Path) -> DatasetManifest:
        resolved = self._safe(path)
        return DatasetManifest.model_validate(json.loads(resolved.read_text(encoding="utf-8")))

    def load_annotation(self, sample_annotation_path: str) -> GroundTruthAnnotation:
        resolved = self._safe(sample_annotation_path)
        return GroundTruthAnnotation.model_validate(json.loads(resolved.read_text(encoding="utf-8")))

    def validate(self, manifest: DatasetManifest, *, require_media: bool = False) -> list[str]:
        warnings: list[str] = []
        for sample in manifest.samples:
            annotation = self.load_annotation(sample.annotation_path)
            if annotation.sample_id != sample.sample_id:
                raise DatasetValidationError(f"annotation sample mismatch for {sample.sample_id}")
            if require_media and sample.source.relative_path:
                media = self._safe(sample.source.relative_path)
                if not media.exists() or media.stat().st_size <= 0:
                    raise DatasetValidationError(f"media missing for {sample.sample_id}")
            elif sample.source.relative_path:
                media = self._safe(sample.source.relative_path)
                if not media.exists():
                    warnings.append(f"MEDIA_MISSING:{sample.sample_id}")
        return warnings


class TemporalMatcher:
    """Deterministic one-to-one timestamp-window matcher."""

    def match(
        self,
        states: Iterable[GroundTruthVisualState],
        predictions: Iterable[PredictionRecord],
        *,
        include_optional: bool = True,
    ) -> tuple[list[MatchRecord], list[str], list[str]]:
        considered = [
            state
            for state in states
            if state.importance is Importance.REQUIRED
            or (include_optional and state.importance is Importance.OPTIONAL)
        ]
        considered.sort(key=lambda s: (s.target_timestamp_seconds, s.state_id))
        preds = sorted(predictions, key=lambda p: (p.timestamp_seconds, p.prediction_id))
        used: set[str] = set()
        matches: list[MatchRecord] = []
        matched_states: set[str] = set()

        # Duplicate/equivalence groups are satisfied once by any member.
        satisfied_groups: set[str] = set()
        for state in considered:
            group = state.acceptable_equivalence_group or state.duplicate_group
            if group and group in satisfied_groups:
                matched_states.add(state.state_id)
                continue
            candidates = [
                pred
                for pred in preds
                if pred.prediction_id not in used
                and state.acceptable_start_seconds <= pred.timestamp_seconds <= state.acceptable_end_seconds
            ]
            if not candidates:
                continue
            chosen = min(
                candidates,
                key=lambda pred: (
                    abs(pred.timestamp_seconds - state.target_timestamp_seconds),
                    pred.timestamp_seconds,
                    pred.prediction_id,
                ),
            )
            used.add(chosen.prediction_id)
            matched_states.add(state.state_id)
            if group:
                satisfied_groups.add(group)
            matches.append(
                MatchRecord(
                    ground_truth_state_id=state.state_id,
                    prediction_id=chosen.prediction_id,
                    timestamp_error_seconds=abs(chosen.timestamp_seconds - state.target_timestamp_seconds),
                )
            )

        required_ids = {s.state_id for s in considered if s.importance is Importance.REQUIRED}
        false_negatives = sorted(required_ids - matched_states)
        false_positives = sorted(pred.prediction_id for pred in preds if pred.prediction_id not in used)
        return matches, false_positives, false_negatives


def _quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


class MetricEngine:
    @staticmethod
    def safe_ratio(numerator: int | float, denominator: int | float) -> float:
        return 0.0 if denominator == 0 else float(numerator) / float(denominator)

    def summarize(
        self,
        *,
        annotation: GroundTruthAnnotation,
        predictions: list[PredictionRecord],
        matches: list[MatchRecord],
        false_positives: list[str],
        false_negatives: list[str],
    ) -> MetricSummary:
        required = [s for s in annotation.states if s.importance is Importance.REQUIRED]
        matched_required_ids = {m.ground_truth_state_id for m in matches if m.ground_truth_state_id in {s.state_id for s in required}}
        tp = len(matches)
        fp = len(false_positives)
        fn = len(false_negatives)
        precision = self.safe_ratio(tp, tp + fp)
        recall = self.safe_ratio(tp, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        errors = [m.timestamp_error_seconds for m in matches]
        negative_hits = 0
        for pred in predictions:
            if any(n.start_seconds <= pred.timestamp_seconds <= n.end_seconds for n in annotation.negative_states):
                negative_hits += 1
        return MetricSummary(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=round(precision, 6),
            recall=round(recall, 6),
            f1=round(f1, 6),
            required_state_recall=round(self.safe_ratio(len(matched_required_ids), len(required)), 6),
            negative_state_violation_rate=round(self.safe_ratio(negative_hits, len(predictions)), 6),
            timestamp_error_mean=round(statistics.mean(errors), 6) if errors else None,
            timestamp_error_median=round(statistics.median(errors), 6) if errors else None,
            timestamp_error_p90=round(_quantile(errors, 0.90), 6) if errors else None,
            timestamp_error_max=round(max(errors), 6) if errors else None,
        )

    def aggregate(self, metrics: list[MetricSummary]) -> MetricSummary:
        tp = sum(m.tp for m in metrics)
        fp = sum(m.fp for m in metrics)
        fn = sum(m.fn for m in metrics)
        precision = self.safe_ratio(tp, tp + fp)
        recall = self.safe_ratio(tp, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return MetricSummary(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=round(precision, 6),
            recall=round(recall, 6),
            f1=round(f1, 6),
            required_state_recall=round(
                statistics.mean([m.required_state_recall for m in metrics]) if metrics else 0.0, 6
            ),
            negative_state_violation_rate=round(
                statistics.mean([m.negative_state_violation_rate for m in metrics]) if metrics else 0.0,
                6,
            ),
        )


class NotifyPredictionAdapter:
    """Read-only prediction adapter.

    Phase-9 official runs can map sample ids to per-stage golden/reference JSON files or
    generated job exports. Keeping this adapter file-based makes CI offline and ensures
    evaluation never mutates the production workspace.
    """

    def load_predictions(self, path: Path, level: PredictionLevel) -> list[PredictionRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get(level.value, payload if isinstance(payload, list) else [])
        return [PredictionRecord.model_validate(value) for value in values]


class BenchmarkRunner:
    def __init__(self) -> None:
        self.matcher = TemporalMatcher()
        self.metrics = MetricEngine()

    def evaluate_stage(
        self,
        annotation: GroundTruthAnnotation,
        predictions: list[PredictionRecord],
        level: PredictionLevel,
        config: BenchmarkConfig,
    ) -> StageEvaluationResult:
        matches, fps, fns = self.matcher.match(
            annotation.states,
            predictions,
            include_optional=config.include_optional_states,
        )
        summary = self.metrics.summarize(
            annotation=annotation,
            predictions=predictions,
            matches=matches,
            false_positives=fps,
            false_negatives=fns,
        )
        return StageEvaluationResult(
            stage=level,
            prediction_count=len(predictions),
            ground_truth_required_count=sum(s.importance is Importance.REQUIRED for s in annotation.states),
            metrics=summary,
            matches=matches,
            false_positives=fps,
            false_negatives=fns,
        )

    def run(
        self,
        *,
        dataset: DatasetManifest,
        annotations: dict[str, GroundTruthAnnotation],
        predictions: dict[str, dict[PredictionLevel, list[PredictionRecord]]],
        config: BenchmarkConfig,
    ) -> BenchmarkReport:
        sample_results: list[SampleQualityEvaluation] = []
        category_metrics: dict[str, list[MetricSummary]] = defaultdict(list)
        tag_metrics: dict[str, list[MetricSummary]] = defaultdict(list)
        for sample in sorted(dataset.samples, key=lambda item: item.sample_id):
            annotation = annotations[sample.sample_id]
            if annotation.review_status not in config.allowed_review_status:
                if config.strict:
                    raise DatasetValidationError(f"annotation not reviewed: {sample.sample_id}")
                continue
            level_predictions = predictions.get(sample.sample_id, {})
            stage_result = self.evaluate_stage(
                annotation,
                level_predictions.get(config.prediction_level, []),
                config.prediction_level,
                config,
            )
            sample_eval = SampleQualityEvaluation(
                sample_id=sample.sample_id,
                stages={config.prediction_level: stage_result},
            )
            sample_results.append(sample_eval)
            for category in sample.categories:
                category_metrics[category.value].append(stage_result.metrics)
            for tag in sample.tags:
                tag_metrics[tag].append(stage_result.metrics)
        aggregate = self.metrics.aggregate(
            [item.stages[config.prediction_level].metrics for item in sample_results]
        )
        dataset_fp = stable_fingerprint(dataset)
        benchmark_fp = stable_fingerprint(
            {
                "dataset": dataset_fp,
                "config": config,
                "predictions": predictions,
            }
        )
        return BenchmarkReport(
            dataset_fingerprint=dataset_fp,
            benchmark_input_fingerprint=benchmark_fp,
            prediction_level=config.prediction_level,
            aggregate=aggregate,
            by_category={k: self.metrics.aggregate(v) for k, v in sorted(category_metrics.items())},
            by_tag={k: self.metrics.aggregate(v) for k, v in sorted(tag_metrics.items())},
            samples=sample_results,
        )


class EndToEndQualityEvaluator:
    STAGES = (
        PredictionLevel.CANDIDATE,
        PredictionLevel.SEMANTIC,
        PredictionLevel.FINAL_SCREENSHOT,
        PredictionLevel.DOCUMENT,
    )

    def __init__(self) -> None:
        self.runner = BenchmarkRunner()
        self.metrics = MetricEngine()

    def evaluate_sample(
        self,
        sample_id: str,
        annotation: GroundTruthAnnotation,
        predictions: dict[PredictionLevel, list[PredictionRecord]],
    ) -> SampleQualityEvaluation:
        cfg = BenchmarkConfig(strict=True)
        stages = {
            level: self.runner.evaluate_stage(annotation, predictions.get(level, []), level, cfg)
            for level in self.STAGES
        }
        outcomes: list[RequiredStateOutcome] = []
        failures: list[FailureRecord] = []
        for state in annotation.states:
            if state.importance is not Importance.REQUIRED:
                continue
            found = {
                level: state.state_id in {m.ground_truth_state_id for m in stages[level].matches}
                for level in self.STAGES
            }
            earliest: str | None = None
            if not found[PredictionLevel.CANDIDATE]:
                earliest = "CANDIDATE_MISS"
            elif not found[PredictionLevel.SEMANTIC]:
                earliest = "SEMANTIC_FALSE_REJECTION"
            elif not found[PredictionLevel.FINAL_SCREENSHOT]:
                earliest = "PHASE7_LOSS"
            elif not found[PredictionLevel.DOCUMENT]:
                earliest = "DOCUMENT_LOSS"
            outcomes.append(
                RequiredStateOutcome(
                    ground_truth_state_id=state.state_id,
                    candidate_found=found[PredictionLevel.CANDIDATE],
                    semantic_kept=found[PredictionLevel.SEMANTIC],
                    final_screenshot_found=found[PredictionLevel.FINAL_SCREENSHOT],
                    document_found=found[PredictionLevel.DOCUMENT],
                    earliest_loss_stage=earliest,
                )
            )
            if earliest:
                failures.append(
                    FailureRecord(
                        sample_id=sample_id,
                        failure_type=earliest,
                        severity="CRITICAL",
                        ground_truth_state_id=state.state_id,
                        target_timestamp_seconds=state.target_timestamp_seconds,
                    )
                )
        # final-stage false positives get a stable diagnostic class.
        for prediction_id in stages[PredictionLevel.FINAL_SCREENSHOT].false_positives:
            failures.append(
                FailureRecord(
                    sample_id=sample_id,
                    failure_type="NO_GROUND_TRUTH_MATCH",
                    severity="MAJOR",
                    prediction_id=prediction_id,
                )
            )
        return SampleQualityEvaluation(
            sample_id=sample_id,
            stages=stages,
            required_state_outcomes=outcomes,
            failure_records=failures,
        )

    def evaluate(
        self,
        *,
        dataset: DatasetManifest,
        annotations: dict[str, GroundTruthAnnotation],
        predictions: dict[str, dict[PredictionLevel, list[PredictionRecord]]],
        pipeline_fingerprint: str,
        lock: bool = False,
    ) -> QualityBaselineReport:
        samples = [
            self.evaluate_sample(sample.sample_id, annotations[sample.sample_id], predictions.get(sample.sample_id, {}))
            for sample in sorted(dataset.samples, key=lambda s: s.sample_id)
            if annotations[sample.sample_id].review_status in {AnnotationReviewStatus.REVIEWED, AnnotationReviewStatus.LOCKED}
        ]
        stage_aggregates: dict[PredictionLevel, MetricSummary] = {}
        for level in self.STAGES:
            stage_aggregates[level] = self.metrics.aggregate([s.stages[level].metrics for s in samples])
        losses: dict[str, int] = defaultdict(int)
        failures: list[FailureRecord] = []
        for sample in samples:
            failures.extend(sample.failure_records)
            for outcome in sample.required_state_outcomes:
                if outcome.earliest_loss_stage:
                    losses[outcome.earliest_loss_stage] += 1
        final = stage_aggregates[PredictionLevel.DOCUMENT]
        return QualityBaselineReport(
            dataset_fingerprint=stable_fingerprint(dataset),
            pipeline_fingerprint=pipeline_fingerprint,
            overall=final,
            candidate=stage_aggregates[PredictionLevel.CANDIDATE],
            semantic=stage_aggregates[PredictionLevel.SEMANTIC],
            final_screenshot=stage_aggregates[PredictionLevel.FINAL_SCREENSHOT],
            document=stage_aggregates[PredictionLevel.DOCUMENT],
            required_visual_state_recall=final.required_state_recall,
            final_visual_note_precision=final.precision,
            final_visual_note_f1=final.f1,
            stage_loss_counts=dict(sorted(losses.items())),
            samples=samples,
            failures=failures,
            status="LOCKED" if lock else "DRAFT",
        )

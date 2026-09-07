from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .models import DatasetSplit, QualityBaselineReport, stable_fingerprint

CALIBRATION_PROFILE_VERSION = "1"
CALIBRATION_ALGORITHM_VERSION = "1"


class SearchStrategy(str, Enum):
    LOCAL_SWEEP = "LOCAL_SWEEP"
    GRID = "GRID"
    MANUAL_PROFILE = "MANUAL_PROFILE"


class RecommendationStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class CalibrationParameter(BaseModel):
    name: str
    phase: Literal[3, 4, 6, 7]
    current_value: float | int
    min_value: float | int
    max_value: float | int
    step: float | int
    dependency_stage: str
    description: str = ""

    @model_validator(mode="after")
    def range_is_valid(self) -> "CalibrationParameter":
        if self.min_value > self.current_value or self.current_value > self.max_value:
            raise ValueError("calibration current value outside allowed range")
        if self.step <= 0:
            raise ValueError("calibration step must be positive")
        return self


class CalibrationParameterRegistry:
    """Explicit allow-list of behavior-affecting existing settings.

    Values mirror the current Phase-8 defaults. The registry is metadata only; it never
    mutates AppSettings or .env by itself.
    """

    def __init__(self) -> None:
        self.parameters = {
            p.name: p
            for p in [
                CalibrationParameter(name="major_change_min_score", phase=3, current_value=0.30, min_value=0.10, max_value=0.60, step=0.05, dependency_stage="PHASE_3"),
                CalibrationParameter(name="timeline_stable_max_score", phase=3, current_value=0.10, min_value=0.04, max_value=0.25, step=0.02, dependency_stage="PHASE_3"),
                CalibrationParameter(name="min_stable_duration_seconds", phase=3, current_value=2.0, min_value=0.5, max_value=6.0, step=0.5, dependency_stage="PHASE_3"),
                CalibrationParameter(name="stability_window_min_duration_seconds", phase=4, current_value=2.0, min_value=0.5, max_value=6.0, step=0.5, dependency_stage="PHASE_4"),
                CalibrationParameter(name="min_candidate_spacing_seconds", phase=4, current_value=1.0, min_value=0.25, max_value=4.0, step=0.25, dependency_stage="PHASE_4"),
                CalibrationParameter(name="top_candidates_per_window", phase=4, current_value=2, min_value=1, max_value=5, step=1, dependency_stage="PHASE_4"),
                CalibrationParameter(name="min_alternate_ranking_score", phase=4, current_value=0.25, min_value=0.0, max_value=0.75, step=0.05, dependency_stage="PHASE_4"),
                CalibrationParameter(name="semantic_keep_threshold", phase=6, current_value=0.55, min_value=0.20, max_value=0.90, step=0.05, dependency_stage="PHASE_6", description="Logical decision threshold; bind to actual configured field when present."),
                CalibrationParameter(name="screenshot_quality_min_score", phase=7, current_value=0.35, min_value=0.10, max_value=0.80, step=0.05, dependency_stage="PHASE_7", description="Logical Phase-7 quality threshold registry entry."),
                CalibrationParameter(name="duplicate_phash_distance", phase=7, current_value=6, min_value=1, max_value=20, step=1, dependency_stage="PHASE_7", description="Logical Phase-7 duplicate threshold registry entry."),
            ]
        }

    def get(self, name: str) -> CalibrationParameter:
        if name not in self.parameters:
            raise KeyError(f"unknown calibration parameter: {name}")
        return self.parameters[name]

    def local_sweep(self, name: str, radius: int = 2) -> list[float | int]:
        p = self.get(name)
        values: list[float | int] = []
        for offset in range(-radius, radius + 1):
            value = p.current_value + p.step * offset
            value = max(p.min_value, min(p.max_value, value))
            if isinstance(p.current_value, int):
                value = int(round(value))
            else:
                value = round(float(value), 8)
            if value not in values:
                values.append(value)
        return values


class CalibrationProfile(BaseModel):
    version: str = CALIBRATION_PROFILE_VERSION
    profile_id: str
    base_pipeline_fingerprint: str
    changes: dict[str, float | int]

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "version": self.version,
                "base": self.base_pipeline_fingerprint,
                "changes": self.changes,
                "algorithm": CALIBRATION_ALGORITHM_VERSION,
            }
        )


class CalibrationGuardrails(BaseModel):
    min_meaningful_delta: float = 0.0025
    max_precision_regression: float = 0.01
    max_candidate_growth_ratio: float = 0.25
    max_category_recall_regression: float = 0.02
    reject_new_critical_failure: bool = True
    reject_new_required_state_miss: bool = True


class CalibrationScorecard(BaseModel):
    primary_metric_pass: bool
    precision_guardrail_pass: bool
    required_state_regression_pass: bool
    critical_failure_pass: bool
    category_guardrail_pass: bool = True
    candidate_growth_pass: bool = True
    validation_pass: bool = True

    @property
    def accepted(self) -> bool:
        return all(self.model_dump().values())


class CalibrationComparison(BaseModel):
    baseline_recall: float
    candidate_recall: float
    recall_delta: float
    baseline_precision: float
    candidate_precision: float
    precision_delta: float
    baseline_f1: float
    candidate_f1: float
    f1_delta: float
    new_critical_failures: int = 0
    new_required_misses: int = 0
    scorecard: CalibrationScorecard
    recommendation: RecommendationStatus


class CalibrationRunner:
    def __init__(self, registry: CalibrationParameterRegistry | None = None) -> None:
        self.registry = registry or CalibrationParameterRegistry()

    def generate_local_profiles(
        self,
        *,
        parameter_name: str,
        base_pipeline_fingerprint: str,
        max_runs: int = 50,
    ) -> list[CalibrationProfile]:
        values = self.registry.local_sweep(parameter_name)
        profiles = [
            CalibrationProfile(
                profile_id=f"{parameter_name}-{index:02d}",
                base_pipeline_fingerprint=base_pipeline_fingerprint,
                changes={parameter_name: value},
            )
            for index, value in enumerate(values)
        ]
        return profiles[:max_runs]

    def compare(
        self,
        baseline: QualityBaselineReport,
        candidate: QualityBaselineReport,
        *,
        guardrails: CalibrationGuardrails | None = None,
        new_critical_failures: int = 0,
        new_required_misses: int = 0,
        validation_pass: bool = True,
    ) -> CalibrationComparison:
        g = guardrails or CalibrationGuardrails()
        recall_delta = candidate.required_visual_state_recall - baseline.required_visual_state_recall
        precision_delta = candidate.final_visual_note_precision - baseline.final_visual_note_precision
        f1_delta = candidate.final_visual_note_f1 - baseline.final_visual_note_f1
        scorecard = CalibrationScorecard(
            primary_metric_pass=recall_delta >= g.min_meaningful_delta or f1_delta >= g.min_meaningful_delta,
            precision_guardrail_pass=precision_delta >= -g.max_precision_regression,
            required_state_regression_pass=(not g.reject_new_required_state_miss or new_required_misses == 0),
            critical_failure_pass=(not g.reject_new_critical_failure or new_critical_failures == 0),
            validation_pass=validation_pass,
        )
        if scorecard.accepted:
            recommendation = RecommendationStatus.ACCEPT
        elif new_critical_failures or new_required_misses:
            recommendation = RecommendationStatus.REJECT
        else:
            recommendation = RecommendationStatus.MANUAL_REVIEW
        return CalibrationComparison(
            baseline_recall=baseline.required_visual_state_recall,
            candidate_recall=candidate.required_visual_state_recall,
            recall_delta=round(recall_delta, 6),
            baseline_precision=baseline.final_visual_note_precision,
            candidate_precision=candidate.final_visual_note_precision,
            precision_delta=round(precision_delta, 6),
            baseline_f1=baseline.final_visual_note_f1,
            candidate_f1=candidate.final_visual_note_f1,
            f1_delta=round(f1_delta, 6),
            new_critical_failures=new_critical_failures,
            new_required_misses=new_required_misses,
            scorecard=scorecard,
            recommendation=recommendation,
        )

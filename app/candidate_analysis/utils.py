from __future__ import annotations

import bisect
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

import numpy as np

from app.core.exceptions import CandidateAnalysisValidationError
from app.storage.workspace import WorkspaceManager

T = TypeVar("T")


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def normalized_weighted(values: Sequence[tuple[float, float]]) -> float:
    total = sum(weight for _, weight in values if weight > 0)
    if total <= 0:
        return 0.0
    return clamp01(sum(clamp01(value) * weight for value, weight in values if weight > 0) / total)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def safe_workspace_path(workspace: WorkspaceManager, job_id: str, relative: str) -> Path:
    root = workspace.workspace(job_id)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CandidateAnalysisValidationError("Candidate artifact path escaped the job workspace.") from exc
    return candidate


def time_slice(items: Sequence[T], timestamps: Sequence[float], start: float, end: float) -> Sequence[T]:
    left = bisect.bisect_left(timestamps, start - 1e-9)
    right = bisect.bisect_right(timestamps, end + 1e-9)
    return items[left:right]


def ratio(count: int, total: int) -> float:
    return float(count / total) if total > 0 else 0.0


def duration_saturation(duration: float, saturation: float) -> float:
    return clamp01(duration / max(saturation, 1e-9))


def score_distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
    }


def unique_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

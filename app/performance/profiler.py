from __future__ import annotations

import hashlib
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median
from typing import Iterator

import psutil

from .models import (
    ArtifactMetrics,
    HardwareSummary,
    PerformanceComparison,
    PerformanceOptimizationScorecard,
    PerformanceProfilingConfig,
    PerformanceRun,
    StagePerformanceMetrics,
)
from app.evaluation.models import stable_fingerprint


class StageTimer:
    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self._durations: dict[str, list[float]] = {}

    def start(self, stage: str) -> None:
        if stage in self._starts:
            raise RuntimeError(f"stage timer already running: {stage}")
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        if stage not in self._starts:
            raise RuntimeError(f"stage timer not running: {stage}")
        duration = time.perf_counter() - self._starts.pop(stage)
        self._durations.setdefault(stage, []).append(duration)
        return duration

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        self.start(stage)
        try:
            yield
        finally:
            self.stop(stage)

    def durations(self, stage: str) -> tuple[float, ...]:
        return tuple(self._durations.get(stage, ()))


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval = max(0.1, interval_seconds)
        self.process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss: list[int] = []
        self._cpu: list[float] = []

    def _loop(self) -> None:
        self.process.cpu_percent(interval=None)
        while not self._stop.wait(self.interval):
            try:
                self._rss.append(self.process.memory_info().rss)
                self._cpu.append(self.process.cpu_percent(interval=None))
            except (psutil.Error, OSError):
                break

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int | None, float | None, float | None]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval * 2))
        peak_rss = max(self._rss) if self._rss else self.process.memory_info().rss
        cpu_mean = mean(self._cpu) if self._cpu else None
        cpu_peak = max(self._cpu) if self._cpu else None
        return peak_rss, cpu_mean, cpu_peak


class PipelinePerformanceProfiler:
    def __init__(self, config: PerformanceProfilingConfig | None = None) -> None:
        self.config = config or PerformanceProfilingConfig()
        self.timer = StageTimer()
        self._stage_start_rss: dict[str, int] = {}
        self._stage_monitor: dict[str, ResourceMonitor] = {}
        self._metrics: list[StagePerformanceMetrics] = []

    def start_stage(self, stage: str) -> None:
        self._stage_start_rss[stage] = psutil.Process(os.getpid()).memory_info().rss
        monitor = ResourceMonitor(self.config.resource_sample_interval_seconds)
        self._stage_monitor[stage] = monitor
        monitor.start()
        self.timer.start(stage)

    def stop_stage(self, stage: str, *, input_count: int | None = None, output_count: int | None = None, cache_hit: bool = False) -> StagePerformanceMetrics:
        duration = self.timer.stop(stage)
        monitor = self._stage_monitor.pop(stage)
        peak_rss, cpu_mean, cpu_peak = monitor.stop()
        start_rss = self._stage_start_rss.pop(stage)
        metric = StagePerformanceMetrics(
            stage=stage,
            duration_seconds=round(duration, 6),
            cpu_percent_mean=round(cpu_mean, 3) if cpu_mean is not None else None,
            cpu_percent_peak=round(cpu_peak, 3) if cpu_peak is not None else None,
            rss_memory_start_bytes=start_rss,
            rss_memory_peak_bytes=peak_rss,
            rss_memory_delta_bytes=(peak_rss - start_rss) if peak_rss is not None else None,
            input_count=input_count,
            output_count=output_count,
            cache_hit=cache_hit,
        )
        self._metrics.append(metric)
        return metric

    @contextmanager
    def stage(self, stage: str, **stop_kwargs: int | bool | None) -> Iterator[None]:
        self.start_stage(stage)
        try:
            yield
        finally:
            self.stop_stage(stage, **stop_kwargs)

    def build_run(self, *, run_id: str, sample_id: str, mode, pipeline_fingerprint: str, video_duration_seconds: float = 0, counts: dict[str, int] | None = None, artifact_metrics: ArtifactMetrics | None = None) -> PerformanceRun:
        vm = psutil.virtual_memory()
        hardware = HardwareSummary(
            cpu_count_logical=psutil.cpu_count(logical=True) or 1,
            ram_total_bytes=vm.total,
            platform=platform.platform(),
            python_version=platform.python_version(),
        )
        return PerformanceRun(
            run_id=run_id,
            sample_id=sample_id,
            mode=mode,
            pipeline_fingerprint=pipeline_fingerprint,
            performance_config_fingerprint=stable_fingerprint(self.config),
            hardware=hardware,
            stage_metrics=list(self._metrics),
            artifact_metrics=artifact_metrics or ArtifactMetrics(),
            video_duration_seconds=video_duration_seconds,
            counts=counts or {},
        )


def calculate_workspace_size(root: Path) -> int:
    root = root.resolve()
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_symlink():
                continue
            resolved = path.resolve()
            resolved.relative_to(root)
            if path.is_file():
                total += path.stat().st_size
        except (OSError, ValueError):
            continue
    return total


def streaming_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_repetitions(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def compare_performance(
    baseline: PerformanceRun,
    candidate: PerformanceRun,
    *,
    quality_passed: bool = True,
    critical_regressions: int = 0,
    config: PerformanceProfilingConfig | None = None,
) -> PerformanceComparison:
    cfg = config or PerformanceProfilingConfig()
    base_runtime = baseline.total_runtime_seconds
    cand_runtime = candidate.total_runtime_seconds
    improvement = 0.0 if base_runtime == 0 else (base_runtime - cand_runtime) / base_runtime
    base_peak = max((m.rss_memory_peak_bytes or 0 for m in baseline.stage_metrics), default=0)
    cand_peak = max((m.rss_memory_peak_bytes or 0 for m in candidate.stage_metrics), default=0)
    memory_regression = 0.0 if base_peak == 0 else (cand_peak - base_peak) / base_peak
    base_disk = baseline.artifact_metrics.workspace_bytes
    cand_disk = candidate.artifact_metrics.workspace_bytes
    disk_regression = 0.0 if base_disk == 0 else (cand_disk - base_disk) / base_disk
    scorecard = PerformanceOptimizationScorecard(
        runtime_improved=improvement >= cfg.min_meaningful_delta_ratio,
        memory_within_guardrail=memory_regression <= cfg.max_memory_regression_ratio,
        disk_within_guardrail=disk_regression <= cfg.max_disk_regression_ratio,
        quality_passed=quality_passed,
        critical_regressions_zero=critical_regressions == 0,
    )
    return PerformanceComparison(
        baseline_runtime_seconds=round(base_runtime, 6),
        candidate_runtime_seconds=round(cand_runtime, 6),
        runtime_delta_seconds=round(cand_runtime - base_runtime, 6),
        runtime_improvement_ratio=round(improvement, 6),
        baseline_peak_memory_bytes=base_peak,
        candidate_peak_memory_bytes=cand_peak,
        memory_regression_ratio=round(memory_regression, 6),
        baseline_workspace_bytes=base_disk,
        candidate_workspace_bytes=cand_disk,
        disk_regression_ratio=round(disk_regression, 6),
        scorecard=scorecard,
    )

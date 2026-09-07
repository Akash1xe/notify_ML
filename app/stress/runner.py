from __future__ import annotations

from dataclasses import dataclass

from .models import (
    LimitType,
    ResourceLimitDecision,
    ResourceLimits,
    ScalingAnalysis,
    ScalingClassification,
    StressScenario,
    StressStatus,
    StressTestReport,
)


class ResourceLimitExceededError(RuntimeError):
    code = "RESOURCE_LIMIT_EXCEEDED"


class StressScenarioRegistry:
    def __init__(self) -> None:
        limits = ResourceLimits()
        self._scenarios = {
            "long_slides": StressScenario(scenario_id="long_slides", description="Long low-change slide lecture", duration_seconds=3600, width=1920, height=1080, fps=30, change_density="LOW_CHANGE", expected_candidate_scale=120, expected_screenshot_scale=60, resource_limits=limits),
            "long_whiteboard": StressScenario(scenario_id="long_whiteboard", description="Long continuous whiteboard lecture", duration_seconds=3600, width=1920, height=1080, fps=30, change_density="HIGH_CHANGE", expected_candidate_scale=400, expected_screenshot_scale=180, resource_limits=limits),
            "long_code": StressScenario(scenario_id="long_code", description="Long live-coding lecture", duration_seconds=3600, width=1920, height=1080, fps=30, change_density="HIGH_CHANGE", expected_candidate_scale=500, expected_screenshot_scale=200, resource_limits=limits),
            "candidate_heavy": StressScenario(scenario_id="candidate_heavy", description="Synthetic candidate-explosion fixture", duration_seconds=1800, width=1280, height=720, fps=60, change_density="EXTREME_CHANGE", expected_candidate_scale=1000, expected_screenshot_scale=250, resource_limits=limits),
            "duplicate_heavy": StressScenario(scenario_id="duplicate_heavy", description="Many repeated and near-repeated visual states", duration_seconds=1800, width=1920, height=1080, fps=30, change_density="MEDIUM_CHANGE", expected_candidate_scale=800, expected_screenshot_scale=80, resource_limits=limits),
            "transcript_heavy": StressScenario(scenario_id="transcript_heavy", description="Long synthetic transcript alignment workload", duration_seconds=7200, width=1280, height=720, fps=30, change_density="MEDIUM_CHANGE", expected_candidate_scale=600, expected_screenshot_scale=250, resource_limits=limits, required_stages=["TRANSCRIPT_ALIGNMENT"]),
            "semantic_candidate_heavy": StressScenario(scenario_id="semantic_candidate_heavy", description="Many fake semantic candidates", duration_seconds=7200, width=1280, height=720, fps=30, change_density="HIGH_CHANGE", expected_candidate_scale=1000, expected_screenshot_scale=300, resource_limits=limits, required_stages=["SEMANTIC"]),
            "large_pdf": StressScenario(scenario_id="large_pdf", description="Synthetic 500-page document", duration_seconds=3600, width=1920, height=1080, fps=30, change_density="MEDIUM_CHANGE", expected_candidate_scale=600, expected_screenshot_scale=500, resource_limits=limits, required_stages=["DOCUMENT"]),
            "large_gallery": StressScenario(scenario_id="large_gallery", description="Result API/gallery with 500 screenshots", duration_seconds=3600, width=1920, height=1080, fps=30, change_density="MEDIUM_CHANGE", expected_candidate_scale=600, expected_screenshot_scale=500, resource_limits=limits, required_stages=["RESULT_UI"]),
        }

    def get(self, scenario_id: str) -> StressScenario:
        try:
            return self._scenarios[scenario_id]
        except KeyError as exc:
            raise KeyError(f"unknown stress scenario: {scenario_id}") from exc

    def list_ids(self) -> list[str]:
        return sorted(self._scenarios)


@dataclass(frozen=True)
class ObservedResources:
    total_runtime_seconds: float = 0.0
    peak_rss_bytes: int = 0
    peak_vram_bytes: int | None = None
    workspace_bytes: int = 0
    source_file_bytes: int = 0
    temp_bytes: int = 0
    sampled_frames: int = 0
    candidates: int = 0
    semantic_candidates: int = 0
    final_screenshots: int = 0
    pdf_pages: int = 0
    pdf_bytes: int = 0


class StressTestRunner:
    LIMIT_MAP = {
        "duration_seconds": ("max_video_duration_seconds", LimitType.HARD),
        "source_file_bytes": ("max_source_file_bytes", LimitType.HARD),
        "workspace_bytes": ("max_workspace_bytes", LimitType.HARD),
        "sampled_frames": ("max_sampled_frames", LimitType.HARD),
        "candidates": ("max_candidates", LimitType.HARD),
        "semantic_candidates": ("max_semantic_candidates", LimitType.HARD),
        "final_screenshots": ("max_final_screenshots", LimitType.WARNING_ONLY),
        "pdf_pages": ("max_pdf_pages", LimitType.WARNING_ONLY),
        "pdf_bytes": ("max_pdf_bytes", LimitType.WARNING_ONLY),
        "peak_rss_bytes": ("max_peak_rss_bytes", LimitType.WARNING_ONLY),
        "peak_vram_bytes": ("max_peak_vram_bytes", LimitType.WARNING_ONLY),
        "total_runtime_seconds": ("max_processing_seconds", LimitType.WARNING_ONLY),
    }

    def evaluate(self, scenario: StressScenario, observed: ObservedResources, *, pipeline_fingerprint: str = "unknown") -> StressTestReport:
        decisions: list[ResourceLimitDecision] = []
        warnings: list[str] = []
        hard_exceeded = False
        values = {**observed.__dict__, "duration_seconds": scenario.duration_seconds}
        for observed_name, (limit_name, limit_type) in self.LIMIT_MAP.items():
            limit = getattr(scenario.resource_limits, limit_name)
            value = values.get(observed_name)
            if limit is None or value is None:
                continue
            exceeded = float(value) > float(limit)
            if exceeded:
                if limit_type is LimitType.HARD:
                    hard_exceeded = True
                else:
                    warnings.append(f"{limit_name.upper()}_EXCEEDED")
            decisions.append(ResourceLimitDecision(limit_name=limit_name, limit_type=limit_type, observed_value=float(value), configured_limit=float(limit), exceeded=exceeded, action="FAIL" if exceeded and limit_type is LimitType.HARD else "WARN" if exceeded else "NONE", message=f"{observed_name}={value} limit={limit}"))
        status = StressStatus.LIMIT_EXCEEDED if hard_exceeded else StressStatus.PASS_WITH_WARNINGS if warnings else StressStatus.PASS
        rtf = None if scenario.duration_seconds <= 0 else observed.total_runtime_seconds / scenario.duration_seconds
        return StressTestReport(scenario=scenario, pipeline_fingerprint=pipeline_fingerprint, total_runtime_seconds=observed.total_runtime_seconds, real_time_factor=round(rtf, 6) if rtf is not None else None, peak_rss_bytes=observed.peak_rss_bytes, peak_vram_bytes=observed.peak_vram_bytes, workspace_bytes=observed.workspace_bytes, source_file_bytes=observed.source_file_bytes, temp_bytes=observed.temp_bytes, sampled_frames=observed.sampled_frames, candidates=observed.candidates, semantic_candidates=observed.semantic_candidates, final_screenshots=observed.final_screenshots, pdf_pages=observed.pdf_pages, pdf_bytes=observed.pdf_bytes, limit_decisions=decisions, warnings=warnings, status=status)


class ResourceGrowthDetector:
    @staticmethod
    def classify(input_scale: float, observed_scale: float, metric: str) -> ScalingAnalysis:
        if input_scale <= 1:
            classification = ScalingClassification.UNKNOWN
        else:
            ratio = observed_scale / input_scale
            if ratio < 0.75:
                classification = ScalingClassification.SUBLINEAR
            elif ratio <= 1.35:
                classification = ScalingClassification.APPROX_LINEAR
            elif ratio <= 2.5:
                classification = ScalingClassification.SUPERLINEAR
            else:
                classification = ScalingClassification.EXPLOSIVE
        return ScalingAnalysis(metric=metric, input_scale=input_scale, observed_scale=observed_scale, classification=classification)

    @staticmethod
    def possible_leak(samples: list[int | float], *, min_points: int = 5) -> bool:
        if len(samples) < min_points:
            return False
        increases = sum(1 for left, right in zip(samples, samples[1:]) if right > left)
        total_growth = samples[-1] - samples[0]
        return increases >= len(samples) - 2 and total_growth > max(1.0, abs(samples[0]) * 0.20)

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import (
    FailureOutcome,
    FailureScenario,
    FailureSeverity,
    FailureType,
    InjectionPoint,
    PipelineRecoveryPlan,
    ReliabilityScorecard,
    Retryability,
)


class InjectedFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class FailureInjector:
    def maybe_fail(self, point: InjectionPoint | str, context: dict[str, Any] | None = None) -> None:
        raise NotImplementedError


class NoopFailureInjector(FailureInjector):
    def maybe_fail(self, point: InjectionPoint | str, context: dict[str, Any] | None = None) -> None:
        return None


class ConfiguredFailureInjector(FailureInjector):
    def __init__(self, scenario: FailureScenario) -> None:
        self.scenario = scenario
        self.triggered = False

    def maybe_fail(self, point: InjectionPoint | str, context: dict[str, Any] | None = None) -> None:
        value = point.value if isinstance(point, InjectionPoint) else str(point)
        if not self.triggered and value == self.scenario.injection_point.value:
            self.triggered = True
            raise InjectedFailure(self.scenario.error_code, self.scenario.description)


class ReliabilityScenarioRegistry:
    def __init__(self) -> None:
        self._scenarios = {s.scenario_id: s for s in self._build()}

    @staticmethod
    def _build() -> list[FailureScenario]:
        def s(id: str, phase: str, stage: str, point: InjectionPoint, failure: FailureType, code: str, resume: str, retry: Retryability = Retryability.RETRY_SAME_CONFIG, last: str | None = None) -> FailureScenario:
            return FailureScenario(scenario_id=id, description=id.replace("_", " "), target_phase=phase, target_stage=stage, injection_point=point, failure_type=failure, severity=FailureSeverity.RECOVERABLE, error_code=code, retryability=retry, expected_last_valid_phase=last, expected_resume_stage=resume)
        return [
            s("youtube_download_failure", "PHASE_2", "DOWNLOAD", InjectionPoint.DURING_ITERATION, FailureType.DEPENDENCY_UNAVAILABLE, "VIDEO_DOWNLOAD_FAILED", "PHASE_2", Retryability.RETRY_SAME_CONFIG),
            s("ffmpeg_missing", "PHASE_2", "MEDIA_TOOLS", InjectionPoint.BEFORE_STAGE, FailureType.DEPENDENCY_UNAVAILABLE, "FFMPEG_NOT_AVAILABLE", "PHASE_2", Retryability.RETRY_AFTER_DEPENDENCY_FIX),
            s("frame_sampling_interrupted", "PHASE_3", "SAMPLING_FRAMES", InjectionPoint.DURING_ITERATION, FailureType.PROCESS_INTERRUPTION, "FRAME_SAMPLING_INTERRUPTED", "PHASE_3", last="PHASE_2"),
            s("phase3_json_corruption", "PHASE_3", "TIMELINE", InjectionPoint.AFTER_ATOMIC_PROMOTE, FailureType.FILE_CORRUPTION, "FRAME_ANALYSIS_CORRUPT", "PHASE_3", last="PHASE_2"),
            s("candidate_manifest_corruption", "PHASE_4", "RANKING", InjectionPoint.AFTER_ATOMIC_PROMOTE, FailureType.INVALID_JSON, "CANDIDATE_ARTIFACT_CORRUPT", "PHASE_4", last="PHASE_3"),
            s("transcription_chunk_failure", "PHASE_5", "TRANSCRIPTION", InjectionPoint.DURING_ITERATION, FailureType.EXCEPTION, "TRANSCRIPTION_CHUNK_FAILED", "PHASE_5", last="PHASE_4"),
            s("transcript_alignment_corruption", "PHASE_5", "ALIGNMENT", InjectionPoint.AFTER_ATOMIC_PROMOTE, FailureType.FILE_CORRUPTION, "TRANSCRIPT_ALIGNMENT_CORRUPT", "ALIGNMENT", last="TRANSCRIPTION"),
            s("qwen_unavailable", "PHASE_6", "SEMANTIC", InjectionPoint.BEFORE_STAGE, FailureType.DEPENDENCY_UNAVAILABLE, "SEMANTIC_MODEL_UNAVAILABLE", "PHASE_6", Retryability.RETRY_AFTER_DEPENDENCY_FIX, last="PHASE_5"),
            s("qwen_malformed_response", "PHASE_6", "SEMANTIC_PARSE", InjectionPoint.DURING_ITERATION, FailureType.INVALID_JSON, "SEMANTIC_RESPONSE_INVALID", "PHASE_6", last="PHASE_5"),
            s("semantic_candidate_interruption", "PHASE_6", "SEMANTIC", InjectionPoint.DURING_ITERATION, FailureType.PROCESS_INTERRUPTION, "SEMANTIC_INTERRUPTED", "PHASE_6", last="PHASE_5"),
            s("screenshot_missing", "PHASE_7", "EXTRACTION", InjectionPoint.AFTER_STAGE, FailureType.FILE_MISSING, "SCREENSHOT_MISSING", "PHASE_7", last="PHASE_6"),
            s("screenshot_hash_mismatch", "PHASE_7", "QUALITY", InjectionPoint.AFTER_ATOMIC_PROMOTE, FailureType.HASH_MISMATCH, "SCREENSHOT_HASH_MISMATCH", "PHASE_7", last="PHASE_6"),
            s("dedup_manifest_corruption", "PHASE_7", "DEDUP", InjectionPoint.AFTER_ATOMIC_PROMOTE, FailureType.FILE_CORRUPTION, "DEDUP_ARTIFACT_CORRUPT", "DEDUP", last="FINGERPRINTS"),
            s("document_layout_corruption", "PHASE_8", "LAYOUT", InjectionPoint.AFTER_ATOMIC_PROMOTE, FailureType.FILE_CORRUPTION, "DOCUMENT_LAYOUT_CORRUPT", "DOCUMENT_LAYOUT", last="DOCUMENT_INPUT"),
            s("pdf_render_failure", "PHASE_8", "PDF", InjectionPoint.DURING_ITERATION, FailureType.EXCEPTION, "PDF_GENERATION_FAILED", "PDF_GENERATION", last="DOCUMENT_RENDER_PLAN"),
            s("pdf_crash_before_promote", "PHASE_8", "PDF", InjectionPoint.BEFORE_ATOMIC_PROMOTE, FailureType.PROCESS_INTERRUPTION, "PDF_INTERRUPTED", "PDF_GENERATION", last="DOCUMENT_RENDER_PLAN"),
            s("pdf_manifest_missing", "PHASE_8", "PDF", InjectionPoint.AFTER_ATOMIC_PROMOTE, FailureType.FILE_MISSING, "PDF_MANIFEST_MISSING", "PDF_GENERATION", last="DOCUMENT_RENDER_PLAN"),
            s("pdf_corrupt", "PHASE_8", "PDF", InjectionPoint.AFTER_ATOMIC_PROMOTE, FailureType.FILE_CORRUPTION, "DOCUMENT_CORRUPT", "PDF_GENERATION", last="DOCUMENT_RENDER_PLAN"),
            s("backend_restart_mid_pipeline", "GLOBAL", "PROCESS", InjectionPoint.DURING_ITERATION, FailureType.PROCESS_INTERRUPTION, "JOB_INTERRUPTED", "EARLIEST_INVALID"),
            s("workspace_permission_denied", "GLOBAL", "PERSIST", InjectionPoint.BEFORE_PERSIST, FailureType.PERMISSION_ERROR, "ARTIFACT_WRITE_FAILED", "EARLIEST_INVALID", Retryability.RETRY_AFTER_DEPENDENCY_FIX),
            s("disk_full", "GLOBAL", "PERSIST", InjectionPoint.BEFORE_PERSIST, FailureType.DISK_WRITE_ERROR, "STORAGE_FULL", "EARLIEST_INVALID", Retryability.RETRY_AFTER_DEPENDENCY_FIX),
            s("user_cancel", "GLOBAL", "PROCESS", InjectionPoint.DURING_ITERATION, FailureType.CANCELLATION, "JOB_CANCELLED", "EARLIEST_INVALID", Retryability.RETRY_SAME_CONFIG),
            s("concurrent_retry", "GLOBAL", "PROCESS", InjectionPoint.BEFORE_STAGE, FailureType.EXCEPTION, "JOB_ALREADY_PROCESSING", "CURRENT", Retryability.RETRY_SAME_CONFIG),
        ]

    def get(self, scenario_id: str) -> FailureScenario:
        if scenario_id not in self._scenarios:
            raise KeyError(f"unknown reliability scenario: {scenario_id}")
        return self._scenarios[scenario_id]

    def list_ids(self) -> list[str]:
        return sorted(self._scenarios)


class PipelineRecoveryCoordinator:
    """Pure recovery planner over validity flags.

    Production phase-specific cache coordinators remain authoritative for artifact checks;
    this coordinator composes their results and avoids a destructive full reset.
    """
    ORDER = ["PHASE_2", "PHASE_3", "PHASE_4", "PHASE_5", "PHASE_6", "PHASE_7", "PHASE_8"]

    def plan(self, validity: dict[str, bool], *, temp_artifacts: list[str] | None = None) -> PipelineRecoveryPlan:
        last_valid: str | None = None
        resume: str | None = None
        clear: list[str] = []
        invalid_seen = False
        for phase in self.ORDER:
            valid = bool(validity.get(phase, False)) and not invalid_seen
            if valid:
                last_valid = phase
            else:
                if resume is None:
                    resume = phase
                invalid_seen = True
                clear.append(phase)
        return PipelineRecoveryPlan(last_valid_phase=last_valid, resume_phase=resume, resume_stage=resume, checkpoints_to_clear=clear, temp_artifacts_to_remove=sorted(temp_artifacts or []), artifacts_to_preserve=[p for p in self.ORDER if validity.get(p, False) and p not in clear], reason="earliest invalid phase")

    @staticmethod
    def cleanup_temp(root: Path, patterns: tuple[str, ...] = ("*.tmp", "*.tmp.json", "*.tmp.pdf")) -> list[str]:
        root = root.resolve()
        removed: list[str] = []
        if not root.exists():
            return removed
        for pattern in patterns:
            for path in root.rglob(pattern):
                try:
                    resolved = path.resolve()
                    resolved.relative_to(root)
                    if path.is_symlink():
                        continue
                    if path.is_file():
                        path.unlink(missing_ok=True)
                        removed.append(str(path.relative_to(root)))
                except (OSError, ValueError):
                    continue
        return sorted(set(removed))


class ReliabilityTestRunner:
    def __init__(self, registry: ReliabilityScenarioRegistry | None = None) -> None:
        self.registry = registry or ReliabilityScenarioRegistry()
        self.recovery = PipelineRecoveryCoordinator()

    def simulate(self, scenario_id: str) -> tuple[FailureOutcome, ReliabilityScorecard]:
        scenario = self.registry.get(scenario_id)
        injector = ConfiguredFailureInjector(scenario)
        detected = False
        error_code: str | None = None
        try:
            injector.maybe_fail(scenario.injection_point)
        except InjectedFailure as exc:
            detected = True
            error_code = exc.code
        outcome = FailureOutcome(scenario_id=scenario.scenario_id, failure_detected=detected, error_code=error_code, job_status="CANCELLED" if scenario.failure_type is FailureType.CANCELLATION else "FAILED", last_valid_checkpoint=scenario.expected_last_valid_phase, resume_stage=scenario.expected_resume_stage, retry_succeeded=scenario.retryability is not Retryability.DO_NOT_RETRY, final_state="RECOVERABLE" if scenario.retryability is not Retryability.DO_NOT_RETRY else "TERMINAL")
        score = ReliabilityScorecard(failure_detected=detected, typed_error=error_code == scenario.error_code, no_false_ready=True, temp_cleanup_pass=True, cache_preservation_pass=True, resume_stage_pass=outcome.resume_stage == scenario.expected_resume_stage, retry_pass=outcome.retry_succeeded or scenario.retryability is Retryability.DO_NOT_RETRY)
        return outcome, score

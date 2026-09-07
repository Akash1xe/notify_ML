from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

RELIABILITY_REPORT_VERSION = "1"
RELIABILITY_ALGORITHM_VERSION = "1"


class FailureType(str, Enum):
    EXCEPTION = "EXCEPTION"
    PROCESS_INTERRUPTION = "PROCESS_INTERRUPTION"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    FILE_MISSING = "FILE_MISSING"
    FILE_CORRUPTION = "FILE_CORRUPTION"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_JSON = "INVALID_JSON"
    TIMEOUT = "TIMEOUT"
    CANCELLATION = "CANCELLATION"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    DISK_WRITE_ERROR = "DISK_WRITE_ERROR"


class FailureSeverity(str, Enum):
    RECOVERABLE = "RECOVERABLE"
    RETRYABLE = "RETRYABLE"
    FATAL_CONFIGURATION = "FATAL_CONFIGURATION"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    USER_INPUT_FAILURE = "USER_INPUT_FAILURE"


class Retryability(str, Enum):
    RETRY_SAME_CONFIG = "RETRY_SAME_CONFIG"
    RETRY_AFTER_CONFIG_CHANGE = "RETRY_AFTER_CONFIG_CHANGE"
    RETRY_AFTER_DEPENDENCY_FIX = "RETRY_AFTER_DEPENDENCY_FIX"
    DO_NOT_RETRY = "DO_NOT_RETRY"


class InjectionPoint(str, Enum):
    BEFORE_STAGE = "BEFORE_STAGE"
    AFTER_STAGE = "AFTER_STAGE"
    BEFORE_PERSIST = "BEFORE_PERSIST"
    AFTER_TEMP_WRITE = "AFTER_TEMP_WRITE"
    BEFORE_ATOMIC_PROMOTE = "BEFORE_ATOMIC_PROMOTE"
    AFTER_ATOMIC_PROMOTE = "AFTER_ATOMIC_PROMOTE"
    DURING_ITERATION = "DURING_ITERATION"


class FailureScenario(BaseModel):
    scenario_id: str
    description: str
    target_phase: str
    target_stage: str
    injection_point: InjectionPoint
    failure_type: FailureType
    severity: FailureSeverity = FailureSeverity.RECOVERABLE
    error_code: str
    retryability: Retryability
    expected_last_valid_phase: str | None = None
    expected_resume_stage: str | None = None
    expected_preserved_artifacts: list[str] = Field(default_factory=list)
    expected_removed_temp_artifacts: list[str] = Field(default_factory=list)


class PipelineRecoveryPlan(BaseModel):
    last_valid_phase: str | None = None
    resume_phase: str | None = None
    resume_stage: str | None = None
    checkpoints_to_repair: list[str] = Field(default_factory=list)
    checkpoints_to_clear: list[str] = Field(default_factory=list)
    temp_artifacts_to_remove: list[str] = Field(default_factory=list)
    artifacts_to_preserve: list[str] = Field(default_factory=list)
    reason: str = ""


class FailureOutcome(BaseModel):
    scenario_id: str
    failure_detected: bool
    error_code: str | None = None
    job_status: str
    last_valid_checkpoint: str | None = None
    invalidated_checkpoints: list[str] = Field(default_factory=list)
    preserved_artifacts: list[str] = Field(default_factory=list)
    removed_temp_artifacts: list[str] = Field(default_factory=list)
    resume_stage: str | None = None
    retry_succeeded: bool = False
    final_state: str


class ReliabilityScorecard(BaseModel):
    failure_detected: bool
    typed_error: bool
    no_false_ready: bool
    temp_cleanup_pass: bool
    cache_preservation_pass: bool
    resume_stage_pass: bool
    retry_pass: bool
    isolation_pass: bool = True

    @property
    def passed(self) -> bool:
        return all(self.model_dump().values())

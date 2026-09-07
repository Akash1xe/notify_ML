from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

DIAGNOSTIC_SUMMARY_VERSION = "1"
OBSERVABILITY_EVENT_VERSION = "1"
ENVIRONMENT_REPORT_VERSION = "1"
CONFIG_SNAPSHOT_VERSION = "1"


class EventName(str, Enum):
    JOB_CREATED = "JOB_CREATED"
    JOB_STARTED = "JOB_STARTED"
    JOB_RESUMED = "JOB_RESUMED"
    JOB_CANCELLED = "JOB_CANCELLED"
    JOB_FAILED = "JOB_FAILED"
    JOB_COMPLETED = "JOB_COMPLETED"
    STAGE_STARTED = "STAGE_STARTED"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    STAGE_FAILED = "STAGE_FAILED"
    CACHE_HIT = "CACHE_HIT"
    CACHE_MISS = "CACHE_MISS"
    CACHE_STALE = "CACHE_STALE"
    CACHE_CORRUPT = "CACHE_CORRUPT"
    ARTIFACT_VALIDATED = "ARTIFACT_VALIDATED"
    ARTIFACT_REBUILT = "ARTIFACT_REBUILT"
    RESOURCE_WARNING = "RESOURCE_WARNING"
    PDF_READY = "PDF_READY"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"


class ObservabilityEvent(BaseModel):
    version: str = OBSERVABILITY_EVENT_VERSION
    event_name: EventName
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: str = "INFO"
    job_id: str | None = None
    phase: str | None = None
    stage: str | None = None
    status: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    cache_hit: bool | None = None
    cache_state: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    resource_metrics: dict[str, float | int | None] = Field(default_factory=dict)
    error_code: str | None = None
    retryability: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobDiagnosticSummary(BaseModel):
    version: str = DIAGNOSTIC_SUMMARY_VERSION
    job_id: str
    job_status: str
    last_checkpoint: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    pipeline_version: str
    total_runtime_seconds: float | None = Field(default=None, ge=0)
    phase_summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cache_summary: dict[str, int] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    artifact_summary: dict[str, Any] = Field(default_factory=dict)
    resource_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    failure_summary: dict[str, Any] | None = None
    retryability: str | None = None
    document_summary: dict[str, Any] = Field(default_factory=dict)


class EnvironmentCheckStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class EnvironmentCheckSeverity(str, Enum):
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    FEATURE_DEPENDENT = "FEATURE_DEPENDENT"


class EnvironmentCheckResult(BaseModel):
    name: str
    status: EnvironmentCheckStatus
    severity: EnvironmentCheckSeverity
    message: str
    details_safe: dict[str, Any] = Field(default_factory=dict)


class SystemReadinessReport(BaseModel):
    version: str = ENVIRONMENT_REPORT_VERSION
    ready: bool
    checks: list[EnvironmentCheckResult]
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


class ConfigSnapshot(BaseModel):
    version: str = CONFIG_SNAPSHOT_VERSION
    application_version: str
    processor_mode: str
    app_env: str
    log_level: str
    resource_limits: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    document: dict[str, Any] = Field(default_factory=dict)
    config_fingerprint: str

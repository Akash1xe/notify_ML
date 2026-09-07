from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.models import stable_fingerprint

RELEASE_VALIDATION_VERSION = "1"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


class ReleaseReadiness(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"


class ReleaseGate(BaseModel):
    gate_id: str
    name: str
    required: bool = True
    status: GateStatus
    duration_seconds: float = Field(default=0.0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None


class ReleaseValidationReport(BaseModel):
    version: str = RELEASE_VALIDATION_VERSION
    release_version: str
    source_revision: str | None = None
    pipeline_fingerprint: str
    production_config_fingerprint: str
    gates: list[ReleaseGate]
    warnings: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    release_ready: ReleaseReadiness

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "version": self.version,
                "release_version": self.release_version,
                "pipeline_fingerprint": self.pipeline_fingerprint,
                "production_config_fingerprint": self.production_config_fingerprint,
                "gates": [
                    {
                        "gate_id": gate.gate_id,
                        "required": gate.required,
                        "status": gate.status,
                    }
                    for gate in self.gates
                ],
            }
        )


class ReleaseManifest(BaseModel):
    version: str
    source_revision: str | None = None
    backend_version: str
    frontend_version: str
    quality_report: str | None = None
    performance_report: str | None = None
    stress_report: str | None = None
    reliability_report: str | None = None
    production_config_fingerprint: str
    release_report_sha256: str
    release_fingerprint: str

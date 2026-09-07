from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .models import GateStatus, ReleaseGate, ReleaseReadiness, ReleaseValidationReport


class ReleaseValidationRunner:
    def __init__(self, *, release_version: str, pipeline_fingerprint: str, production_config_fingerprint: str, source_revision: str | None = None) -> None:
        self.release_version = release_version
        self.pipeline_fingerprint = pipeline_fingerprint
        self.production_config_fingerprint = production_config_fingerprint
        self.source_revision = source_revision
        self.gates: list[ReleaseGate] = []

    def add_gate(self, gate_id: str, name: str, func: Callable[[], tuple[bool, dict]], *, required: bool = True) -> ReleaseGate:
        started = time.perf_counter()
        failure_reason: str | None = None
        details: dict = {}
        try:
            ok, details = func()
            status = GateStatus.PASS if ok else GateStatus.FAIL if required else GateStatus.WARNING
        except Exception as exc:  # release runner records, rather than hiding, gate failures.
            ok = False
            status = GateStatus.FAIL if required else GateStatus.WARNING
            failure_reason = f"{type(exc).__name__}: {exc}"
        gate = ReleaseGate(gate_id=gate_id, name=name, required=required, status=status, duration_seconds=round(time.perf_counter() - started, 6), details=details, failure_reason=failure_reason)
        self.gates.append(gate)
        return gate

    def run_command_gate(self, gate_id: str, name: str, command: list[str], *, cwd: Path | None = None, required: bool = True, timeout: int = 600) -> ReleaseGate:
        def run() -> tuple[bool, dict]:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
            return result.returncode == 0, {"returncode": result.returncode, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]}
        return self.add_gate(gate_id, name, run, required=required)

    def finalize(self, *, known_limitations: list[str] | None = None) -> ReleaseValidationReport:
        required_failed = any(g.required and g.status is GateStatus.FAIL for g in self.gates)
        warnings = [g.gate_id for g in self.gates if g.status is GateStatus.WARNING]
        if required_failed:
            readiness = ReleaseReadiness.NOT_READY
        elif warnings:
            readiness = ReleaseReadiness.READY_WITH_WARNINGS
        else:
            readiness = ReleaseReadiness.READY
        return ReleaseValidationReport(release_version=self.release_version, source_revision=self.source_revision, pipeline_fingerprint=self.pipeline_fingerprint, production_config_fingerprint=self.production_config_fingerprint, gates=self.gates, warnings=warnings, known_limitations=known_limitations or [], release_ready=readiness)

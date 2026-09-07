from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .models import (
    ConfigSnapshot,
    EnvironmentCheckResult,
    EnvironmentCheckSeverity,
    EnvironmentCheckStatus,
    SystemReadinessReport,
)
from app.evaluation.models import stable_fingerprint


class SensitiveValueRedactor:
    SENSITIVE_KEYS = ("password", "secret", "token", "authorization", "cookie", "api_key", "private_key")

    @classmethod
    def redact(cls, value: Any, *, key: str = "") -> Any:
        if any(part in key.lower() for part in cls.SENSITIVE_KEYS):
            return "***REDACTED***"
        if isinstance(value, Mapping):
            return {str(k): cls.redact(v, key=str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.redact(item, key=key) for item in value]
        return value


class EnvironmentValidator:
    def __init__(self, settings: Any | None = None, *, min_free_disk_bytes: int = 2 * 1024**3) -> None:
        self.settings = settings
        self.min_free_disk_bytes = min_free_disk_bytes

    @staticmethod
    def _tool_check(name: str, explicit: Path | str | None = None) -> EnvironmentCheckResult:
        executable = str(explicit) if explicit else shutil.which(name)
        if not executable:
            return EnvironmentCheckResult(name=name, status=EnvironmentCheckStatus.FAIL, severity=EnvironmentCheckSeverity.REQUIRED, message=f"{name} was not found")
        try:
            result = subprocess.run([executable, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
            ok = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        return EnvironmentCheckResult(name=name, status=EnvironmentCheckStatus.PASS if ok else EnvironmentCheckStatus.FAIL, severity=EnvironmentCheckSeverity.REQUIRED, message=f"{name} available" if ok else f"{name} could not be executed")

    def _storage_check(self) -> tuple[EnvironmentCheckResult, EnvironmentCheckResult]:
        root = Path(getattr(self.settings, "storage_root", "storage/jobs"))
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".notify-write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            writable = True
        except OSError:
            writable = False
        storage = EnvironmentCheckResult(name="storage", status=EnvironmentCheckStatus.PASS if writable else EnvironmentCheckStatus.FAIL, severity=EnvironmentCheckSeverity.REQUIRED, message="storage writable" if writable else "storage is not writable")
        try:
            free = shutil.disk_usage(root).free
            if free < self.min_free_disk_bytes:
                disk = EnvironmentCheckResult(name="free_disk", status=EnvironmentCheckStatus.WARNING, severity=EnvironmentCheckSeverity.REQUIRED, message="available disk space is below recommended minimum", details_safe={"free_bytes": free, "recommended_minimum_bytes": self.min_free_disk_bytes})
            else:
                disk = EnvironmentCheckResult(name="free_disk", status=EnvironmentCheckStatus.PASS, severity=EnvironmentCheckSeverity.REQUIRED, message="disk space available", details_safe={"free_bytes": free})
        except OSError:
            disk = EnvironmentCheckResult(name="free_disk", status=EnvironmentCheckStatus.WARNING, severity=EnvironmentCheckSeverity.REQUIRED, message="disk space could not be inspected")
        return storage, disk

    def validate(self, *, deep: bool = False) -> SystemReadinessReport:
        checks: list[EnvironmentCheckResult] = []
        py_ok = sys.version_info >= (3, 11)
        checks.append(EnvironmentCheckResult(name="python", status=EnvironmentCheckStatus.PASS if py_ok else EnvironmentCheckStatus.FAIL, severity=EnvironmentCheckSeverity.REQUIRED, message=f"Python {sys.version_info.major}.{sys.version_info.minor}"))
        checks.append(self._tool_check("ffmpeg", getattr(self.settings, "ffmpeg_path", None)))
        checks.append(self._tool_check("ffprobe", getattr(self.settings, "ffprobe_path", None)))
        checks.extend(self._storage_check())
        for module_name, label in (("faster_whisper", "whisper_runtime"), ("cv2", "opencv"), ("reportlab", "reportlab"), ("pypdf", "pypdf")):
            present = importlib.util.find_spec(module_name) is not None
            severity = EnvironmentCheckSeverity.FEATURE_DEPENDENT if module_name == "faster_whisper" else EnvironmentCheckSeverity.REQUIRED
            checks.append(EnvironmentCheckResult(name=label, status=EnvironmentCheckStatus.PASS if present else EnvironmentCheckStatus.WARNING if severity is EnvironmentCheckSeverity.FEATURE_DEPENDENT else EnvironmentCheckStatus.FAIL, severity=severity, message=f"{label} {'available' if present else 'unavailable'}"))
        if deep:
            transformers = importlib.util.find_spec("transformers") is not None
            checks.append(EnvironmentCheckResult(name="qwen_runtime", status=EnvironmentCheckStatus.PASS if transformers else EnvironmentCheckStatus.WARNING, severity=EnvironmentCheckSeverity.FEATURE_DEPENDENT, message="transformers runtime available" if transformers else "transformers runtime unavailable"))
        failures = [c.name for c in checks if c.status is EnvironmentCheckStatus.FAIL and c.severity is EnvironmentCheckSeverity.REQUIRED]
        warnings = [c.name for c in checks if c.status is EnvironmentCheckStatus.WARNING]
        return SystemReadinessReport(ready=not failures, checks=checks, warnings=warnings, failures=failures)

    def config_snapshot(self, *, application_version: str) -> ConfigSnapshot:
        settings = self.settings
        payload = {
            "processor_mode": getattr(settings, "processor_mode", "unknown"),
            "app_env": getattr(settings, "app_env", "unknown"),
            "log_level": getattr(settings, "log_level", "INFO"),
            "resource_limits": {
                "max_concurrent_jobs": getattr(settings, "max_concurrent_jobs", None),
                "max_sampled_frames": getattr(settings, "max_sampled_frames", None),
                "max_total_candidates": getattr(settings, "max_total_candidates", None),
                "max_video_download_gb": getattr(settings, "max_video_download_gb", None),
            },
            "models": {
                "whisper_model_size": getattr(settings, "whisper_model_size", None),
                "qwen_vl_model_tier": getattr(settings, "qwen_vl_model_tier", None),
            },
            "document": {
                "page_size": getattr(settings, "pdf_page_size", None),
            },
        }
        redacted = SensitiveValueRedactor.redact(payload)
        return ConfigSnapshot(application_version=application_version, config_fingerprint=stable_fingerprint(redacted), **redacted)

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import JobDiagnosticSummary, ObservabilityEvent
from .environment import SensitiveValueRedactor


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class JsonLineEventSink:
    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, event: ObservabilityEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = SensitiveValueRedactor.redact(event.model_dump(mode="json", exclude_none=True))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        events: list[dict] = []
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
        return events


class JobSummaryBuilder:
    @staticmethod
    def save(path: Path, summary: JobDiagnosticSummary) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, SensitiveValueRedactor.redact(summary.model_dump(mode="json", exclude_none=True)))


class DiagnosticExporter:
    @staticmethod
    def export(path: Path, *, summary: JobDiagnosticSummary, config_snapshot: dict | None = None) -> None:
        payload = {
            "summary": SensitiveValueRedactor.redact(summary.model_dump(mode="json", exclude_none=True)),
            "config": SensitiveValueRedactor.redact(config_snapshot or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)

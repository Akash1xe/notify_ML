from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.storage.workspace import WorkspaceManager


class ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "job_id"):
            record.job_id = "-"
        if not hasattr(record, "stage"):
            record.stage = "-"
        return super().format(record)


class JsonContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "job_id": getattr(record, "job_id", None),
            "stage": getattr(record, "stage", None),
            "message": record.getMessage(),
        }
        return json.dumps({k: v for k, v in payload.items() if v is not None}, ensure_ascii=False, sort_keys=True)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    log_format = os.getenv("LOG_FORMAT", "text").strip().lower()
    if log_format == "json":
        handler.setFormatter(JsonContextFormatter())
    else:
        handler.setFormatter(
            ContextFormatter(
                "%(asctime)s %(levelname)s job=%(job_id)s stage=%(stage)s %(name)s: %(message)s"
            )
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def log_job(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    job_id: str,
    stage: str,
    exc_info: bool = False,
) -> None:
    logger.log(
        level,
        message,
        extra={"job_id": job_id, "stage": stage},
        exc_info=exc_info,
    )


class JobEventLogger:
    def __init__(self, workspace_manager: "WorkspaceManager") -> None:
        self._workspace_manager = workspace_manager
        self._lock = RLock()

    def write(self, job_id: str, *, level: str, stage: str, message: str) -> None:
        log_dir = self._workspace_manager.logs_dir(job_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level.upper(),
            "job_id": job_id,
            "stage": stage,
            "message": message,
        }
        try:
            with self._lock, (log_dir / "events.ndjson").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            logging.getLogger(__name__).exception("Unable to append per-job event log")

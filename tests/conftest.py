from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        storage_root=tmp_path / "jobs",
        max_concurrent_jobs=2,
        fake_processor_step_delay=0.002,
        log_level="CRITICAL",
    )


@pytest.fixture
def client(settings: AppSettings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def wait_for_status(client: TestClient, job_id: str, expected: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        latest = response.json()
        if latest["status"] == expected:
            return latest
        time.sleep(0.005)
    raise AssertionError(f"Job {job_id} did not reach {expected}; latest={latest}")

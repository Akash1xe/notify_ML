import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.main import create_app
from tests.conftest import wait_for_status


def test_job_api_end_to_end(client: TestClient):
    created = client.post(
        "/api/jobs", json={"source_url": "https://youtube.com/watch?v=abcdef12345"}
    )
    assert created.status_code == 201
    job_id = created.json()["id"]
    final = wait_for_status(client, job_id, "COMPLETED")
    assert final["progress"] == 100
    assert final["stage"] == "COMPLETED"
    listing = client.get("/api/jobs")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1


def test_invalid_url_returns_422(client: TestClient):
    response = client.post("/api/jobs", json={"source_url": "not-a-url"})
    assert response.status_code == 422


def test_unknown_job_returns_404(client: TestClient):
    response = client.get("/api/jobs/00000000-0000-0000-0000-000000000001")
    assert response.status_code == 404
    assert response.json()["error"] == "job_not_found"


def test_running_job_can_be_cancelled(tmp_path: Path):
    settings = AppSettings(
        storage_root=tmp_path / "jobs",
        processor_mode="fake",
        fake_processor_step_delay=0.1,
        max_concurrent_jobs=1,
        log_level="CRITICAL",
    )
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/jobs", json={"source_url": "https://youtube.com/watch?v=abcdef12345"}
        )
        job_id = created.json()["id"]
        wait_for_status(client, job_id, "RUNNING")
        cancelled = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "CANCELLED"
        time.sleep(0.15)
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "CANCELLED"


def test_processor_failure_is_isolated(tmp_path: Path):
    settings = AppSettings(
        storage_root=tmp_path / "jobs",
        processor_mode="fake",
        fake_processor_step_delay=0,
        simulated_failure_at_progress=40,
        max_concurrent_jobs=1,
        log_level="CRITICAL",
    )
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/jobs", json={"source_url": "https://youtube.com/watch?v=abcdef12345"}
        )
        final = wait_for_status(client, created.json()["id"], "FAILED")
        assert final["progress"] == 40
        assert final["error"]["code"] == "processing_error"
        assert client.get("/health").status_code == 200


def test_concurrency_limit_is_respected(tmp_path: Path):
    settings = AppSettings(
        storage_root=tmp_path / "jobs",
        processor_mode="fake",
        fake_processor_step_delay=0.03,
        max_concurrent_jobs=1,
        log_level="CRITICAL",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        ids = [
            client.post(
                "/api/jobs", json={"source_url": f"https://youtube.com/watch?v=abcde1234{i}Z"}
            ).json()["id"]
            for i in range(3)
        ]
        for job_id in ids:
            wait_for_status(client, job_id, "COMPLETED", timeout=3)
        assert app.state.job_runner.max_observed_concurrency == 1

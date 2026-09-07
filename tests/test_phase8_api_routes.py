from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.main import create_app


def test_document_status_route_is_read_only_for_new_job(tmp_path):
    settings = AppSettings(storage_root=tmp_path / "jobs", processor_mode="fake", fake_processor_step_delay=0, log_level="CRITICAL")
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/jobs", json={"source_url": "https://www.youtube.com/watch?v=abc"}).json()
        response = client.get(f"/api/jobs/{created['id']}/document")
        assert response.status_code == 200
        assert response.json()["status"] in {"PROCESSING", "NOT_STARTED"}

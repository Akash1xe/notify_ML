from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.main import create_app


def test_settings_defaults_and_normalization(tmp_path: Path):
    settings = AppSettings(storage_root=tmp_path / "jobs", log_level="debug")
    assert settings.app_name == "Notify"
    assert settings.log_level == "DEBUG"
    assert settings.port == 8000


def test_health_and_root(client: TestClient):
    assert client.get("/health").json() == {"status": "ok", "service": "notify"}
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["phase"] == "1-foundation"


def test_startup_creates_storage_root(tmp_path: Path):
    root = tmp_path / "missing" / "jobs"
    settings = AppSettings(storage_root=root, fake_processor_step_delay=0)
    assert not root.exists()
    with TestClient(create_app(settings)):
        assert root.exists()

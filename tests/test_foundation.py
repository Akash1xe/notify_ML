from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.main import create_app
from app.version import APP_VERSION


def test_settings_defaults_and_normalization(tmp_path: Path):
    settings = AppSettings(storage_root=tmp_path / "jobs", log_level="debug")
    assert settings.app_name == "Notify"
    assert settings.log_level == "DEBUG"
    assert settings.port == 8000
    assert settings.processor_mode == "document"
    assert settings.video_max_height == 1080


def test_health_and_root(client: TestClient):
    assert client.get("/health").json() == {
        "status": "ok",
        "service": "notify",
        "version": APP_VERSION,
    }
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["version"] == APP_VERSION
    assert root.json()["phase"] == "9-release-hardening"
    assert root.json()["current_phase"] == "9-release-hardening"


def test_startup_creates_storage_root(tmp_path: Path):
    root = tmp_path / "missing" / "jobs"
    settings = AppSettings(storage_root=root, processor_mode="fake", fake_processor_step_delay=0)
    assert not root.exists()
    with TestClient(create_app(settings)):
        assert root.exists()


def test_empty_ffmpeg_env_paths_are_none(tmp_path: Path):
    settings = AppSettings(storage_root=tmp_path / "jobs", ffmpeg_path="", ffprobe_path="")
    assert settings.ffmpeg_path is None
    assert settings.ffprobe_path is None

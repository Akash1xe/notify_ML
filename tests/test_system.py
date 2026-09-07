from fastapi.testclient import TestClient

from app.system.hardware import ProcessingProfile, get_system_capabilities


def test_hardware_detection_degrades_gracefully():
    caps = get_system_capabilities()
    assert caps.logical_cpu_count >= 1
    assert caps.memory_gb > 0
    assert caps.recommended_profile in {
        ProcessingProfile.LOW,
        ProcessingProfile.BALANCED,
        ProcessingProfile.QUALITY,
    }


def test_capabilities_endpoint(client: TestClient):
    response = client.get("/api/system/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert "memory_gb" in body
    assert "recommended_profile" in body
    assert "media_tools" in body


def test_vlm_runtime_endpoint_does_not_load_model(tmp_path):
    from app.core.config import AppSettings
    from app.main import create_app
    from fastapi.testclient import TestClient

    settings = AppSettings(storage_root=tmp_path / "jobs", processor_mode="fake", fake_processor_step_delay=0)
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/system/vlm")
        assert response.status_code == 200
        payload = response.json()
        assert payload["configured_model_tier"] == "4b"
        assert payload["model_loaded"] is False

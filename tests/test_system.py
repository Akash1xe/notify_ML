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

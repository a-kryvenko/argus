from fastapi.testclient import TestClient
from api.main import app


def test_application_exposes_current_routes():
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert {path for path in paths if path.startswith("/observations")} == {"/observations", "/observations/history"}
    assert not any(path.startswith(("/public/", "/private/")) or "{visibility}" in path for path in paths)
    assert "/auth/token" not in paths
    assert not any(path.startswith(("/private/risk/", "/private/model/")) for path in paths)

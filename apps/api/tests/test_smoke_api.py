from fastapi.testclient import TestClient
from app.main import app


def test_application_exposes_current_routes():
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/public/observations/latest" in paths
    assert "/private/risk/outlook" in paths

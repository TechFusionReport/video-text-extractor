from fastapi.testclient import TestClient

from video_text_extractor.main import app


client = TestClient(app)


def test_health_is_public() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_requires_authentication() -> None:
    response = client.post("/v1/extract", json={"url": "https://example.com/video"})
    assert response.status_code == 401


def test_extract_rejects_invalid_token() -> None:
    response = client.post(
        "/v1/extract",
        headers={"Authorization": "Bearer wrong"},
        json={"url": "https://example.com/video"},
    )
    assert response.status_code == 401


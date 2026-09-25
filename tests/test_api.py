from fastapi.testclient import TestClient

from video_text_extractor.main import app
from video_text_extractor.extractor import is_duplicate_ocr
from video_text_extractor.models import OcrSegment


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


def test_ocr_deduplicates_near_identical_frames() -> None:
    existing = [OcrSegment(start=0, end=2, text="1 cup sweet corn", confidence=0.98)]
    assert is_duplicate_ocr("1 cup sweet corn!", existing)
    assert not is_duplicate_ocr("8 oz cream cheese", existing)

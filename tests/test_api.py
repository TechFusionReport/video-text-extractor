from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from yt_dlp.networking.impersonate import ImpersonateTarget

from video_text_extractor.config import Settings
from video_text_extractor.extractor import VideoExtractor, is_duplicate_ocr
from video_text_extractor.main import app
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


def test_yt_dlp_requests_use_browser_impersonation() -> None:
    extractor = VideoExtractor(Settings(extractor_api_key="test-secret"))

    assert extractor._yt_dlp_request_options()["impersonate"] == ImpersonateTarget(client="chrome")


def test_yt_dlp_requests_use_configured_cookie_file(tmp_path: Path) -> None:
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    extractor = VideoExtractor(Settings(
        extractor_api_key="test-secret",
        yt_dlp_cookie_file=str(cookie_file),
    ))

    assert extractor._yt_dlp_request_options()["cookiefile"] == str(cookie_file)


def test_yt_dlp_rejects_missing_cookie_file(tmp_path: Path) -> None:
    extractor = VideoExtractor(Settings(
        extractor_api_key="test-secret",
        yt_dlp_cookie_file=str(tmp_path / "missing.txt"),
    ))

    with pytest.raises(ValueError, match="YT_DLP_COOKIE_FILE does not exist"):
        extractor._yt_dlp_request_options()

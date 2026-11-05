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


# --- description, sound tags, warnings, opt-in OCR, limits, observability ---

import json  # noqa: E402

from video_text_extractor.extractor import clean_caption_text  # noqa: E402
from video_text_extractor.main import get_job_slots  # noqa: E402

AUTH = {"Authorization": "Bearer test-secret"}


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def _caption_extractor(tmp_path: Path, events: list[str], **settings) -> tuple[VideoExtractor, dict]:
    payload = {"events": [{"tStartMs": i * 1000, "dDurationMs": 1000, "segs": [{"utf8": text}]} for i, text in enumerate(events)]}
    (tmp_path / "source.en.json3").write_text(json.dumps(payload), encoding="utf-8")
    extractor = VideoExtractor(Settings(extractor_api_key="test-secret", **settings))
    info = {"id": "abc", "title": "T", "duration": 60, "description": "1 cup flour\n2 eggs", "extractor_key": "Youtube"}
    extractor._metadata = lambda url, temp_dir: info  # type: ignore[method-assign]
    extractor._read_caption = lambda i, d: VideoExtractor._read_caption(i, tmp_path)  # type: ignore[method-assign]
    return extractor, info


def test_clean_caption_text_strips_sound_tags() -> None:
    assert clean_caption_text("[Music]") == ""
    assert clean_caption_text("♪ la la ♪") == "la la"
    assert clean_caption_text("so [Applause] welcome  back") == "so welcome back"
    assert clean_caption_text("see reference [1]") == "see reference [1]"


def test_extract_returns_description_and_strips_tags(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_text_extractor.extractor.validate_public_url", lambda url: None)
    extractor, _ = _caption_extractor(tmp_path, ["[Music]", _words(40)])
    result = extractor.extract("https://example.com/v", None, False)

    assert result.method == "captions"
    assert result.description == "1 cup flour\n2 eggs"
    assert "[Music]" not in result.transcript
    assert result.warnings == []


def test_extract_warns_on_missing_description_and_short_transcript(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_text_extractor.extractor.validate_public_url", lambda url: None)
    extractor, info = _caption_extractor(tmp_path, ["hello there"])
    info["description"] = ""
    result = extractor.extract("https://example.com/v", None, False)

    assert result.warnings == ["no_description", "transcript_short"]


def test_ocr_not_run_unless_requested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_text_extractor.extractor.validate_public_url", lambda url: None)
    extractor, _ = _caption_extractor(tmp_path, [_words(40)], ocr_enabled=True)
    extractor._extract_ocr = lambda *a: pytest.fail("OCR must be opt-in")  # type: ignore[method-assign]

    assert extractor.extract("https://example.com/v", None, False).ocr_segments == []


def test_ocr_requested_but_disabled_warns(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_text_extractor.extractor.validate_public_url", lambda url: None)
    extractor, _ = _caption_extractor(tmp_path, [_words(40)], ocr_enabled=False)
    result = extractor.extract("https://example.com/v", None, False, ocr=True)

    assert result.warnings == ["ocr_unavailable"]


def test_ocr_skipped_for_long_media(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_text_extractor.extractor.validate_public_url", lambda url: None)
    extractor, _ = _caption_extractor(tmp_path, [_words(40)], ocr_enabled=True, ocr_max_media_seconds=30)
    result = extractor.extract("https://example.com/v", None, False, ocr=True)

    assert result.warnings == ["ocr_skipped_too_long"]


def test_ocr_failure_is_non_fatal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_text_extractor.extractor.validate_public_url", lambda url: None)
    extractor, _ = _caption_extractor(tmp_path, [_words(40)], ocr_enabled=True)

    def boom(*args):
        raise RuntimeError("ocr exploded")

    extractor._extract_ocr = boom  # type: ignore[method-assign]
    result = extractor.extract("https://example.com/v", None, False, ocr=True)

    assert result.transcript  # transcript survives
    assert result.warnings == ["ocr_failed"]


def test_whisper_fallback_respects_duration_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_text_extractor.extractor.validate_public_url", lambda url: None)
    extractor = VideoExtractor(Settings(extractor_api_key="test-secret", whisper_max_media_seconds=60))
    extractor._metadata = lambda url, temp_dir: {"id": "x", "duration": 600}  # type: ignore[method-assign]
    extractor._read_caption = lambda info, directory: None  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="transcription limit"):
        extractor.extract("https://example.com/v", None, False)


def test_info_requires_authentication_and_reports_runtime() -> None:
    assert client.get("/v1/info").status_code == 401
    response = client.get("/v1/info", headers=AUTH)
    body = response.json()

    assert response.status_code == 200
    assert body["limits"]["max_concurrent_jobs"] == 2
    assert set(body["runtime"]) == {"deno", "ffmpeg", "cache_dir_writable"}
    assert "test-secret" not in response.text


def test_unexpected_error_is_logged_and_generic(monkeypatch, caplog) -> None:
    def boom(self, *args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(VideoExtractor, "extract", boom)
    with caplog.at_level("ERROR", logger="video_text_extractor"):
        response = client.post("/v1/extract", headers=AUTH, json={"url": "https://example.com/v"})

    assert response.status_code == 502
    assert "secret internal detail" not in response.text
    assert "ref " in response.json()["detail"]
    assert "RuntimeError" in caplog.text


def test_busy_extractor_returns_503_with_retry_after(monkeypatch) -> None:
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")
    monkeypatch.setenv("QUEUE_TIMEOUT_SECONDS", "1")
    from video_text_extractor.config import get_settings

    get_settings.cache_clear()
    get_job_slots.cache_clear()
    slots = get_job_slots()
    assert slots.acquire(timeout=1)  # occupy the only slot
    try:
        response = client.post("/v1/extract", headers=AUTH, json={"url": "https://example.com/v"})
    finally:
        slots.release()
        get_settings.cache_clear()
        get_job_slots.cache_clear()

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"

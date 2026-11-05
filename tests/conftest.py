import os

os.environ.setdefault("EXTRACTOR_API_KEY", "test-secret")


import pytest


@pytest.fixture(autouse=True)
def isolated_subtitle_breaker(monkeypatch):
    from video_text_extractor import extractor, main
    from video_text_extractor.subtitle_breaker import SubtitleCircuitBreaker

    breaker = SubtitleCircuitBreaker()
    monkeypatch.setattr(extractor, "subtitle_breaker", breaker)
    monkeypatch.setattr(main, "subtitle_breaker", breaker)
    return breaker

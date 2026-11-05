import json
from pathlib import Path

from video_text_extractor.config import Settings
from video_text_extractor.extractor import VideoExtractor


def _extractor(tmp_path: Path, events: list[str]) -> VideoExtractor:
    payload = {"events": [{"tStartMs": i * 1000, "dDurationMs": 1000, "segs": [{"utf8": e}]} for i, e in enumerate(events)]}
    (tmp_path / "source.en.json3").write_text(json.dumps(payload), encoding="utf-8")
    extractor = VideoExtractor(Settings(extractor_api_key="test-secret"))
    extractor._metadata = lambda url, temp_dir: {"id": "x", "duration": 60, "description": "d"}  # type: ignore[method-assign]
    extractor._read_caption = lambda info, directory: VideoExtractor._read_caption(info, tmp_path)  # type: ignore[method-assign]
    extractor._download_audio = lambda *a: (_ for _ in ()).throw(AssertionError("Whisper must not run"))  # type: ignore[method-assign]
    return extractor


def test_tag_only_captions_return_empty_transcript_with_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_text_extractor.extractor.validate_public_url", lambda url: None)
    result = _extractor(tmp_path, ["[Music]", "[Applause]", "♪"]).extract("https://example.com/v", None, False)

    assert result.method == "captions"
    assert result.transcript == ""
    assert result.warnings == ["captions_only_sound_tags", "transcript_short"]


def test_caption_file_without_events_still_falls_back(tmp_path: Path) -> None:
    (tmp_path / "source.en.json3").write_text(json.dumps({"events": []}), encoding="utf-8")

    assert VideoExtractor._read_caption({}, tmp_path) is None

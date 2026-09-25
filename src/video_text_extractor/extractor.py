import json
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

from .config import Settings
from .models import ExtractResponse, TranscriptSegment
from .security import validate_public_url


class VideoExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: WhisperModel | None = None

    def extract(self, url: str, language: str | None, force_transcription: bool) -> ExtractResponse:
        validate_public_url(url)
        with tempfile.TemporaryDirectory(prefix="tfr-extract-") as temp_dir:
            info = self._metadata(url, temp_dir)
            self._validate_duration(info)
            if not force_transcription:
                caption = self._read_caption(info, Path(temp_dir))
                if caption:
                    segments, caption_language = caption
                    return self._response(url, info, "captions", segments, caption_language)

            media_path = self._download_audio(url, temp_dir)
            segments, detected_language = self._transcribe(media_path, language)
            return self._response(url, info, "whisper", segments, detected_language)

    def _metadata(self, url: str, temp_dir: str) -> dict:
        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "json3",
            "subtitleslangs": ["en", "en-orig"],
            "outtmpl": str(Path(temp_dir) / "source.%(ext)s"),
        }
        with YoutubeDL(options) as ydl:
            return ydl.extract_info(url, download=True)

    def _validate_duration(self, info: dict) -> None:
        duration = info.get("duration")
        if duration and float(duration) > self.settings.max_media_seconds:
            raise ValueError(f"Media exceeds the {self.settings.max_media_seconds}-second limit")

    @staticmethod
    def _read_caption(info: dict, directory: Path) -> tuple[list[TranscriptSegment], str | None] | None:
        files = sorted(directory.glob("source*.json3"))
        if not files:
            return None
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        segments: list[TranscriptSegment] = []
        for event in payload.get("events", []):
            text = "".join(part.get("utf8", "") for part in event.get("segs", [])).strip()
            if not text or text == "\n":
                continue
            start = event.get("tStartMs", 0) / 1000
            duration = event.get("dDurationMs", 0) / 1000
            segments.append(TranscriptSegment(start=start, end=start + duration, text=text))
        return (segments, info.get("language")) if segments else None

    def _download_audio(self, url: str, temp_dir: str) -> Path:
        output = str(Path(temp_dir) / "audio.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "outtmpl": output,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        }
        with YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
        path = Path(temp_dir) / "audio.wav"
        if not path.exists():
            raise RuntimeError("Audio download did not produce a usable file")
        return path

    def _transcribe(self, path: Path, language: str | None) -> tuple[list[TranscriptSegment], str | None]:
        if self._model is None:
            self._model = WhisperModel(
                self.settings.whisper_model,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
            )
        generated, info = self._model.transcribe(str(path), language=language, vad_filter=True)
        segments = [
            TranscriptSegment(start=item.start, end=item.end, text=item.text.strip())
            for item in generated
            if item.text.strip()
        ]
        return segments, info.language

    @staticmethod
    def _response(url: str, info: dict, method: str, segments: list[TranscriptSegment], language: str | None) -> ExtractResponse:
        return ExtractResponse(
            source_url=url,
            platform=info.get("extractor_key") or info.get("extractor") or "unknown",
            source_id=info.get("id"),
            title=info.get("title"),
            author=info.get("uploader") or info.get("channel"),
            duration_seconds=info.get("duration"),
            language=language,
            method=method,  # type: ignore[arg-type]
            transcript=" ".join(segment.text for segment in segments),
            segments=segments,
        )


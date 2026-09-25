import json
import re
import subprocess
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

from .config import Settings
from .models import ExtractResponse, OcrSegment, TranscriptSegment
from .security import validate_public_url


class VideoExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: WhisperModel | None = None
        self._ocr: Any | None = None

    def extract(self, url: str, language: str | None, force_transcription: bool) -> ExtractResponse:
        validate_public_url(url)
        with tempfile.TemporaryDirectory(prefix="tfr-extract-") as temp_dir:
            info = self._metadata(url, temp_dir)
            self._validate_duration(info)
            if not force_transcription:
                caption = self._read_caption(info, Path(temp_dir))
                if caption:
                    segments, caption_language = caption
                    ocr_segments = self._extract_ocr(url, info, Path(temp_dir))
                    return self._response(url, info, "captions", segments, caption_language, ocr_segments)

            media_path = self._download_audio(url, temp_dir)
            segments, detected_language = self._transcribe(media_path, language)
            ocr_segments = self._extract_ocr(url, info, Path(temp_dir))
            return self._response(url, info, "whisper", segments, detected_language, ocr_segments)

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

    def _extract_ocr(self, url: str, info: dict, directory: Path) -> list[OcrSegment]:
        if not self.settings.ocr_enabled:
            return []

        duration = float(info.get("duration") or 0)
        interval = max(
            self.settings.ocr_frame_interval_seconds,
            duration / max(self.settings.ocr_max_frames, 1),
        )
        video_path = self._download_video(url, directory)
        frames_dir = directory / "frames"
        frames_dir.mkdir()
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path), "-vf", f"fps=1/{interval}",
                "-frames:v", str(self.settings.ocr_max_frames),
                "-q:v", "3", str(frames_dir / "frame-%06d.jpg"),
            ],
            check=True,
            timeout=self.settings.job_timeout_seconds,
        )

        engine = self._get_ocr_engine()
        found: list[OcrSegment] = []
        for index, frame in enumerate(sorted(frames_dir.glob("frame-*.jpg"))):
            result, _ = engine(str(frame))
            if not result:
                continue
            lines = [
                (clean_ocr_text(str(item[1])), float(item[2]))
                for item in result
                if len(item) >= 3 and float(item[2]) >= self.settings.ocr_min_confidence
            ]
            text = " | ".join(value for value, _ in lines if value)
            if not text or is_duplicate_ocr(text, found):
                continue
            confidence = sum(score for value, score in lines if value) / max(
                sum(1 for value, _ in lines if value), 1
            )
            start = index * interval
            found.append(OcrSegment(
                start=start,
                end=min(start + interval, duration) if duration else start + interval,
                text=text,
                confidence=round(confidence, 4),
            ))
        return found

    def _get_ocr_engine(self) -> Any:
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        return self._ocr

    @staticmethod
    def _download_video(url: str, directory: Path) -> Path:
        output = str(directory / "ocr-video.%(ext)s")
        options = {
            "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "outtmpl": output,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }
        with YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
        candidates = [path for path in directory.glob("ocr-video.*") if path.suffix not in {".part", ".ytdl"}]
        if not candidates:
            raise RuntimeError("Video download did not produce a usable file for OCR")
        return max(candidates, key=lambda path: path.stat().st_size)

    @staticmethod
    def _response(
        url: str,
        info: dict,
        method: str,
        segments: list[TranscriptSegment],
        language: str | None,
        ocr_segments: list[OcrSegment],
    ) -> ExtractResponse:
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
            ocr_text="\n".join(segment.text for segment in ocr_segments),
            ocr_segments=ocr_segments,
        )


def clean_ocr_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" |\n\t")


def normalize_ocr_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def is_duplicate_ocr(text: str, existing: list[OcrSegment]) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return True
    for segment in existing[-4:]:
        previous = normalize_ocr_text(segment.text)
        if normalized == previous or SequenceMatcher(None, normalized, previous).ratio() >= 0.9:
            return True
    return False

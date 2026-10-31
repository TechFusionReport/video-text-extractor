from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class ExtractRequest(BaseModel):
    url: AnyHttpUrl
    language: str | None = Field(default=None, max_length=20)
    force_transcription: bool = False
    # OCR is opt-in per request: it downloads the full video and is slow.
    ocr: bool = False


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class OcrSegment(BaseModel):
    start: float
    end: float
    text: str
    confidence: float


class ExtractResponse(BaseModel):
    source_url: str
    platform: str
    source_id: str | None = None
    title: str | None = None
    author: str | None = None
    duration_seconds: float | None = None
    language: str | None = None
    method: Literal["captions", "whisper"]
    transcript: str
    segments: list[TranscriptSegment]
    ocr_text: str = ""
    ocr_segments: list[OcrSegment] = Field(default_factory=list)
    # Uploader description; for recipe videos it often holds the full recipe.
    description: str = ""
    # Machine-readable notes about degraded or skipped steps (e.g. ocr_failed).
    warnings: list[str] = Field(default_factory=list)

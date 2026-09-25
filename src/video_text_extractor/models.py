from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class ExtractRequest(BaseModel):
    url: AnyHttpUrl
    language: str | None = Field(default=None, max_length=20)
    force_transcription: bool = False


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


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


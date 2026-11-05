from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    extractor_api_key: str
    whisper_model: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    max_media_seconds: int = 3600
    job_timeout_seconds: int = 1800
    yt_dlp_impersonate_target: str | None = "chrome"
    yt_dlp_cookie_file: str | None = None
    # Route yt-dlp requests through the configured proxy.
    # No direct fallback if the proxy is unavailable.
    yt_dlp_proxy: str | None = None

    subtitle_breaker_seconds: int = Field(default=1800, ge=0, le=86400)

    # OCR requires both this switch and ocr=true in the request.
    ocr_enabled: bool = False
    ocr_max_media_seconds: int = 600
    ocr_frame_interval_seconds: float = 2.0
    ocr_max_frames: int = 180
    ocr_min_confidence: float = 0.55
    # Whisper on CPU is slow; cap the media length it will accept (seconds).
    whisper_max_media_seconds: int = 1800
    # Simultaneous extraction jobs; extra requests wait, then get 503.
    max_concurrent_jobs: int = 2
    queue_timeout_seconds: int = 30
    # Set at image build time (--build-arg GIT_SHA) so /v1/info shows what is running.
    git_sha: str = "unknown"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

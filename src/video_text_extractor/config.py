from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    extractor_api_key: str
    whisper_model: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    max_media_seconds: int = 3600
    job_timeout_seconds: int = 1800
    ocr_enabled: bool = True
    ocr_frame_interval_seconds: float = 2.0
    ocr_max_frames: int = 180
    ocr_min_confidence: float = 0.55


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

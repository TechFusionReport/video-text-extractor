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


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


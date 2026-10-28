import asyncio
import logging
import os
import shutil
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException

from .config import Settings, get_settings
from .extractor import VideoExtractor
from .models import ExtractRequest, ExtractResponse
from .security import require_api_key

logger = logging.getLogger("video_text_extractor")

app = FastAPI(title="TFR Video Text Extractor", version="0.2.0")


class QueueTimeout(Exception):
    """Raised when every job slot stayed busy for the whole queue timeout."""


@lru_cache
def get_extractor() -> VideoExtractor:
    return VideoExtractor(get_settings())


@lru_cache
def get_job_slots() -> threading.BoundedSemaphore:
    # A threading semaphore (not asyncio) so it works from the worker threads
    # and does not depend on which event loop created it.
    return threading.BoundedSemaphore(max(get_settings().max_concurrent_jobs, 1))


def run_job(request: ExtractRequest) -> ExtractResponse:
    """Run one extraction inside a concurrency slot (executes in a worker thread)."""
    settings = get_settings()
    slots = get_job_slots()
    if not slots.acquire(timeout=settings.queue_timeout_seconds):
        raise QueueTimeout
    try:
        return get_extractor().extract(
            str(request.url), request.language, request.force_transcription, request.ocr
        )
    finally:
        slots.release()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _cache_writable() -> bool:
    cache = Path.home() / ".cache"
    return cache.is_dir() and os.access(cache, os.W_OK)


@app.get("/v1/info", dependencies=[Depends(require_api_key)])
def info() -> dict:
    """Authenticated runtime facts, so a deploy can be verified without shell access."""
    settings: Settings = get_settings()
    return {
        "version": app.version,
        "git_sha": settings.git_sha,
        "whisper_model": settings.whisper_model,
        "ocr_enabled": settings.ocr_enabled,
        "limits": {
            "max_media_seconds": settings.max_media_seconds,
            "whisper_max_media_seconds": settings.whisper_max_media_seconds,
            "ocr_max_media_seconds": settings.ocr_max_media_seconds,
            "job_timeout_seconds": settings.job_timeout_seconds,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "queue_timeout_seconds": settings.queue_timeout_seconds,
        },
        "yt_dlp": {
            "impersonate_target": settings.yt_dlp_impersonate_target,
            "cookie_file_configured": bool(settings.yt_dlp_cookie_file),
        },
        "runtime": {
            "deno": shutil.which("deno") is not None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "cache_dir_writable": _cache_writable(),
        },
    }


@app.post("/v1/extract", response_model=ExtractResponse, dependencies=[Depends(require_api_key)])
async def extract(request: ExtractRequest) -> ExtractResponse:
    settings = get_settings()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(run_job, request),
            timeout=settings.job_timeout_seconds,
        )
    except QueueTimeout as exc:
        raise HTTPException(
            status_code=503,
            detail="Extractor is busy; retry shortly",
            headers={"Retry-After": str(settings.queue_timeout_seconds)},
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Extraction timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # The client only gets a generic message plus a reference; the real error stays in the logs.
        reference = uuid.uuid4().hex[:12]
        logger.exception(
            "Extraction failed ref=%s host=%s error=%s",
            reference,
            urlparse(str(request.url)).hostname,
            type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail=f"Source extraction failed (ref {reference})") from exc

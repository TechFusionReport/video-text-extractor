import asyncio
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from .config import get_settings
from .extractor import VideoExtractor
from .models import ExtractRequest, ExtractResponse
from .security import require_api_key

app = FastAPI(title="TFR Video Text Extractor", version="0.1.0")


@lru_cache
def get_extractor() -> VideoExtractor:
    return VideoExtractor(get_settings())


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/extract", response_model=ExtractResponse, dependencies=[Depends(require_api_key)])
async def extract(request: ExtractRequest) -> ExtractResponse:
    settings = get_settings()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                get_extractor().extract,
                str(request.url),
                request.language,
                request.force_transcription,
            ),
            timeout=settings.job_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Extraction timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Source extraction failed") from exc


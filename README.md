# TFR Video Text Extractor

Caption-first transcription plus optional local video-frame OCR. The API returns spoken
text in `transcript`/`segments`, the uploader's `description`, and (when requested)
visible on-screen text in `ocr_text`/`ocr_segments`. OCR runs locally with RapidOCR and
does not require a paid external service.

Standalone, caption-first video transcription API for TechFusion Report. It is intentionally independent of Cookbook so other TFR systems can reuse it.

## Processing order

1. Validate that the submitted URL resolves only to public addresses.
2. Use `yt-dlp` to retrieve metadata and native/automatic captions.
3. Return captions immediately when available.
4. Otherwise download one audio stream and transcribe it locally with Faster-Whisper.
5. Return normalized text plus timestamped segments and source metadata.

## Run with Docker

```bash
cp .env.example .env
# Set EXTRACTOR_API_KEY to a long random value.
docker compose up --build -d
curl http://127.0.0.1:8000/healthz
```

## API

```bash
curl -X POST http://127.0.0.1:8000/v1/extract \
  -H "Authorization: Bearer $EXTRACTOR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}'
```

The response includes `method` (`captions` or `whisper`), `transcript`, timestamped `segments`, `description`, detected language, title, author, platform, duration, source ID, and `warnings`.

- Sound cues such as `[Music]` and `♪` are stripped from caption text.
- `description` is the uploader's video description. Recipe videos often carry the full recipe there, so use it first.
- `warnings` lists degraded steps instead of failing: `no_description`, `transcript_short` (under 30 words, likely a text-overlay video), `no_captions_whisper_fallback`, `ocr_unavailable`, `ocr_skipped_too_long`, `ocr_failed`.
- OCR is opt-in per request: send `"ocr": true`. It also needs `OCR_ENABLED=true` on the server and is skipped for media longer than `OCR_MAX_MEDIA_SECONDS`. An OCR failure never discards the transcript.
- `GET /v1/info` (same Bearer auth) reports version, git SHA, limits, and runtime checks (deno, ffmpeg, writable cache dir).
- Errors: 422 bad input or over a duration limit, 503 + `Retry-After` when all `MAX_CONCURRENT_JOBS` slots stay busy, 504 timeout, 502 unexpected failure (the real error is in the container logs under the returned `ref`).

## Cookbook integration contract

Cookbook should call this service only for supported video URLs. Configure the client with:

- `VIDEO_TEXT_EXTRACTOR_URL`, such as `https://extractor.techfusionreport.com`
- `VIDEO_TEXT_EXTRACTOR_API_KEY`, matching `EXTRACTOR_API_KEY`

On a successful response, Cookbook should combine `transcript` with the source title/description before recipe structuring. A failed extraction must remain an explicit error and must not silently masquerade as a complete recipe.

## Operational notes

- The default `small.en` CPU/int8 configuration is suited to low-cost deployment. Change to a multilingual model when required.
- Models download on first transcription and persist in the Docker cache volume.
- Put the service behind Cloudflare Tunnel or another TLS reverse proxy; the container binds to localhost by default.
- Platform extraction can require updated `yt-dlp` releases as websites change.
- Requests impersonate an available Chrome target by default. Override or disable
  this with `YT_DLP_IMPERSONATE_TARGET`; verify targets with
  `yt-dlp --list-impersonate-targets` inside the image.
- If YouTube requires sign-in/bot verification, export a Netscape-format cookie
  file from an authorized account, mount it read-only into the container, and set
  `YT_DLP_COOKIE_FILE` to its in-container path. Never commit the cookie file.
- Only process media you are authorized to access and follow the source platform's terms.

## Provenance

The architecture is informed by the Apache-2.0 licensed AI Video Transcriber project. This implementation is an independent, minimal API using `yt-dlp` and Faster-Whisper; it does not copy that project's user interface or application code.

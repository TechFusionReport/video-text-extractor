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
- `warnings` lists degraded steps instead of failing: `no_description`, `transcript_short` (under 30 words, likely a text-overlay video), `no_captions_whisper_fallback`, `subtitles_unavailable` (subtitle download failed or the breaker skipped captions; use the duration-limited Whisper fallback), `ocr_unavailable`, `ocr_skipped_too_long`, `ocr_failed`.
- OCR is opt-in per request: send `"ocr": true`. It also needs `OCR_ENABLED=true` on the server and is skipped for media longer than `OCR_MAX_MEDIA_SECONDS`. An OCR failure never discards the transcript.
- `GET /v1/info` (same Bearer auth) reports version, git SHA, limits, and runtime checks (deno, ffmpeg, writable cache dir).
- Errors: 422 bad input or over a duration limit, 503 + `Retry-After` when all `MAX_CONCURRENT_JOBS` slots stay busy, 504 timeout, 502 unexpected failure (the real error is in the container logs under the returned `ref`).

| Warning | Meaning |
| --- | --- |
| `subtitles_unavailable` | Subtitle download failed or the breaker skipped captions; transcription uses the existing Whisper duration limit. Failures retry metadata once without subtitles; skips start without subtitles. |

Forced transcription disables both native subtitles and automatic captions in yt-dlp.

## Subtitle throttle circuit breaker

`SUBTITLE_BREAKER_SECONDS` is an integer from 0 to 86400, default **1800**
(30 minutes). Set it to **0** to disable the breaker and retain the single retry
on every subtitle failure. Only a subtitle HTTP 429 opens the breaker. Other
subtitle failures keep the existing retry and warning without changing its state.

While open, requests start metadata extraction with both native subtitles and
auto-captions disabled, return `subtitles_unavailable`, and use the existing
Whisper duration limit (422 when exceeded). After the window expires, one caller
atomically claims a 60-second probe lease; callers during that lease skip captions.
A successful metadata probe closes the breaker, even when no captions exist. A
429 probe starts the full cooldown again. A non-429 subtitle failure or unrelated
probe error leaves the lease in place until it expires. Forced transcription
never reads or changes the breaker.

State is shared by worker threads within each process, protected by a lock and
measured with a monotonic clock. It resets on restart; separate processes have
independent state. No extraction results are cached. Only opening and closing
transitions produce INFO logs, without request identifiers or exception text.

Authenticated `GET /v1/info` adds exactly these fields under `subtitles`:

| Field | Type | Meaning |
| --- | --- | --- |
| `breaker_open` | boolean | Requests currently skip captions, including during a probe lease. |
| `retry_after_seconds` | integer | Remaining cooldown or lease, rounded up; 0 when closed or eligible for a probe. |

Reading this status never claims a probe or changes breaker state. With the
breaker disabled, status is always false and 0.

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
- Datacenter IPs are often bot-checked even with impersonation. Set `YT_DLP_PROXY`
  (e.g. `http://10.10.0.5:8888`, a residential egress reachable over WireGuard) to
  route every yt-dlp request through it. Extraction fails closed if the proxy is
  down; it never falls back to a direct connection.
- Only process media you are authorized to access and follow the source platform's terms.

## Provenance

The architecture is informed by the Apache-2.0 licensed AI Video Transcriber project. This implementation is an independent, minimal API using `yt-dlp` and Faster-Whisper; it does not copy that project's user interface or application code.

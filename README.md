# TFR Video Text Extractor

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

The response includes `method` (`captions` or `whisper`), `transcript`, timestamped `segments`, detected language, title, author, platform, duration, and source ID.

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
- Only process media you are authorized to access and follow the source platform's terms.

## Provenance

The architecture is informed by the Apache-2.0 licensed AI Video Transcriber project. This implementation is an independent, minimal API using `yt-dlp` and Faster-Whisper; it does not copy that project's user interface or application code.

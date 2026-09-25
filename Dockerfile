FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 extractor
USER extractor
EXPOSE 8000
CMD ["uvicorn", "video_text_extractor.main:app", "--host", "0.0.0.0", "--port", "8000"]


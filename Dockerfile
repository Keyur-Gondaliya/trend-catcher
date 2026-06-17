# Trend Radar — slim image that runs the CLI.
# The app is a scheduled task runner (seed / digest / feedback / reset), not a
# long-lived server, so the entrypoint is `python run.py` and the command picks
# the subcommand. Example: `docker compose run --rm trend-radar digest`.
FROM python:3.12-slim

# Keep Python lean and unbuffered so logs stream out of the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install core deps first so this layer caches across code changes.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code.
COPY pyproject.toml run.py ./
COPY trend_radar/ ./trend_radar/

# Data dir is mounted as a volume in compose; create it for standalone runs too.
RUN mkdir -p data

ENTRYPOINT ["python", "run.py"]
CMD ["digest"]

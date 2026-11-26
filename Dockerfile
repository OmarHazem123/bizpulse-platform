# syntax=docker/dockerfile:1
#
# BizPulse production image (multi-stage).
#   builder: resolve dependencies from uv.lock into /app/.venv (no dev tools).
#   runtime: slim Python + the venv only, non-root uid 10001, HEALTHCHECK, and the
#            deterministic demo warehouse baked in so the dashboard is useful on start.
#
#   docker build -t bizpulse-platform .
#   docker run -p 8501:8501 bizpulse-platform              # dashboard
#   docker run --rm bizpulse-platform bizpulse smoke       # end-to-end self-test
#   docker run --rm bizpulse-platform bizpulse report --dataset ecommerce

ARG PYTHON_IMAGE=python:3.12-slim

# --------------------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first (cached until pyproject.toml / uv.lock change) ...
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
# ... then the project itself, installed as a regular (non-editable) package.
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# --------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="bizpulse-platform" \
      org.opencontainers.image.description="Decision intelligence platform: ingestion, data contracts, DuckDB star schema, BI, forecasting, anomaly detection and what-if simulation" \
      org.opencontainers.image.licenses="MIT"

# libgomp1: OpenMP runtime required by LightGBM. The app user owns only /app and its data.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 bizpulse \
    && useradd --uid 10001 --gid 10001 --home-dir /app --no-create-home \
       --shell /usr/sbin/nologin bizpulse \
    && mkdir -p /app/data \
    && chown 10001:10001 /app /app/data

# The venv stays root-owned (read-only for the app user).
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    HOME=/app \
    BIZPULSE_DATA_DIR=/app/data \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
WORKDIR /app
USER 10001:10001

# Deterministic demo data (seed 42, ~30 MB) so the dashboard has data on first start.
RUN bizpulse generate-demo && bizpulse run-pipeline --all

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"]

# No ENTRYPOINT: `docker run IMAGE bizpulse <command>` runs any CLI command.
CMD ["bizpulse", "dashboard", "--address", "0.0.0.0", "--port", "8501", "--headless", "--demo-if-empty"]

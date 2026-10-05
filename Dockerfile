# syntax=docker/dockerfile:1.7

FROM debian:bookworm-slim AS tectonic

ARG TECTONIC_VERSION=0.17.0
ARG TECTONIC_SHA256=8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --show-error \
        "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
        --output /tmp/tectonic.tar.gz \
    && echo "${TECTONIC_SHA256}  /tmp/tectonic.tar.gz" | sha256sum --check --strict \
    && tar --extract --gzip --file /tmp/tectonic.tar.gz --directory /tmp tectonic \
    && chmod 0755 /tmp/tectonic


FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project


FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        fontconfig \
        libfontconfig1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --home-dir /home/app app

COPY --from=tectonic /tmp/tectonic /usr/local/bin/tectonic

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app alembic.ini ./alembic.ini

RUN mkdir --parents /app/output /home/app/.cache \
    && chown --recursive app:app /app/output /home/app/.cache

ENV APP_ENV=production \
    HOME=/home/app \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    LANGGRAPH_STRICT_MSGPACK=true \
    XDG_CACHE_HOME=/home/app/.cache

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]

CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]

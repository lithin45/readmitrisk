# Pipeline + Streamlit image (multi-stage). One image serves both the batch pipeline and
# the demo; the compose service decides which command to run.
#
# A C toolchain is needed because some scikit-survival deps (e.g. ecos) lack prebuilt
# linux/arm64 wheels and compile from source. We build the venv in a "builder" stage and
# copy it into a lean runtime stage so the toolchain doesn't bloat the final image.

# ---------------------------------------------------------------------------
# Builder: install deps (with a compiler available) into /app/.venv
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer — cached unless pyproject/uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Project source + editable install of the package itself.
COPY src ./src
COPY config ./config
COPY sql ./sql
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runtime: slim image with just the OpenMP runtime + the prebuilt venv
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    READMIT_DATA_DIR=/app/data \
    PATH="/app/.venv/bin:${PATH}"

# libgomp1 is the OpenMP runtime scikit-learn / scikit-survival load at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

CMD ["python", "-c", "print('readmitrisk image ready — supply a command via compose')"]

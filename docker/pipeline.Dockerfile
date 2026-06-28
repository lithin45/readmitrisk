# Pipeline + Streamlit image. One image serves both the batch pipeline and the demo;
# the compose service decides which command to run. Prebuilt wheels mean no compiler
# is needed for scikit-survival / numba / shap on Python 3.11.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    READMIT_DATA_DIR=/app/data \
    PATH="/app/.venv/bin:${PATH}"

# libgomp1 is the OpenMP runtime scikit-learn / scikit-survival load at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer — cached unless pyproject/uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Project source.
COPY src ./src
COPY config ./config
COPY sql ./sql
RUN uv sync --frozen --no-dev

CMD ["python", "-c", "print('readmitrisk image ready — supply a command via compose')"]

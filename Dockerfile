# =============================================================================
# Multi-stage build.
#
# Stage 1 installs the dependencies into a virtual environment; stage 2 copies
# only that environment and the application, so compilers and build caches
# never reach the published image. That matters more here than in most
# projects: scikit-learn and its numerical stack pull in a large build
# toolchain, and shipping it would roughly double the image.
#
# The container runs as a non-root user, asserted in CI rather than trusted.
# =============================================================================

# ---------- Stage 1: builder -------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip setuptools wheel && pip install .

# ---------- Stage 2: runtime -------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="ml-production-platform" \
      org.opencontainers.image.description="Churn prediction platform: point-in-time features, promotion gates, drift monitoring." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/legend-cell05/ml-production-platform"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POLARIS_PROJECT_ROOT=/app \
    POLARIS_LOG_FORMAT=json \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    # scikit-learn and numpy each start a thread pool sized to the host's CPU
    # count. In a container with a CPU limit they oversubscribe it and every
    # prediction gets slower -- the opposite of what the numbers suggest.
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN apt-get update \
    && apt-get install --no-install-recommends -y postgresql-client curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 10001 polaris

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=polaris:polaris sql ./sql
COPY --chown=polaris:polaris README.md LICENSE pyproject.toml ./
RUN mkdir -p /app/data/raw /app/data/artifacts /app/data/reports \
    && chown -R polaris:polaris /app/data

USER polaris

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD polaris doctor > /dev/null 2>&1 || exit 1

ENTRYPOINT ["polaris"]
CMD ["--help"]

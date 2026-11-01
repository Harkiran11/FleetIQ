# syntax=docker/dockerfile:1.7

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# libgomp is required by scikit-learn's histogram gradient boosting kernels;
# curl is used by the container health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY frontend ./frontend
COPY models ./models
COPY docker/entrypoint.sh /usr/local/bin/fleetiq-entrypoint

RUN chmod +x /usr/local/bin/fleetiq-entrypoint \
    && useradd --create-home --uid 10001 fleetiq \
    && mkdir -p /app/data /app/models /mlartifacts \
    && chown -R fleetiq:fleetiq /app /mlartifacts

USER fleetiq

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

ENTRYPOINT ["/usr/local/bin/fleetiq-entrypoint"]
CMD ["gunicorn", "api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]

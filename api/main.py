"""FastAPI application entry point.

Run locally with ``python -m src.cli serve --reload`` or
``uvicorn api.main:app --reload``. The dashboard is served at ``/`` and the
OpenAPI docs at ``/docs``.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from api.routers import analytics, monitoring, predictions, system
from api.schemas import FreightResponse, MaintenanceResponse
from src.config import settings
from src.db import init_db
from src.logging_utils import configure_logging
from src.registry import ModelRegistry

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level, settings.log_json)
    init_db()
    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry
    app.state.started_at = time.time()
    loaded = registry.available()
    logger.info("FleetIQ %s started; models=%s", settings.app_version, loaded)
    if not all(loaded.values()):
        logger.warning("Starting in degraded mode; model errors: %s", registry.errors())
    yield


app = FastAPI(
    title="FleetIQ MLOps API",
    description=(
        "Maintenance risk scoring and freight price quoting for a logistics fleet. "
        "The API also exposes drift monitoring, prediction feedback and Prometheus metrics."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "predictions", "description": "Online scoring"},
        {"name": "analytics", "description": "Fleet and lane reporting"},
        {"name": "monitoring", "description": "Drift, traffic and feedback"},
        {"name": "system", "description": "Health, models and metrics"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


app.include_router(predictions.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(monitoring.router, prefix="/api")
app.include_router(system.router, prefix="/api")


# Compatibility aliases for the short paths used in earlier demos and README
# examples. They are hidden from the OpenAPI schema so /api remains canonical.
app.add_api_route(
    "/predict",
    predictions.predict_maintenance,
    methods=["POST"],
    response_model=MaintenanceResponse,
    include_in_schema=False,
)
app.add_api_route(
    "/predict/freight",
    predictions.predict_freight,
    methods=["POST"],
    response_model=FreightResponse,
    include_in_schema=False,
)
app.add_api_route("/health", system.health, methods=["GET"], include_in_schema=False)
app.add_api_route("/models", system.models, methods=["GET"], include_in_schema=False)
app.add_api_route("/metrics", system.metrics, methods=["GET"], include_in_schema=False)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/ui", include_in_schema=False)
def dashboard_alias() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api", include_in_schema=False)
def api_index() -> JSONResponse:
    return JSONResponse(
        {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "dashboard": "/",
            "health": "/api/health",
        }
    )


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

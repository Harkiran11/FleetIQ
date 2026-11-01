"""Health, readiness, model registry and Prometheus-style metrics."""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.deps import get_db, get_registry, require_admin
from api.schemas import HealthResponse, ModelReloadResponse
from src.config import settings
from src.db import ping
from src.db_models import PredictionEvent
from src.registry import MODEL_KEYS, ModelRegistry, current_dir

router = APIRouter(tags=["system"])


def _uptime(request: Request) -> float:
    started = getattr(request.app.state, "started_at", time.time())
    return round(time.time() - started, 3)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
) -> HealthResponse:
    database_up = ping()
    models = registry.available()
    if not database_up:
        state = "down"
    elif all(models.values()):
        state = "ok"
    else:
        state = "degraded"
    return HealthResponse(
        status=state,
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        database=database_up,
        models=models,
        uptime_seconds=_uptime(request),
    )


@router.get("/ready", summary="Readiness probe (503 until models load)")
def ready(
    registry: ModelRegistry = Depends(get_registry),
) -> dict[str, Any]:
    database_up = ping()
    models = registry.available()
    ready_state = database_up and all(models.values())
    payload = {
        "ready": ready_state,
        "database": database_up,
        "models": models,
        "errors": registry.errors(),
    }
    if not ready_state:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=payload)
    return payload


@router.get("/models", summary="Promoted model metadata")
def models(registry: ModelRegistry = Depends(get_registry)) -> list[dict[str, Any]]:
    return registry.info()


@router.get("/models/{key}/card", summary="Markdown model card for a promoted model")
def model_card(key: str) -> PlainTextResponse:
    if key not in MODEL_KEYS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown model key: {key}"
        )
    path = current_dir(key) / "model_card.md"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No promoted model card for {key}. Run `make train`.",
        )
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown")


@router.post(
    "/admin/reload",
    response_model=ModelReloadResponse,
    summary="Reload model bundles from disk or the MLflow registry",
)
def reload_models(
    source: Literal["local", "mlflow"] | None = Query(default=None),
    registry: ModelRegistry = Depends(get_registry),
    _: None = Depends(require_admin),
) -> ModelReloadResponse:
    registry.reload(source)
    return ModelReloadResponse(
        status="reloaded",
        source=registry.source,
        loaded_at=registry.loaded_at.isoformat() if registry.loaded_at else None,
        models=registry.info(),
        errors=registry.errors(),
    )


@router.get("/metrics", summary="Prometheus text exposition")
def metrics(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
    session: Session = Depends(get_db),
) -> Response:
    lines: list[str] = []
    lines.append("# HELP fleetiq_info Build information")
    lines.append("# TYPE fleetiq_info gauge")
    lines.append(
        f'fleetiq_info{{version="{settings.app_version}",environment="{settings.app_env}"}} 1'
    )
    lines.append("# HELP fleetiq_uptime_seconds Process uptime")
    lines.append("# TYPE fleetiq_uptime_seconds gauge")
    lines.append(f"fleetiq_uptime_seconds {_uptime(request)}")
    lines.append("# HELP fleetiq_db_up Database reachability")
    lines.append("# TYPE fleetiq_db_up gauge")
    lines.append(f"fleetiq_db_up {1 if ping() else 0}")

    lines.append("# HELP fleetiq_model_available Whether a promoted model is loaded")
    lines.append("# TYPE fleetiq_model_available gauge")
    for key, available in registry.available().items():
        lines.append(f'fleetiq_model_available{{model="{key}"}} {1 if available else 0}')

    lines.append("# HELP fleetiq_model_metric Holdout metric stored in the model bundle")
    lines.append("# TYPE fleetiq_model_metric gauge")
    for item in registry.info():
        if not item.get("available"):
            continue
        for metric, value in (item.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                lines.append(
                    f'fleetiq_model_metric{{model="{item["key"]}",metric="{metric}"}} {value}'
                )

    rows = session.execute(
        select(
            PredictionEvent.endpoint,
            func.count(PredictionEvent.id),
            func.avg(PredictionEvent.latency_ms),
        ).group_by(PredictionEvent.endpoint)
    ).all()
    lines.append("# HELP fleetiq_prediction_events_total Served predictions by endpoint")
    lines.append("# TYPE fleetiq_prediction_events_total counter")
    for endpoint, count, _latency in rows:
        lines.append(f'fleetiq_prediction_events_total{{endpoint="{endpoint}"}} {int(count)}')
    lines.append("# HELP fleetiq_prediction_latency_ms_avg Average inference latency")
    lines.append("# TYPE fleetiq_prediction_latency_ms_avg gauge")
    for endpoint, _count, latency in rows:
        average = round(float(latency or 0.0), 3)
        lines.append(f'fleetiq_prediction_latency_ms_avg{{endpoint="{endpoint}"}} {average}')

    feedback_total = int(
        session.execute(
            select(func.count(PredictionEvent.id)).where(PredictionEvent.actual_value.is_not(None))
        ).scalar_one()
    )
    lines.append("# HELP fleetiq_feedback_events_total Predictions with an observed outcome")
    lines.append("# TYPE fleetiq_feedback_events_total gauge")
    lines.append(f"fleetiq_feedback_events_total {feedback_total}")

    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

"""Shared FastAPI dependencies and small persistence helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from fastapi import Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.config import settings
from src.db import get_session_factory
from src.db_models import PredictionEvent
from src.registry import LoadedModel, ModelRegistry

logger = logging.getLogger(__name__)


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover - startup always sets this
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry has not been initialised",
        )
    return registry


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def require_admin(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> None:
    """Protect operational endpoints when ADMIN_TOKEN is configured.

    Local development runs without a token; production deployments should set
    one in the environment (docker-compose does).
    """
    if not settings.admin_token:
        return
    if x_admin_token != settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Token header",
        )


def client_label(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    if request.client:
        return request.client.host[:64]
    return None


def record_prediction(
    session: Session,
    *,
    endpoint: str,
    model: LoadedModel,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    latency_ms: float,
    client: str | None = None,
) -> str | None:
    """Persist one served prediction so drift and feedback have data to read."""
    if not settings.prediction_log_enabled:
        return None

    risk_probability = None
    predicted_cost = None
    if model.model_key == "maintenance":
        risk_probability = response_payload.get("risk", {}).get("probability")
    else:
        predicted_cost = response_payload.get("estimated_cost_usd")

    event = PredictionEvent(
        endpoint=endpoint,
        model_name=str(model.metadata.get("model_name", model.model_key)),
        model_version=model.version,
        run_id=model.run_id,
        request=request_payload,
        response=response_payload,
        latency_ms=latency_ms,
        risk_probability=risk_probability,
        predicted_cost=predicted_cost,
        client=client,
    )
    try:
        session.add(event)
        session.commit()
        return event.id
    except Exception:  # pragma: no cover - logging must never break serving
        session.rollback()
        logger.exception("Failed to persist prediction event")
        return None


def utc_iso(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat()

"""Drift, traffic, prediction log and human-feedback endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.deps import get_db, get_registry
from api.schemas import FeedbackRequest, FeedbackResponse
from src.config import settings
from src.db_models import PredictionEvent
from src.monitoring import (
    drift_from_events,
    feedback_summary,
    prediction_summary,
    recent_events,
)
from src.registry import ModelNotFoundError, ModelRegistry

router = APIRouter(tags=["monitoring"])


def _serialize(event: PredictionEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "endpoint": event.endpoint,
        "model_name": event.model_name,
        "model_version": event.model_version,
        "run_id": event.run_id,
        "latency_ms": round(float(event.latency_ms or 0.0), 2),
        "risk_probability": event.risk_probability,
        "predicted_cost": event.predicted_cost,
        "actual_value": event.actual_value,
        "feedback_at": event.feedback_at.isoformat() if event.feedback_at else None,
        "client": event.client,
    }


@router.get("/monitoring/drift", summary="PSI drift report from recent requests")
def get_drift(
    model: Literal["all", "maintenance", "freight"] = Query(default="all"),
    window: int | None = Query(default=None, ge=25, le=5000),
    registry: ModelRegistry = Depends(get_registry),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    keys = ["maintenance", "freight"] if model == "all" else [model]
    reports: dict[str, Any] = {}
    for key in keys:
        try:
            reports[key] = drift_from_events(registry.get(key), session, window=window)
        except ModelNotFoundError as exc:
            reports[key] = {
                "model": key,
                "status": "model_unavailable",
                "error": str(exc),
                "window": 0,
                "features": [],
            }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_limit": window or settings.drift_window,
        "reports": reports,
    }


@router.get("/monitoring/traffic", summary="Prediction traffic and latency")
def get_traffic(
    hours: int = Query(default=24, ge=1, le=720),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return prediction_summary(session, hours=hours)


@router.get("/monitoring/feedback", summary="Accuracy from labelled predictions")
def get_feedback(session: Session = Depends(get_db)) -> dict[str, Any]:
    return feedback_summary(session)


@router.get("/monitoring/events", summary="Most recent logged predictions")
def get_events(
    limit: int = Query(default=25, ge=1, le=200),
    session: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    return [_serialize(event) for event in recent_events(session, limit=limit)]


@router.post(
    "/feedback/{event_id}",
    response_model=FeedbackResponse,
    summary="Attach the observed outcome to a logged prediction",
)
def submit_feedback(
    event_id: str,
    payload: FeedbackRequest,
    session: Session = Depends(get_db),
) -> FeedbackResponse:
    event = session.get(PredictionEvent, event_id)
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction event {event_id} not found",
        )
    event.actual_value = float(payload.actual_value)
    event.feedback_at = datetime.now(UTC)
    if payload.note:
        response = dict(event.response or {})
        response["feedback_note"] = payload.note
        event.response = response
    session.commit()
    return FeedbackResponse(
        event_id=event.id,
        actual_value=event.actual_value,
        feedback_at=event.feedback_at.isoformat(),
    )

"""Prediction endpoints for maintenance risk and freight quotes."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.deps import client_label, get_db, get_registry, record_prediction
from api.schemas import (
    BatchError,
    BatchResponse,
    FreightRequest,
    FreightResponse,
    MaintenanceRequest,
    MaintenanceResponse,
)
from src.config import settings
from src.features import lane_benchmark
from src.predict import FreightPredictor, MaintenancePredictor
from src.registry import LoadedModel, ModelNotFoundError, ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predictions"])


def _load(registry: ModelRegistry, key: str) -> LoadedModel:
    try:
        return registry.get(key)
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.post(
    "/predict/maintenance",
    response_model=MaintenanceResponse,
    summary="Score one vehicle for maintenance risk",
)
def predict_maintenance(
    payload: MaintenanceRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
    session: Session = Depends(get_db),
) -> MaintenanceResponse:
    model = _load(registry, "maintenance")
    started = perf_counter()
    result = MaintenancePredictor(model).predict(payload.model_dump(exclude_none=True))
    latency = (perf_counter() - started) * 1000.0
    prediction_id = record_prediction(
        session,
        endpoint="/api/predict/maintenance",
        model=model,
        request_payload=payload.model_dump(mode="json"),
        response_payload=result,
        latency_ms=latency,
        client=client_label(request),
    )
    result["prediction_id"] = prediction_id
    return MaintenanceResponse(**result)


@router.post(
    "/predict/maintenance/batch",
    response_model=BatchResponse,
    summary="Score up to 500 vehicles in one call",
)
def predict_maintenance_batch(
    request: Request,
    payloads: list[MaintenanceRequest] = Body(..., min_length=1),
    registry: ModelRegistry = Depends(get_registry),
    session: Session = Depends(get_db),
) -> BatchResponse:
    if len(payloads) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {len(payloads)} exceeds the {settings.max_batch_size} record limit",
        )
    model = _load(registry, "maintenance")
    predictor = MaintenancePredictor(model)
    results: list[dict[str, Any]] = []
    errors: list[BatchError] = []
    for index, payload in enumerate(payloads):
        try:
            started = perf_counter()
            result = predictor.predict(payload.model_dump(exclude_none=True))
            latency = (perf_counter() - started) * 1000.0
            result["prediction_id"] = record_prediction(
                session,
                endpoint="/api/predict/maintenance/batch",
                model=model,
                request_payload=payload.model_dump(mode="json"),
                response_payload=result,
                latency_ms=latency,
                client=client_label(request) if request else None,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("Batch record %s failed: %s", index, exc)
            errors.append(BatchError(index=index, error=str(exc)))
    return BatchResponse(
        count=len(payloads),
        succeeded=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )


@router.post(
    "/predict/freight",
    response_model=FreightResponse,
    summary="Quote a shipment and return a 90% cost interval",
)
def predict_freight(
    payload: FreightRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
    session: Session = Depends(get_db),
) -> FreightResponse:
    model = _load(registry, "freight")
    raw = payload.model_dump(mode="json")
    benchmark = lane_benchmark(session.get_bind(), payload.origin_warehouse, payload.destination)
    started = perf_counter()
    result = FreightPredictor(model).predict(raw, benchmark)
    latency = (perf_counter() - started) * 1000.0
    prediction_id = record_prediction(
        session,
        endpoint="/api/predict/freight",
        model=model,
        request_payload=raw,
        response_payload=result,
        latency_ms=latency,
        client=client_label(request),
    )
    result["prediction_id"] = prediction_id
    return FreightResponse(**result)


@router.post(
    "/predict/freight/batch",
    response_model=BatchResponse,
    summary="Quote up to 500 shipments in one call",
)
def predict_freight_batch(
    request: Request,
    payloads: list[FreightRequest] = Body(..., min_length=1),
    registry: ModelRegistry = Depends(get_registry),
    session: Session = Depends(get_db),
) -> BatchResponse:
    if len(payloads) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {len(payloads)} exceeds the {settings.max_batch_size} record limit",
        )
    model = _load(registry, "freight")
    predictor = FreightPredictor(model)
    engine = session.get_bind()
    results: list[dict[str, Any]] = []
    errors: list[BatchError] = []
    for index, payload in enumerate(payloads):
        try:
            raw = payload.model_dump(mode="json")
            benchmark = lane_benchmark(engine, payload.origin_warehouse, payload.destination)
            started = perf_counter()
            result = predictor.predict(raw, benchmark)
            latency = (perf_counter() - started) * 1000.0
            result["prediction_id"] = record_prediction(
                session,
                endpoint="/api/predict/freight/batch",
                model=model,
                request_payload=raw,
                response_payload=result,
                latency_ms=latency,
                client=client_label(request) if request else None,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("Freight batch record %s failed: %s", index, exc)
            errors.append(BatchError(index=index, error=str(exc)))
    return BatchResponse(
        count=len(payloads),
        succeeded=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )

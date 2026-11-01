"""Dashboard analytics: fleet summaries, segments, lanes and a scored queue."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.deps import get_db, get_registry
from src import analytics
from src.features import (
    brake_condition_stats,
    carrier_statistics,
    lane_benchmark,
    lane_statistics,
    monthly_maintenance,
    segment_risk,
)
from src.monitoring import prediction_summary
from src.predict import MaintenancePredictor
from src.registry import ModelNotFoundError, ModelRegistry

router = APIRouter(tags=["analytics"])


@router.get("/analytics/overview", summary="Fleet, shipment and traffic KPIs")
def get_overview(
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    payload = analytics.overview(session.get_bind())
    payload["predictions"] = prediction_summary(session, hours=24)
    return payload


@router.get("/analytics/segments", summary="Risk by vehicle type")
def get_segments(session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return segment_risk(session.get_bind())


@router.get("/analytics/maintenance-trend", summary="Monthly service volume and risk rate")
def get_trend(session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return monthly_maintenance(session.get_bind())


@router.get("/analytics/lanes", summary="Busiest freight lanes")
def get_lanes(
    min_shipments: int = Query(default=3, ge=1, le=100),
    session: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    return lane_statistics(session.get_bind(), min_shipments=min_shipments)


@router.get("/analytics/carriers", summary="Carrier cost comparison")
def get_carriers(session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return carrier_statistics(session.get_bind())


@router.get("/analytics/brake-condition", summary="Risk by brake condition")
def get_brake_stats(session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return brake_condition_stats(session.get_bind())


@router.get("/analytics/lane-benchmark", summary="Observed cost distribution for a lane")
def get_lane_benchmark(
    origin: str = Query(..., min_length=1),
    destination: str = Query(..., min_length=1),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    result = lane_benchmark(session.get_bind(), origin, destination)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No completed shipments found for {origin} -> {destination}",
        )
    return result


@router.get("/analytics/options", summary="Categorical values used by the forms")
def get_options(session: Session = Depends(get_db)) -> dict[str, list[str]]:
    return analytics.options(session.get_bind())


@router.get("/shipments/recent", summary="Recent completed shipments")
def get_recent_shipments(
    limit: int = Query(default=12, ge=1, le=100),
    session: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    return analytics.shipment_sample(session.get_bind(), limit=limit)


@router.get("/vehicles/priority", summary="Model-scored maintenance priority queue")
def get_priority_queue(
    limit: int = Query(default=20, ge=1, le=100),
    registry: ModelRegistry = Depends(get_registry),
    session: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    try:
        model = registry.get("maintenance")
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    predictor = MaintenancePredictor(model)
    rows: list[dict[str, Any]] = []
    for item in analytics.priority_vehicles(session.get_bind(), limit=limit):
        record = item["record"]
        result = predictor.predict(record)
        signals = result.get("signals") or []
        rows.append(
            {
                "vehicle_id": record.get("vehicle_id"),
                "make_and_model": record.get("make_and_model"),
                "vehicle_type": record.get("vehicle_type"),
                "usage_hours": record.get("usage_hours"),
                "brake_condition": record.get("brake_condition"),
                "failure_history": record.get("failure_history"),
                "anomalies_detected": record.get("anomalies_detected"),
                "probability": result["risk"]["probability"],
                "level": result["risk"]["level"],
                "threshold": result["risk"]["threshold"],
                "maintenance_required": result["risk"]["maintenance_required"],
                "actual_maintenance_required": item.get("actual_maintenance_required"),
                "top_signal": signals[0]["label"] if signals else None,
                "recommended_action": result["recommended_action"],
            }
        )
    return sorted(rows, key=lambda row: row["probability"], reverse=True)


@router.get("/vehicles/{vehicle_id}", summary="Vehicle profile plus live risk score")
def get_vehicle(
    vehicle_id: int,
    registry: ModelRegistry = Depends(get_registry),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    profile = analytics.vehicle_profile(session.get_bind(), vehicle_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle {vehicle_id} not found",
        )
    record = {
        key: value for key, value in profile.items() if key in analytics.MAINTENANCE_INPUT_COLUMNS
    }
    response: dict[str, Any] = {
        "vehicle_id": vehicle_id,
        "record": record,
        "observed": {
            "maintenance_required": profile.get("maintenance_required"),
            "maintenance_cost": profile.get("maintenance_cost"),
            "downtime_maintenance": profile.get("downtime_maintenance"),
            "last_maintenance_date": profile.get("last_maintenance_date"),
        },
        "prediction": None,
        "error": None,
    }
    try:
        model = registry.get("maintenance")
        response["prediction"] = MaintenancePredictor(model).predict(record)
    except Exception as exc:
        response["error"] = str(exc)
    return response

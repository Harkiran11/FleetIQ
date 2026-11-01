from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.predict import (
    FreightPredictor,
    MaintenancePredictor,
    _action,
    _model_info,
    _risk_level,
)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.90, "critical"),
        (0.50, "critical"),
        (0.40, "high"),
        (0.20, "moderate"),
        (0.10, "low"),
    ],
)
def test_risk_bands_are_anchored_to_the_threshold(probability, expected):
    assert _risk_level(probability, 0.5) == expected


def test_risk_bands_work_when_threshold_is_near_one():
    # A separable synthetic dataset pushes the tuned threshold to ~1.0. The
    # bands must still be readable instead of collapsing into one level.
    threshold = 0.9999
    assert _risk_level(1.0, threshold) == "critical"
    assert _risk_level(0.80, threshold) == "high"
    assert _risk_level(0.40, threshold) == "moderate"
    assert _risk_level(0.10, threshold) == "low"


def test_action_prioritises_an_urgent_service_note():
    action = _action("low", "Good", {"urgency": "high"})
    assert "service note immediately" in action


def test_action_prioritises_poor_brakes():
    action = _action("low", "Poor", {"urgency": "low"})
    assert "brake inspection" in action


@pytest.mark.parametrize(
    ("level", "fragment"),
    [
        ("critical", "48 hours"),
        ("high", "7 days"),
        ("moderate", "next scheduled service"),
        ("low", "No immediate action"),
    ],
)
def test_action_for_each_risk_level(level, fragment):
    assert fragment in _action(level, "Good", {"urgency": "low"})


def test_maintenance_predictor_returns_explainable_result(maintenance_model, maintenance_payload):
    result = MaintenancePredictor(maintenance_model).predict(maintenance_payload)

    assert result["status"] == "success"
    assert result["model"]["name"] == "fleetiq-maintenance-risk"
    assert result["model"]["source"] == "local"
    assert 0.0 <= result["risk"]["probability"] <= 1.0
    assert result["risk"]["level"] in {"low", "moderate", "high", "critical"}
    assert result["risk"]["threshold"] == pytest.approx(round(maintenance_model.threshold, 6))
    assert isinstance(result["risk"]["maintenance_required"], bool)
    assert result["recommended_action"]
    assert len(result["signals"]) <= 4
    assert all({"label", "impact", "direction"} <= set(signal) for signal in result["signals"])
    assert result["service_note"]["error_codes"] == ["ERR-431"]
    assert result["service_note"]["mileage"] == 184000


def test_maintenance_predictor_without_note_has_no_parsed_note(
    maintenance_model, maintenance_payload
):
    maintenance_payload.pop("service_note")

    result = MaintenancePredictor(maintenance_model).predict(maintenance_payload)

    assert result["service_note"] is None


def test_model_info_exposes_version_and_run(maintenance_model):
    info = _model_info(maintenance_model)

    assert info["version"] == maintenance_model.version
    assert info["run_id"] == maintenance_model.run_id
    assert info["trained_at"]


def test_freight_predictor_returns_interval_and_per_mile_cost(freight_model, freight_payload):
    result = FreightPredictor(freight_model).predict(freight_payload)

    assert result["status"] == "success"
    assert result["estimated_cost_usd"] >= 0
    low = result["interval_usd"]["low"]
    high = result["interval_usd"]["high"]
    assert low <= result["estimated_cost_usd"] <= high
    assert result["interval_usd"]["coverage"] == 0.9
    assert result["cost_per_mile_usd"] == pytest.approx(
        result["estimated_cost_usd"] / freight_payload["distance_miles"], abs=1e-3
    )
    expected = date.fromisoformat(freight_payload["shipment_date"]) + timedelta(days=7)
    assert result["quote_valid_until"] == expected.isoformat()
    assert result["lane_benchmark"] is None


def test_freight_predictor_passes_through_lane_benchmark(freight_model, freight_payload):
    benchmark = {
        "origin": freight_payload["origin_warehouse"],
        "destination": freight_payload["destination"],
        "shipments": 14,
        "median_cost": 221.0,
        "p25_cost": 103.0,
        "p75_cost": 257.0,
        "avg_distance": 1100.0,
    }

    result = FreightPredictor(freight_model).predict(freight_payload, benchmark)

    assert result["lane_benchmark"] == benchmark


def test_freight_predictor_handles_zero_distance(freight_model, freight_payload):
    freight_payload["distance_miles"] = 0.0

    result = FreightPredictor(freight_model).predict(freight_payload)

    assert result["cost_per_mile_usd"] is None

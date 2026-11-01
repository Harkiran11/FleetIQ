from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FREIGHT_FEATURES,
    MAINTENANCE_EXCLUDED_COLUMNS,
    MAINTENANCE_FEATURES,
    brake_condition_stats,
    carrier_statistics,
    lane_benchmark,
    lane_statistics,
    monthly_maintenance,
    parse_service_note,
    prepare_freight_features,
    prepare_maintenance_features,
    segment_risk,
    validate_maintenance_frame,
    validate_shipment_frame,
)


def test_feature_contract_has_no_leakage_columns():
    assert set(MAINTENANCE_FEATURES).isdisjoint(MAINTENANCE_EXCLUDED_COLUMNS)
    # Post-service fields must never be treated as inputs.
    for leaky in (
        "maintenance_cost",
        "maintenance_severity",
        "predictive_score",
        "downtime_maintenance",
        "days_since_last_maintenance",
        "data_split",
    ):
        assert leaky not in MAINTENANCE_FEATURES


def test_prepare_maintenance_features_derives_ratio_features():
    frame = pd.DataFrame(
        [
            {
                "year_of_manufacture": 2015,
                "usage_hours": 10000.0,
                "load_capacity": 10000.0,
                "actual_load": 9000.0,
                "engine_temperature": 100.0,
                "vibration_levels": 4.0,
                "diagnostic_trouble_code_count": 8,
                "sensor_packet_loss_rate": 0.05,
                "historical_maintenance_cost": 2000.0,
                "brake_condition": "Poor",
                "failure_history": 3,
                "anomalies_detected": 2,
                "make_and_model": "Volvo FH",
                "vehicle_type": "Heavy Truck",
            }
        ]
    )

    result = prepare_maintenance_features(frame, reference_year=2025)

    assert list(result.columns) == MAINTENANCE_FEATURES
    row = result.iloc[0]
    assert row["vehicle_age"] == pytest.approx(10.0)
    assert row["usage_per_year"] == pytest.approx(1000.0)
    assert row["load_ratio"] == pytest.approx(0.9)
    assert row["temp_delta"] == pytest.approx(10.0)
    assert row["vibration_per_1k_hours"] == pytest.approx(0.4)
    assert row["dtc_per_10k_hours"] == pytest.approx(8.0)
    assert row["packet_loss_pct"] == pytest.approx(5.0)
    assert row["maintenance_cost_per_hour"] == pytest.approx(0.2)
    assert row["brake_wear_flag"] == 1.0
    assert row["failure_flag"] == 1.0
    assert row["anomaly_flag"] == 1.0
    assert row["make_and_model"] == "Volvo FH"


def test_prepare_maintenance_features_tolerates_missing_columns():
    result = prepare_maintenance_features(pd.DataFrame([{"vehicle_id": 1}]))

    assert list(result.columns) == MAINTENANCE_FEATURES
    assert result["vehicle_age"].iloc[0] == 0.0
    assert result["make_and_model"].iloc[0] == "unknown"
    assert np.isnan(result["usage_hours"].iloc[0])


def test_prepare_freight_features_uses_booking_time_only():
    frame = pd.DataFrame(
        [
            {
                "origin_warehouse": "Warehouse_CHI",
                "destination": "Dallas",
                "carrier": "UPS",
                "shipment_date": "2025-03-15",
                "weight_kg": 40.0,
                "distance_miles": 1000.0,
                # Post-delivery columns are present but must not be used.
                "cost": 999.0,
                "transit_days": 9,
                "delivery_date": "2025-03-24",
            }
        ]
    )

    result = prepare_freight_features(frame)

    assert list(result.columns) == FREIGHT_FEATURES
    row = result.iloc[0]
    assert row["month"] == 3.0
    assert row["day_of_week"] == 5.0  # Saturday
    assert row["is_weekend"] == 1.0
    assert row["weight_per_mile"] == pytest.approx(0.04)
    assert row["lane"] == "Warehouse_CHI->Dallas"
    assert row["distance_band"] == "1000-1500"
    assert row["weight_band"] == "25-50"
    assert "cost" not in result.columns
    assert "transit_days" not in result.columns


def test_parse_service_note_extracts_codes_parts_and_urgency():
    note = (
        "Driver reports brake vibration and an overheat smell near the wheel, "
        "ERR-431 and err 902. P/N ABC-1234 fitted at 184,000 miles."
    )

    parsed = parse_service_note(note)

    assert parsed["error_codes"] == ["ERR-431", "ERR-902"]
    assert parsed["part_numbers"] == ["ABC-1234"]
    assert parsed["mileage"] == 184000
    assert parsed["urgency"] == "high"
    assert parsed["urgency_score"] >= 4
    assert {"brake", "vibration", "overheat"} <= set(parsed["keywords"])


def test_parse_service_note_empty_and_benign_text():
    assert parse_service_note(None)["urgency"] == "low"
    assert parse_service_note("   ")["error_codes"] == []

    benign = parse_service_note("Routine oil change completed, no issues found.")
    assert benign["urgency"] == "low"
    assert benign["error_codes"] == []
    assert benign["mileage"] is None


def test_validate_maintenance_frame_reports_quality_issues():
    frame = pd.DataFrame(
        {
            "vehicle_id": [1, 1],
            "maintenance_required": [1, 2],
        }
    )

    report = validate_maintenance_frame(frame)

    assert report["rows"] == 2
    assert report["missing_columns"]
    assert report["invalid_target_rows"] == 1
    assert report["duplicate_vehicle_ids"] == 1
    assert report["ok"] is False


def test_validate_shipment_frame_reports_negative_distance():
    frame = pd.DataFrame(
        {
            "shipment_id": ["S1", "S1"],
            "origin_warehouse": ["A", "A"],
            "destination": ["B", "B"],
            "carrier": ["UPS", "UPS"],
            "shipment_date": ["2025-01-01", "2025-01-02"],
            "distance_miles": [100.0, -5.0],
        }
    )

    report = validate_shipment_frame(frame)

    assert report["negative_distance_rows"] == 1
    assert report["duplicate_shipment_ids"] == 1
    assert report["ok"] is False


def test_sql_analytics_run_against_the_test_warehouse(engine):
    segments = segment_risk(engine)
    assert segments
    assert {"vehicle_type", "vehicles", "risk_rate"} <= set(segments[0])

    trend = monthly_maintenance(engine)
    assert trend
    assert {"month", "events", "risk_rate"} <= set(trend[0])

    lanes = lane_statistics(engine, min_shipments=2)
    assert lanes
    assert {"origin_warehouse", "destination", "shipments", "avg_cost"} <= set(lanes[0])

    benchmark = lane_benchmark(engine, lanes[0]["origin_warehouse"], lanes[0]["destination"])
    assert benchmark is not None
    assert benchmark["p25_cost"] <= benchmark["median_cost"] <= benchmark["p75_cost"]

    carriers = carrier_statistics(engine)
    assert carriers
    assert {"carrier", "shipments", "avg_cost"} <= set(carriers[0])

    brakes = brake_condition_stats(engine)
    assert brakes
    assert {"brake_condition", "vehicles", "risk_rate"} <= set(brakes[0])

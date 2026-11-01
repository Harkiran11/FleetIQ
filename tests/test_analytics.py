from __future__ import annotations

from src import analytics


def test_overview_summarises_the_test_warehouse(engine):
    payload = analytics.overview(engine)

    maintenance = payload["maintenance"]
    assert maintenance["total_vehicles"] == 5000
    assert 0.0 <= maintenance["maintenance_rate"] <= 1.0
    assert maintenance["distinct_models"] > 0
    assert maintenance["poor_brake_vehicles"] > 0

    shipments = payload["shipments"]
    assert shipments["total_shipments"] == 243
    assert shipments["total_spend"] > 0
    assert shipments["avg_distance"] > 0

    statuses = {row["status"] for row in payload["shipment_status"]}
    assert {"Delivered", "Delayed"} <= statuses


def test_priority_vehicles_returns_raw_records(engine):
    rows = analytics.priority_vehicles(engine, limit=5)

    assert len(rows) == 5
    for item in rows:
        assert {"record", "actual_maintenance_required"} <= set(item)
        record = item["record"]
        assert record["vehicle_id"] is not None
        assert record["make_and_model"]
        assert "maintenance_cost" not in record
        assert "predictive_score" not in record
        assert item["actual_maintenance_required"] in (0, 1)


def test_vehicle_profile_returns_inputs_and_observed_outcomes(engine):
    row = analytics.priority_vehicles(engine, limit=1)[0]["record"]

    profile = analytics.vehicle_profile(engine, row["vehicle_id"])

    assert profile is not None
    assert profile["vehicle_id"] == row["vehicle_id"]
    assert "maintenance_required" in profile
    assert "maintenance_cost" in profile
    assert "last_maintenance_date" in profile


def test_vehicle_profile_returns_none_for_unknown_id(engine):
    assert analytics.vehicle_profile(engine, 999_999_999) is None


def test_options_returns_form_values(engine):
    options = analytics.options(engine, limit=10)

    assert set(analytics.OPTION_COLUMNS["maintenance"]) <= set(options)
    assert set(analytics.OPTION_COLUMNS["shipments"]) <= set(options)
    assert options["brake_condition"]
    assert options["origin_warehouse"]
    assert all(len(values) <= 10 for values in options.values())


def test_shipment_sample_returns_recent_completed_rows(engine):
    rows = analytics.shipment_sample(engine, limit=5)

    assert len(rows) == 5
    assert all(row["cost"] is not None for row in rows)
    dates = [row["shipment_date"] for row in rows]
    assert dates == sorted(dates, reverse=True)


def test_number_helper_rounds_floats_only():
    assert analytics._number(None) is None
    assert analytics._number(1.23456) == 1.2346
    assert analytics._number(7) == 7
    assert analytics._number("text") == "text"

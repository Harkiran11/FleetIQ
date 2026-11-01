from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingest import (
    _clean_maintenance,
    _clean_shipments,
    _json_safe,
    _read_csv,
    _records,
    locate_sources,
    normalize_column_name,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Vehicle ID", "vehicle_id"),
        ("Maintenance-Required", "maintenance_required"),
        ("CANMessageRateHz", "can_message_rate_hz"),
        ("  sensor packet loss rate ", "sensor_packet_loss_rate"),
        ("HTTPStatus", "http_status"),
        ("__already_snake__", "already_snake"),
    ],
)
def test_normalize_column_name(raw, expected):
    assert normalize_column_name(raw) == expected


def test_read_csv_normalizes_columns_and_drops_unnamed(tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text(
        "Unnamed: 0,Vehicle ID,Maintenance-Required\n0,101,1\n1,102,0\n",
        encoding="utf-8",
    )

    frame = _read_csv(path)

    assert list(frame.columns) == ["vehicle_id", "maintenance_required"]
    assert len(frame) == 2


def test_clean_maintenance_drops_missing_ids_and_fills_defaults():
    frame = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3, None],
            "maintenance_required": [1, 0, 0, 1],
            "last_maintenance_date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "usage_hours": [100.0, None, 300.0, 400.0],
            "brake_condition": ["Good", None, "Poor", None],
            "make_and_model": ["Volvo FH", "Isuzu NPR", None, "Volvo FH"],
        }
    )

    cleaned, report = _clean_maintenance(frame)

    assert len(cleaned) == 3
    assert report["rows"] == 3
    assert report["target_rate"] == pytest.approx(1 / 3)
    assert report["missing_filled"]["usage_hours"] == 1
    assert report["missing_filled"]["brake_condition"] == 1
    assert cleaned["usage_hours"].tolist() == [100.0, 200.0, 300.0]
    assert cleaned["brake_condition"].tolist() == ["Good", "unknown", "Poor"]
    assert cleaned["make_and_model"].tolist() == ["Volvo FH", "Isuzu NPR", "unknown"]
    assert cleaned["vehicle_id"].dtype.kind in "iu"
    assert cleaned["last_maintenance_date"].notna().all()
    assert report["date_range"] == ["2024-01-02", "2024-01-04"]
    assert report["validation"]["rows"] == 3


def test_clean_maintenance_clips_invalid_targets():
    frame = pd.DataFrame(
        {
            "vehicle_id": [1, 2],
            "maintenance_required": [1, 7],
            "last_maintenance_date": ["2024-01-01", "2024-01-02"],
        }
    )

    cleaned, report = _clean_maintenance(frame)

    assert sorted(cleaned["maintenance_required"].tolist()) == [1, 1]
    assert report["target_rate"] == pytest.approx(1.0)


def test_clean_shipments_drops_rows_without_required_keys():
    frame = pd.DataFrame(
        {
            "shipment_id": ["S1", None, "S3"],
            "origin_warehouse": ["Warehouse_CHI", "Warehouse_MIA", "Warehouse_MIA"],
            "destination": ["Dallas", "Atlanta", None],
            "carrier": ["UPS", "DHL", "DHL"],
            "shipment_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "weight_kg": [40.0, None, 20.0],
            "distance_miles": [1000.0, 500.0, None],
            "cost": [250.0, None, 120.0],
            "status": ["Delivered", "Delivered", None],
            "transit_days": [3, None, 2],
        }
    )

    cleaned, report = _clean_shipments(frame)

    assert len(cleaned) == 1
    assert report["rows"] == 1
    assert report["rows_without_cost"] == 0
    assert report["avg_cost"] == pytest.approx(250.0)
    assert cleaned["shipment_id"].tolist() == ["S1"]
    assert cleaned["status"].tolist() == ["Delivered"]
    assert cleaned["transit_days"].dtype.kind in "iu"
    assert cleaned["shipment_date"].iloc[0].isoformat() == "2025-01-01"
    assert report["validation"]["ok"] is True


def test_clean_shipments_reports_missing_cost_and_fills_measurements():
    frame = pd.DataFrame(
        {
            "shipment_id": ["S1", "S2", "S3"],
            "origin_warehouse": ["A", "A", "A"],
            "destination": ["B", "B", "B"],
            "carrier": ["UPS", "UPS", "UPS"],
            "shipment_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "weight_kg": [10.0, None, 30.0],
            "distance_miles": [100.0, 200.0, None],
            "cost": [100.0, None, 300.0],
            "status": ["Delivered", "Delivered", "Delivered"],
            "transit_days": [1, 2, 3],
        }
    )

    cleaned, report = _clean_shipments(frame)

    assert report["rows_without_cost"] == 1
    assert report["missing_filled"]["weight_kg"] == 1
    assert report["missing_filled"]["distance_miles"] == 1
    assert cleaned["weight_kg"].isna().sum() == 0
    assert cleaned["distance_miles"].isna().sum() == 0


def test_json_safe_and_records_convert_numpy_and_dates():
    payload = {
        "count": np.int64(4),
        "rate": np.float64(0.25),
        "missing": np.nan,
        "date": pd.Timestamp("2025-01-01"),
        "nested": [np.int32(1), np.float32(2.5)],
    }

    safe = _json_safe(payload)

    assert safe["count"] == 4
    assert safe["rate"] == 0.25
    assert safe["missing"] is None
    assert safe["date"] == "2025-01-01T00:00:00"
    assert safe["nested"] == [1, 2.5]

    frame = pd.DataFrame(
        {
            "vehicle_id": [1.0],
            "usage_hours": [np.float64(12.5)],
            "last_maintenance_date": [pd.Timestamp("2024-06-01")],
        }
    )
    records = _records(
        frame, ["vehicle_id", "usage_hours", "last_maintenance_date"], {"vehicle_id"}
    )

    assert records == [
        {
            "vehicle_id": 1,
            "usage_hours": 12.5,
            "last_maintenance_date": pd.Timestamp("2024-06-01").date(),
        }
    ]


def test_locate_sources_finds_both_raw_csvs(tmp_path, monkeypatch):
    from src.config import settings

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "logistics_predictive_maintenance.csv").write_text(
        "vehicle_id,maintenance_required\n1,0\n", encoding="utf-8"
    )
    (raw / "logistics_shipments.csv").write_text(
        "shipment_id,origin_warehouse,destination,carrier\nS1,A,B,UPS\n", encoding="utf-8"
    )

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "maintenance_csv", None)
    monkeypatch.setattr(settings, "shipments_csv", None)

    maintenance, shipments = locate_sources()

    assert maintenance.exists()
    assert shipments.exists()
    assert maintenance == raw / "logistics_predictive_maintenance.csv"
    assert shipments == raw / "logistics_shipments.csv"
    assert "maintenance" in maintenance.name.lower()
    assert "shipment" in shipments.name.lower()

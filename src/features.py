from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import settings

# Raw fields the model is allowed to see. Anything recorded after a service
# event (cost, downtime, severity, predictive score) is deliberately absent.
MAINTENANCE_NUMERIC = [
    "usage_hours",
    "load_capacity",
    "actual_load",
    "engine_temperature",
    "tire_pressure",
    "fuel_consumption",
    "battery_status",
    "vibration_levels",
    "oil_quality",
    "failure_history",
    "anomalies_detected",
    "diagnostic_trouble_code_count",
    "can_message_rate_hz",
    "sensor_packet_loss_rate",
    "historical_maintenance_cost",
    "vehicle_age",
    "usage_per_year",
    "load_ratio",
    "temp_delta",
    "vibration_per_1k_hours",
    "dtc_per_10k_hours",
    "packet_loss_pct",
    "maintenance_cost_per_hour",
    "brake_wear_flag",
    "failure_flag",
    "anomaly_flag",
]

MAINTENANCE_CATEGORICAL = [
    "make_and_model",
    "vehicle_type",
    "route_info",
    "weather_conditions",
    "road_conditions",
    "communication_interface",
    "telematics_gateway",
    "edge_device_class",
    "maintenance_type",
    "brake_condition",
]

MAINTENANCE_FEATURES = MAINTENANCE_NUMERIC + MAINTENANCE_CATEGORICAL

MAINTENANCE_RAW_COLUMNS = [
    "make_and_model",
    "vehicle_type",
    "year_of_manufacture",
    "route_info",
    "weather_conditions",
    "road_conditions",
    "communication_interface",
    "telematics_gateway",
    "edge_device_class",
    "usage_hours",
    "load_capacity",
    "actual_load",
    "engine_temperature",
    "tire_pressure",
    "fuel_consumption",
    "battery_status",
    "vibration_levels",
    "oil_quality",
    "failure_history",
    "anomalies_detected",
    "diagnostic_trouble_code_count",
    "can_message_rate_hz",
    "sensor_packet_loss_rate",
    "historical_maintenance_cost",
    "maintenance_type",
    "brake_condition",
]

# These columns are present in the CSV but cannot be used as model inputs.
# Keeping the list in code makes the leakage audit testable.
MAINTENANCE_EXCLUDED_COLUMNS = {
    "maintenance_severity",
    "maintenance_severity_id",
    "predictive_score",
    "maintenance_cost",
    "downtime_maintenance",
    "impact_on_efficiency",
    "days_since_last_maintenance",
    "last_maintenance_date",
    "pre_event_record",
    "data_split",
    "pcr",
    "uir",
    "tpi",
    "mbf",
    "ads",
    "ohi",
    "cmes",
    "uer",
}

FREIGHT_NUMERIC = [
    "weight_kg",
    "distance_miles",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "weight_per_mile",
]

FREIGHT_CATEGORICAL = [
    "origin_warehouse",
    "destination",
    "carrier",
    "lane",
    "distance_band",
    "weight_band",
]

FREIGHT_FEATURES = FREIGHT_NUMERIC + FREIGHT_CATEGORICAL

ERROR_CODE_RE = re.compile(r"\bERR[-_ ]?(\d{2,4})\b", re.IGNORECASE)
PART_NUMBER_RE = re.compile(
    r"\b(?:P/N|PART(?:\s*NO\.?)?|SKU)[-:\s#]*([A-Z0-9][A-Z0-9-]{3,})\b",
    re.IGNORECASE,
)
MILEAGE_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{4,7})\s*(?:mi|miles|km)\b", re.IGNORECASE)

URGENCY_TERMS = {
    "brake": ("brake", "brakes", "pad", "rotor"),
    "overheat": ("overheat", "overheating", "temperature", "coolant"),
    "smoke": ("smoke", "burning"),
    "leak": ("leak", "leaking", "fluid"),
    "vibration": ("vibration", "shaking", "wobble"),
    "transmission": ("transmission", "gearbox", "shifting"),
    "electrical": ("electrical", "battery", "alternator", "wiring"),
    "tire": ("tire", "tyre", "puncture", "tread"),
}

URGENCY_WEIGHTS = {
    "smoke": 3,
    "brake": 3,
    "overheat": 2,
    "leak": 2,
    "transmission": 2,
    "vibration": 1,
    "electrical": 1,
    "tire": 1,
}


def _num(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _text(df: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    if column in df.columns:
        values = df[column].astype("string")
    else:
        values = pd.Series(pd.NA, index=df.index, dtype="string")
    return values.fillna(default).str.strip().replace("", default)


def prepare_maintenance_features(
    frame: pd.DataFrame, reference_year: int | None = None
) -> pd.DataFrame:
    """Turn raw maintenance records into the exact model input frame.

    The same function is used by training and by the API, which is what keeps
    train/serve feature definitions from drifting apart.
    """
    ref_year = reference_year or settings.reference_year
    out = pd.DataFrame(index=frame.index)

    for column in MAINTENANCE_NUMERIC:
        if column not in {
            "vehicle_age",
            "usage_per_year",
            "load_ratio",
            "temp_delta",
            "vibration_per_1k_hours",
            "dtc_per_10k_hours",
            "packet_loss_pct",
            "maintenance_cost_per_hour",
            "brake_wear_flag",
            "failure_flag",
            "anomaly_flag",
        }:
            out[column] = _num(frame, column)

    vehicle_age = (ref_year - _num(frame, "year_of_manufacture", ref_year)).clip(0, 45)
    usage_hours = out["usage_hours"]
    out["vehicle_age"] = vehicle_age
    out["usage_per_year"] = usage_hours / vehicle_age.clip(lower=1)
    out["load_ratio"] = (out["actual_load"] / out["load_capacity"].replace(0, np.nan)).clip(0, 3)
    out["temp_delta"] = out["engine_temperature"] - 90.0
    out["vibration_per_1k_hours"] = out["vibration_levels"] / (usage_hours / 1000).clip(lower=0.1)
    out["dtc_per_10k_hours"] = out["diagnostic_trouble_code_count"] / (usage_hours / 10000).clip(
        lower=0.1
    )
    out["packet_loss_pct"] = out["sensor_packet_loss_rate"] * 100.0
    out["maintenance_cost_per_hour"] = out["historical_maintenance_cost"] / usage_hours.clip(
        lower=1
    )
    out["brake_wear_flag"] = (_text(frame, "brake_condition").str.lower() == "poor").astype(float)
    out["failure_flag"] = (out["failure_history"] > 0).astype(float)
    out["anomaly_flag"] = (out["anomalies_detected"] > 0).astype(float)

    for column in MAINTENANCE_CATEGORICAL:
        out[column] = _text(frame, column)

    return out[MAINTENANCE_FEATURES].astype(dict.fromkeys(MAINTENANCE_NUMERIC, "float64"))


def prepare_freight_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build booking-time features for a shipment.

    Delivery date, status and actual transit time are excluded because they are
    only known after the shipment has happened.
    """
    out = pd.DataFrame(index=frame.index)
    out["weight_kg"] = _num(frame, "weight_kg")
    out["distance_miles"] = _num(frame, "distance_miles")

    shipment_date = pd.to_datetime(
        frame["shipment_date"]
        if "shipment_date" in frame.columns
        else pd.Series(pd.NaT, index=frame.index),
        errors="coerce",
    )
    out["month"] = shipment_date.dt.month.astype("float64")
    out["day_of_week"] = shipment_date.dt.day_of_week.astype("float64")
    out["day_of_year"] = shipment_date.dt.day_of_year.astype("float64")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("float64")
    out["weight_per_mile"] = out["weight_kg"] / out["distance_miles"].replace(0, np.nan)

    out["origin_warehouse"] = _text(frame, "origin_warehouse")
    out["destination"] = _text(frame, "destination")
    out["carrier"] = _text(frame, "carrier")
    out["lane"] = out["origin_warehouse"] + "->" + out["destination"]

    distance_bins = [0, 250, 500, 1000, 1500, 2500, np.inf]
    distance_labels = ["<250", "250-500", "500-1000", "1000-1500", "1500-2500", "2500+"]
    out["distance_band"] = pd.cut(
        out["distance_miles"], bins=distance_bins, labels=distance_labels, right=False
    ).astype("string")
    weight_bins = [0, 10, 25, 50, 100, np.inf]
    weight_labels = ["<10", "10-25", "25-50", "50-100", "100+"]
    out["weight_band"] = pd.cut(
        out["weight_kg"], bins=weight_bins, labels=weight_labels, right=False
    ).astype("string")

    for column in FREIGHT_CATEGORICAL:
        out[column] = out[column].fillna("unknown").astype("string")

    return out[FREIGHT_FEATURES].astype(dict.fromkeys(FREIGHT_NUMERIC, "float64"))


def parse_service_note(note: str | None) -> dict[str, Any]:
    """Extract structured signals from a free-text maintenance note."""
    if not note or not note.strip():
        return {
            "error_codes": [],
            "part_numbers": [],
            "mileage": None,
            "keywords": [],
            "urgency": "low",
            "urgency_score": 0,
        }

    text_value = note.strip()
    error_codes = [f"ERR-{code}" for code in ERROR_CODE_RE.findall(text_value)]
    part_numbers = [part.upper() for part in PART_NUMBER_RE.findall(text_value)]

    mileage = None
    mileage_match = MILEAGE_RE.search(text_value)
    if mileage_match:
        mileage = int(mileage_match.group(1).replace(",", ""))

    lowered = text_value.lower()
    keywords: list[str] = []
    score = 0
    for label, terms in URGENCY_TERMS.items():
        if any(term in lowered for term in terms):
            keywords.append(label)
            score += URGENCY_WEIGHTS[label]
    if error_codes:
        score += min(len(error_codes), 3)

    if score >= 4:
        urgency = "high"
    elif score >= 2:
        urgency = "medium"
    else:
        urgency = "low"

    return {
        "error_codes": sorted(set(error_codes)),
        "part_numbers": sorted(set(part_numbers)),
        "mileage": mileage,
        "keywords": keywords,
        "urgency": urgency,
        "urgency_score": score,
    }


def fetch_maintenance_frame(
    engine: Engine, split: str | None = None, limit: int | None = None
) -> pd.DataFrame:
    query = "SELECT * FROM maintenance_records"
    params: dict[str, Any] = {}
    if split:
        query += " WHERE data_split = :split"
        params["split"] = split
    if limit:
        query += " LIMIT :limit"
        params["limit"] = int(limit)
    return pd.read_sql(text(query), engine, params=params)


def fetch_shipment_frame(engine: Engine, limit: int | None = None) -> pd.DataFrame:
    query = "SELECT * FROM shipments WHERE cost IS NOT NULL"
    params: dict[str, Any] = {}
    if limit:
        query += " LIMIT :limit"
        params["limit"] = int(limit)
    return pd.read_sql(text(query), engine, params=params)


def segment_risk(engine: Engine) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT vehicle_type,
               COUNT(*) AS vehicles,
               AVG(maintenance_required) AS risk_rate,
               AVG(usage_hours) AS avg_usage_hours,
               AVG(historical_maintenance_cost) AS avg_historical_cost
        FROM maintenance_records
        GROUP BY vehicle_type
        ORDER BY risk_rate DESC, vehicles DESC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def monthly_maintenance(engine: Engine) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT substr(CAST(last_maintenance_date AS VARCHAR), 1, 7) AS month,
               COUNT(*) AS events,
               AVG(maintenance_required) AS risk_rate
        FROM maintenance_records
        WHERE last_maintenance_date IS NOT NULL
        GROUP BY substr(CAST(last_maintenance_date AS VARCHAR), 1, 7)
        ORDER BY month
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def lane_statistics(engine: Engine, min_shipments: int = 3) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT origin_warehouse,
               destination,
               COUNT(*) AS shipments,
               AVG(cost) AS avg_cost,
               AVG(distance_miles) AS avg_distance,
               AVG(transit_days) AS avg_transit_days
        FROM shipments
        WHERE cost IS NOT NULL
        GROUP BY origin_warehouse, destination
        HAVING COUNT(*) >= :min_shipments
        ORDER BY shipments DESC, avg_cost DESC
        LIMIT 40
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"min_shipments": min_shipments}).mappings().all()
    return [dict(row) for row in rows]


def lane_benchmark(engine: Engine, origin: str, destination: str) -> dict[str, Any] | None:
    query = text(
        """
        SELECT cost, distance_miles, transit_days
        FROM shipments
        WHERE cost IS NOT NULL
          AND origin_warehouse = :origin
          AND destination = :destination
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"origin": origin, "destination": destination}).mappings().all()
    if not rows:
        return None
    costs = pd.Series([row["cost"] for row in rows], dtype="float64")
    return {
        "origin": origin,
        "destination": destination,
        "shipments": len(rows),
        "median_cost": float(costs.median()),
        "p25_cost": float(costs.quantile(0.25)),
        "p75_cost": float(costs.quantile(0.75)),
        "avg_distance": float(
            pd.Series([row["distance_miles"] for row in rows], dtype="float64").mean()
        ),
    }


def carrier_statistics(engine: Engine) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT carrier,
               COUNT(*) AS shipments,
               AVG(cost) AS avg_cost,
               AVG(distance_miles) AS avg_distance,
               AVG(transit_days) AS avg_transit_days
        FROM shipments
        WHERE cost IS NOT NULL
        GROUP BY carrier
        ORDER BY avg_cost DESC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def brake_condition_stats(engine: Engine) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT brake_condition,
               COUNT(*) AS vehicles,
               AVG(maintenance_required) AS risk_rate,
               AVG(historical_maintenance_cost) AS avg_historical_cost
        FROM maintenance_records
        GROUP BY brake_condition
        ORDER BY risk_rate DESC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def validate_maintenance_frame(frame: pd.DataFrame) -> dict[str, Any]:
    missing = sorted(set(MAINTENANCE_RAW_COLUMNS) - set(frame.columns))
    target_bad = 0
    if "maintenance_required" in frame.columns:
        target_bad = int((~frame["maintenance_required"].isin([0, 1])).sum())
    duplicate_ids = (
        int(frame["vehicle_id"].duplicated().sum()) if "vehicle_id" in frame.columns else 0
    )
    return {
        "rows": len(frame),
        "missing_columns": missing,
        "invalid_target_rows": target_bad,
        "duplicate_vehicle_ids": duplicate_ids,
        "ok": not missing and target_bad == 0 and duplicate_ids == 0,
    }


def validate_shipment_frame(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"shipment_id", "origin_warehouse", "destination", "carrier", "shipment_date"}
    missing = sorted(required - set(frame.columns))
    distance = (
        pd.to_numeric(frame["distance_miles"], errors="coerce")
        if "distance_miles" in frame.columns
        else pd.Series(dtype="float64")
    )
    negative_distance = int((distance < 0).sum())
    duplicate_ids = (
        int(frame["shipment_id"].duplicated().sum()) if "shipment_id" in frame.columns else 0
    )
    return {
        "rows": len(frame),
        "missing_columns": missing,
        "negative_distance_rows": negative_distance,
        "duplicate_shipment_ids": duplicate_ids,
        "ok": not missing and negative_distance == 0 and duplicate_ids == 0,
    }

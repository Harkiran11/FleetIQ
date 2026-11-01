"""Read-only analytics used by the dashboard and the API.

Keeping the SQL here means the routers stay small and the queries can be tested
against a real SQLite or PostgreSQL engine without starting HTTP.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

MAINTENANCE_INPUT_COLUMNS = [
    "vehicle_id",
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

OPTION_COLUMNS = {
    "maintenance": {
        "make_and_model": "maintenance_records",
        "vehicle_type": "maintenance_records",
        "route_info": "maintenance_records",
        "weather_conditions": "maintenance_records",
        "road_conditions": "maintenance_records",
        "communication_interface": "maintenance_records",
        "telematics_gateway": "maintenance_records",
        "edge_device_class": "maintenance_records",
        "maintenance_type": "maintenance_records",
        "brake_condition": "maintenance_records",
    },
    "shipments": {
        "origin_warehouse": "shipments",
        "destination": "shipments",
        "carrier": "shipments",
    },
}


def overview(engine: Engine) -> dict[str, Any]:
    with engine.connect() as conn:
        maintenance = (
            conn.execute(
                text(
                    """
                    SELECT COUNT(*) AS total_vehicles,
                           AVG(maintenance_required) AS maintenance_rate,
                           AVG(usage_hours) AS avg_usage_hours,
                           AVG(engine_temperature) AS avg_engine_temperature,
                           AVG(historical_maintenance_cost) AS avg_historical_cost,
                           AVG(year_of_manufacture) AS avg_model_year,
                           COUNT(DISTINCT make_and_model) AS distinct_models,
                           SUM(CASE WHEN brake_condition = 'Poor' THEN 1 ELSE 0 END)
                               AS poor_brake_vehicles,
                           SUM(CASE WHEN failure_history >= 2 THEN 1 ELSE 0 END)
                               AS repeat_failure_vehicles
                    FROM maintenance_records
                    """
                )
            )
            .mappings()
            .one()
        )
        shipments = (
            conn.execute(
                text(
                    """
                    SELECT COUNT(*) AS total_shipments,
                           AVG(cost) AS avg_cost,
                           SUM(cost) AS total_spend,
                           AVG(distance_miles) AS avg_distance,
                           AVG(transit_days) AS avg_transit_days
                    FROM shipments
                    WHERE cost IS NOT NULL
                    """
                )
            )
            .mappings()
            .one()
        )
        status_rows = (
            conn.execute(
                text(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM shipments
                    GROUP BY status
                    ORDER BY count DESC
                    """
                )
            )
            .mappings()
            .all()
        )

    return {
        "maintenance": {key: _number(value) for key, value in maintenance.items()},
        "shipments": {key: _number(value) for key, value in shipments.items()},
        "shipment_status": [dict(row) for row in status_rows],
    }


def priority_vehicles(engine: Engine, limit: int = 20) -> list[dict[str, Any]]:
    """Return raw vehicle records ranked by an explainable risk heuristic.

    The model re-scores these rows; the heuristic only decides which rows are
    worth scoring first, which keeps the dashboard responsive on 250k records.
    """
    columns = ", ".join(MAINTENANCE_INPUT_COLUMNS)
    query = text(
        f"""
        SELECT {columns}, maintenance_required
        FROM maintenance_records
        ORDER BY (
            failure_history * 2.0
            + anomalies_detected * 1.5
            + diagnostic_trouble_code_count * 0.5
            + CASE WHEN brake_condition = 'Poor' THEN 3.0 ELSE 0.0 END
            + usage_hours / 20000.0
        ) DESC, usage_hours DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": int(limit)}).mappings().all()
    payload: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        actual = record.pop("maintenance_required", None)
        payload.append({"record": record, "actual_maintenance_required": actual})
    return payload


def vehicle_profile(engine: Engine, vehicle_id: int) -> dict[str, Any] | None:
    columns = ", ".join(MAINTENANCE_INPUT_COLUMNS)
    query = text(
        f"""
        SELECT {columns}, maintenance_required, maintenance_cost,
               downtime_maintenance, last_maintenance_date
        FROM maintenance_records
        WHERE vehicle_id = :vehicle_id
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"vehicle_id": int(vehicle_id)}).mappings().first()
    return dict(row) if row else None


def options(engine: Engine, limit: int = 60) -> dict[str, list[str]]:
    """Distinct categorical values used to populate the dashboard forms."""
    payload: dict[str, list[str]] = {}
    with engine.connect() as conn:
        for group in OPTION_COLUMNS.values():
            for column, table in group.items():
                rows = conn.execute(
                    text(
                        f"SELECT DISTINCT {column} FROM {table} "
                        f"WHERE {column} IS NOT NULL ORDER BY {column} LIMIT :limit"
                    ),
                    {"limit": int(limit)},
                ).scalars()
                payload[column] = [str(value) for value in rows]
    return payload


def shipment_sample(engine: Engine, limit: int = 12) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT shipment_id, origin_warehouse, destination, carrier,
               shipment_date, weight_kg, distance_miles, cost, status
        FROM shipments
        WHERE cost IS NOT NULL
        ORDER BY shipment_date DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": int(limit)}).mappings().all()
    return [dict(row) for row in rows]


def _number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 4)
    return value

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Float, Integer, delete, func, insert, select
from sqlalchemy.engine import Engine

from src.config import settings
from src.db import get_engine, init_db, session_scope
from src.db_models import IngestRun, MaintenanceRecord, Shipment
from src.features import validate_maintenance_frame, validate_shipment_frame

logger = logging.getLogger(__name__)

MAINTENANCE_TABLE_COLUMNS = [
    column.name
    for column in MaintenanceRecord.__table__.columns
    if column.name not in {"ingested_at", "source"}
]
SHIPMENT_TABLE_COLUMNS = [
    column.name for column in Shipment.__table__.columns if column.name != "ingested_at"
]

MAINTENANCE_INT_COLUMNS = {
    "vehicle_id",
    "year_of_manufacture",
    "failure_history",
    "anomalies_detected",
    "diagnostic_trouble_code_count",
    "days_since_last_maintenance",
    "maintenance_severity_id",
    "maintenance_required",
}
MAINTENANCE_DATE_COLUMNS = {"last_maintenance_date"}
MAINTENANCE_NUMERIC_DB_COLUMNS = [
    column.name
    for column in MaintenanceRecord.__table__.columns
    if isinstance(column.type, (Integer, Float)) and column.name not in MAINTENANCE_DATE_COLUMNS
]
SHIPMENT_INT_COLUMNS = {"transit_days"}
SHIPMENT_DATE_COLUMNS = {"shipment_date", "delivery_date"}
SHIPMENT_NUMERIC_DB_COLUMNS = [
    column.name
    for column in Shipment.__table__.columns
    if isinstance(column.type, (Integer, Float)) and column.name not in SHIPMENT_DATE_COLUMNS
]


def normalize_column_name(name: str) -> str:
    value = re.sub(r"[\s\-]+", "_", str(name).strip())
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", value)
    return value.lower().strip("_")


def _find_csv(explicit: Path | None, patterns: list[str], roots: list[Path]) -> Path:
    if explicit:
        candidate = settings.path(explicit)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Configured CSV does not exist: {candidate}")

    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return max(matches, key=lambda item: item.stat().st_size)
    raise FileNotFoundError(
        "Could not locate input CSV. Put files in data/raw/ or set the path in .env."
    )


def locate_sources() -> tuple[Path, Path]:
    roots = [settings.raw_data_path, settings.project_root / "archive", settings.project_root]
    maintenance = _find_csv(
        settings.maintenance_csv,
        ["*predictive_maintenance*.csv", "*maintenance*.csv"],
        roots,
    )
    shipments = _find_csv(
        settings.shipments_csv,
        ["*shipment*.csv"],
        roots,
    )
    return maintenance, shipments


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [normalize_column_name(column) for column in frame.columns]
    frame = frame.loc[:, ~frame.columns.str.startswith("unnamed")]
    return frame


def _clean_maintenance(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    for column in MAINTENANCE_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    for column in MAINTENANCE_DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date

    numeric_columns = MAINTENANCE_NUMERIC_DB_COLUMNS
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["vehicle_id", "maintenance_required"])
    frame["vehicle_id"] = frame["vehicle_id"].astype(int)
    frame["maintenance_required"] = frame["maintenance_required"].astype(int).clip(0, 1)

    missing_before = frame.isna().sum()
    missing_filled: dict[str, int] = {}
    for column in numeric_columns:
        if column in MAINTENANCE_INT_COLUMNS:
            fill_value = 0 if frame[column].isna().all() else int(frame[column].median())
            missing_filled[column] = int(frame[column].isna().sum())
            frame[column] = frame[column].fillna(fill_value).astype(int)
        else:
            missing_filled[column] = int(frame[column].isna().sum())
            frame[column] = frame[column].fillna(frame[column].median())

    text_columns = [
        column
        for column in MAINTENANCE_TABLE_COLUMNS
        if column not in numeric_columns and column not in MAINTENANCE_DATE_COLUMNS
    ]
    for column in text_columns:
        missing_filled[column] = int(frame[column].isna().sum())
        frame[column] = frame[column].astype("string").fillna("unknown").str.strip()

    report = {
        "rows": len(frame),
        "missing_filled": {key: value for key, value in missing_filled.items() if value},
        "missing_before": {
            key: int(value) for key, value in missing_before.items() if int(value) > 0
        },
        "target_rate": float(frame["maintenance_required"].mean()),
        "date_range": [
            str(frame["last_maintenance_date"].min()),
            str(frame["last_maintenance_date"].max()),
        ],
        "validation": validate_maintenance_frame(frame),
    }
    return frame, report


def _clean_shipments(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    for column in SHIPMENT_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    for column in SHIPMENT_DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date

    numeric_columns = SHIPMENT_NUMERIC_DB_COLUMNS
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(
        subset=["shipment_id", "origin_warehouse", "destination", "carrier", "shipment_date"]
    )
    frame["shipment_id"] = frame["shipment_id"].astype(str).str.strip()

    missing_before = frame.isna().sum()
    missing_filled: dict[str, int] = {}
    for column in ["weight_kg", "distance_miles"]:
        missing_filled[column] = int(frame[column].isna().sum())
        frame[column] = frame[column].fillna(frame[column].median())
    for column in ["transit_days"]:
        missing_filled[column] = int(frame[column].isna().sum())
        frame[column] = frame[column].fillna(0).astype(int)

    for column in ["origin_warehouse", "destination", "carrier", "status"]:
        missing_filled[column] = int(frame[column].isna().sum())
        frame[column] = frame[column].astype("string").fillna("unknown").str.strip()

    report = {
        "rows": len(frame),
        "missing_filled": {key: value for key, value in missing_filled.items() if value},
        "missing_before": {
            key: int(value) for key, value in missing_before.items() if int(value) > 0
        },
        "rows_without_cost": int(frame["cost"].isna().sum()),
        "avg_cost": float(frame["cost"].mean()),
        "date_range": [str(frame["shipment_date"].min()), str(frame["shipment_date"].max())],
        "validation": validate_shipment_frame(frame),
    }
    return frame, report


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list)) else False:
        return None
    return value


def _records(frame: pd.DataFrame, columns: list[str], int_columns: set[str]) -> list[dict]:
    selected = frame[columns].copy()
    records: list[dict] = []
    for row in selected.to_dict("records"):
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, (list, dict)):
                cleaned[key] = value
            elif pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, pd.Timestamp):
                cleaned[key] = value.date()
            elif key in int_columns or isinstance(value, np.integer):
                cleaned[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def _insert_in_batches(engine: Engine, table, records: list[dict], batch_size: int = 4000) -> int:
    written = 0
    with engine.begin() as conn:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            conn.execute(insert(table), batch)
            written += len(batch)
    return written


def _table_count(engine: Engine, table) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def run_ingest(
    force: bool = False,
    sample: int | None = None,
    maintenance_csv: Path | None = None,
    shipments_csv: Path | None = None,
) -> dict[str, Any]:
    engine = get_engine()
    init_db(engine)
    maintenance_path = maintenance_csv or locate_sources()[0]
    shipments_path = shipments_csv or locate_sources()[1]

    logger.info("Reading maintenance data from %s", maintenance_path)
    maintenance_raw = _read_csv(maintenance_path)
    logger.info("Reading shipment data from %s", shipments_path)
    shipments_raw = _read_csv(shipments_path)

    if sample:
        maintenance_raw = maintenance_raw.sample(
            n=min(sample, len(maintenance_raw)), random_state=42
        ).reset_index(drop=True)
        shipments_raw = shipments_raw.sample(
            n=min(max(sample // 20, 100), len(shipments_raw)), random_state=42
        ).reset_index(drop=True)

    maintenance, maintenance_report = _clean_maintenance(maintenance_raw)
    shipments, shipments_report = _clean_shipments(shipments_raw)

    maintenance_count = _table_count(engine, MaintenanceRecord.__table__)
    shipment_count = _table_count(engine, Shipment.__table__)

    with session_scope() as session:
        run = IngestRun(
            source=f"{maintenance_path.name}, {shipments_path.name}",
            rows_read=len(maintenance_raw) + len(shipments_raw),
            status="running",
        )
        session.add(run)
        session.flush()
        run_id = run.id

    maintenance_written = 0
    shipment_written = 0

    if force or maintenance_count == 0:
        if maintenance_count:
            with engine.begin() as conn:
                conn.execute(delete(MaintenanceRecord.__table__))
        maintenance_records = _records(
            maintenance, MAINTENANCE_TABLE_COLUMNS, MAINTENANCE_INT_COLUMNS
        )
        maintenance_written = _insert_in_batches(
            engine, MaintenanceRecord.__table__, maintenance_records
        )
    else:
        logger.info(
            "Maintenance table already has %s rows; skipping (use --force)", maintenance_count
        )

    if force or shipment_count == 0:
        if shipment_count:
            with engine.begin() as conn:
                conn.execute(delete(Shipment.__table__))
        shipment_records = _records(shipments, SHIPMENT_TABLE_COLUMNS, SHIPMENT_INT_COLUMNS)
        shipment_written = _insert_in_batches(engine, Shipment.__table__, shipment_records)
    else:
        logger.info("Shipment table already has %s rows; skipping (use --force)", shipment_count)

    processed_dir = settings.processed_data_path
    processed_dir.mkdir(parents=True, exist_ok=True)
    maintenance.to_parquet(processed_dir / "maintenance.parquet", index=False)
    shipments.to_parquet(processed_dir / "shipments.parquet", index=False)

    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "maintenance": {
            **maintenance_report,
            "source": str(maintenance_path),
            "rows_written": maintenance_written,
        },
        "shipments": {
            **shipments_report,
            "source": str(shipments_path),
            "rows_written": shipment_written,
        },
        "database_url": settings.database_url,
        "total_rows": _table_count(engine, MaintenanceRecord.__table__)
        + _table_count(engine, Shipment.__table__),
    }
    report_path = processed_dir / "ingest_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    with session_scope() as session:
        run = session.get(IngestRun, run_id)
        if run:
            run.finished_at = datetime.now(UTC)
            run.rows_written = maintenance_written + shipment_written
            run.rows_rejected = (len(maintenance_raw) - len(maintenance)) + (
                len(shipments_raw) - len(shipments)
            )
            run.status = "success"
            run.details = _json_safe(
                {
                    "maintenance": maintenance_report,
                    "shipments": shipments_report,
                    "rows_written": maintenance_written + shipment_written,
                }
            )
            run.notes = f"report={report_path}"

    logger.info(
        "Ingest finished: maintenance=%s shipments=%s", maintenance_written, shipment_written
    )
    return _json_safe(report)


def load_processed_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    maintenance_path = settings.processed_data_path / "maintenance.parquet"
    shipments_path = settings.processed_data_path / "shipments.parquet"
    if not maintenance_path.exists() or not shipments_path.exists():
        raise FileNotFoundError("Processed parquet files not found. Run `make ingest` first.")
    return pd.read_parquet(maintenance_path), pd.read_parquet(shipments_path)

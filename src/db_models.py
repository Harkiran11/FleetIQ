from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class MaintenanceRecord(Base):
    __tablename__ = "maintenance_records"

    vehicle_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    make_and_model: Mapped[str] = mapped_column(String(64), index=True)
    vehicle_type: Mapped[str] = mapped_column(String(32), index=True)
    year_of_manufacture: Mapped[int] = mapped_column(Integer)
    route_info: Mapped[str] = mapped_column(String(48))
    weather_conditions: Mapped[str] = mapped_column(String(24))
    road_conditions: Mapped[str] = mapped_column(String(24))
    communication_interface: Mapped[str | None] = mapped_column(String(48))
    telematics_gateway: Mapped[str | None] = mapped_column(String(48))
    edge_device_class: Mapped[str | None] = mapped_column(String(48))

    usage_hours: Mapped[float] = mapped_column(Float)
    load_capacity: Mapped[float] = mapped_column(Float)
    actual_load: Mapped[float] = mapped_column(Float)
    engine_temperature: Mapped[float] = mapped_column(Float)
    tire_pressure: Mapped[float] = mapped_column(Float)
    fuel_consumption: Mapped[float] = mapped_column(Float)
    battery_status: Mapped[float] = mapped_column(Float)
    vibration_levels: Mapped[float] = mapped_column(Float)
    oil_quality: Mapped[float] = mapped_column(Float)

    failure_history: Mapped[int] = mapped_column(Integer)
    anomalies_detected: Mapped[int] = mapped_column(Integer)
    diagnostic_trouble_code_count: Mapped[int] = mapped_column(Integer)
    can_message_rate_hz: Mapped[float] = mapped_column(Float)
    sensor_packet_loss_rate: Mapped[float] = mapped_column(Float)

    maintenance_cost: Mapped[float] = mapped_column(Float)
    historical_maintenance_cost: Mapped[float] = mapped_column(Float)
    downtime_maintenance: Mapped[float] = mapped_column(Float)
    impact_on_efficiency: Mapped[float] = mapped_column(Float)
    days_since_last_maintenance: Mapped[int] = mapped_column(Integer)
    last_maintenance_date: Mapped[date | None] = mapped_column(Date, index=True)
    maintenance_type: Mapped[str] = mapped_column(String(48), index=True)
    brake_condition: Mapped[str] = mapped_column(String(16))
    maintenance_severity: Mapped[str | None] = mapped_column(String(32))
    maintenance_severity_id: Mapped[int | None] = mapped_column(Integer)
    maintenance_required: Mapped[int] = mapped_column(Integer, index=True)
    predictive_score: Mapped[float | None] = mapped_column(Float)
    data_split: Mapped[str | None] = mapped_column(String(16), index=True)

    source: Mapped[str] = mapped_column(String(64), default="logistics_predictive_maintenanceV2")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Shipment(Base):
    __tablename__ = "shipments"

    shipment_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    origin_warehouse: Mapped[str] = mapped_column(String(48), index=True)
    destination: Mapped[str] = mapped_column(String(48), index=True)
    carrier: Mapped[str] = mapped_column(String(48), index=True)
    shipment_date: Mapped[date] = mapped_column(Date, index=True)
    delivery_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    weight_kg: Mapped[float] = mapped_column(Float)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(24))
    distance_miles: Mapped[float] = mapped_column(Float)
    transit_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PredictionEvent(Base):
    __tablename__ = "prediction_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    endpoint: Mapped[str] = mapped_column(String(48), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32))
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request: Mapped[dict] = mapped_column(JSON)
    response: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    client: Mapped[str | None] = mapped_column(String(64), nullable=True)


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source: Mapped[str] = mapped_column(String(128))
    rows_read: Mapped[int] = mapped_column(Integer, default=0)
    rows_written: Mapped[int] = mapped_column(Integer, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default="running")
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

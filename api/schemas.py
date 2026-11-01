"""Pydantic request and response contracts for the FleetIQ API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    name: str
    version: str
    run_id: str
    source: str
    trained_at: str | None = None


class MaintenanceRequest(BaseModel):
    """Raw telemetry for one vehicle, at or before the service decision."""

    model_config = ConfigDict(extra="forbid")

    vehicle_id: int | None = Field(default=None, description="Fleet identifier (optional)")
    make_and_model: str = "Volvo FH"
    vehicle_type: str = "Heavy Truck"
    year_of_manufacture: int = Field(default=2017, ge=1980, le=2035)
    route_info: str = "Highway Freight"
    weather_conditions: str = "Clear"
    road_conditions: str = "Moderate"
    communication_interface: str = "CAN-FD/Telematics"
    telematics_gateway: str = "Edge_Telematics_Box"
    edge_device_class: str = "NVIDIA_Jetson_Nano"

    usage_hours: float = Field(default=12000.0, ge=0, le=200000)
    load_capacity: float = Field(default=12000.0, gt=0, le=100000)
    actual_load: float = Field(default=8200.0, ge=0, le=100000)
    engine_temperature: float = Field(default=94.0, ge=-40, le=200)
    tire_pressure: float = Field(default=34.0, ge=0, le=120)
    fuel_consumption: float = Field(default=9.6, ge=0, le=100)
    battery_status: float = Field(default=78.0, ge=0, le=100)
    vibration_levels: float = Field(default=2.1, ge=0, le=100)
    oil_quality: float = Field(default=80.0, ge=0, le=100)

    failure_history: int = Field(default=1, ge=0, le=100)
    anomalies_detected: int = Field(default=1, ge=0, le=100)
    diagnostic_trouble_code_count: int = Field(default=2, ge=0, le=100)
    can_message_rate_hz: float = Field(default=500.0, ge=0, le=10000)
    sensor_packet_loss_rate: float = Field(default=0.02, ge=0, le=1)
    historical_maintenance_cost: float = Field(default=1200.0, ge=0, le=1_000_000)

    maintenance_type: str = "Routine Inspection"
    brake_condition: Literal["Good", "Fair", "Poor", "unknown"] = "Good"
    service_note: str | None = Field(
        default=None,
        max_length=2000,
        description="Free-text note from the driver or service desk",
    )


class ParsedNote(BaseModel):
    error_codes: list[str] = []
    part_numbers: list[str] = []
    mileage: int | None = None
    keywords: list[str] = []
    urgency: Literal["low", "medium", "high"] = "low"
    urgency_score: int = 0


class RiskInfo(BaseModel):
    probability: float
    level: Literal["low", "moderate", "high", "critical"]
    threshold: float
    maintenance_required: bool


class Signal(BaseModel):
    feature: str
    label: str
    value: Any = None
    impact: float
    direction: str
    detail: str


class MaintenanceResponse(BaseModel):
    status: str = "success"
    model: ModelInfo
    risk: RiskInfo
    recommended_action: str
    signals: list[Signal] = []
    service_note: ParsedNote | None = None
    prediction_id: str | None = None


class FreightRequest(BaseModel):
    """Booking-time shipment attributes. Post-delivery fields are not accepted."""

    model_config = ConfigDict(extra="forbid")

    shipment_id: str | None = None
    origin_warehouse: str = "Warehouse_CHI"
    destination: str = "Dallas"
    carrier: str = "UPS"
    shipment_date: date = Field(default_factory=date.today)
    weight_kg: float = Field(default=42.0, ge=0, le=50000)
    distance_miles: float = Field(default=960.0, ge=0, le=20000)


class CostInterval(BaseModel):
    low: float
    high: float
    coverage: float


class LaneBenchmark(BaseModel):
    origin: str
    destination: str
    shipments: int
    median_cost: float
    p25_cost: float
    p75_cost: float
    avg_distance: float


class FreightResponse(BaseModel):
    status: str = "success"
    model: ModelInfo
    estimated_cost_usd: float
    interval_usd: CostInterval
    cost_per_mile_usd: float | None = None
    quote_valid_until: str | None = None
    lane_benchmark: LaneBenchmark | None = None
    prediction_id: str | None = None


class BatchError(BaseModel):
    index: int
    error: str


class BatchResponse(BaseModel):
    count: int
    succeeded: int
    failed: int
    results: list[dict[str, Any]] = []
    errors: list[BatchError] = []


class FeedbackRequest(BaseModel):
    actual_value: float = Field(
        description="Observed outcome: 0/1 for maintenance, cost for freight"
    )
    note: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    event_id: str
    actual_value: float
    feedback_at: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    app: str
    version: str
    environment: str
    database: bool
    models: dict[str, bool]
    uptime_seconds: float


class ModelReloadResponse(BaseModel):
    status: str
    source: str
    loaded_at: str | None
    models: list[dict[str, Any]]
    errors: dict[str, str]

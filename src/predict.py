from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from src.features import parse_service_note, prepare_freight_features, prepare_maintenance_features
from src.registry import LoadedModel

FEATURE_LABELS = {
    "year_of_manufacture": "Model year",
    "vehicle_age": "Vehicle age",
    "usage_hours": "Usage hours",
    "usage_per_year": "Usage per year",
    "load_capacity": "Load capacity",
    "actual_load": "Actual load",
    "load_ratio": "Load utilisation",
    "engine_temperature": "Engine temperature",
    "temp_delta": "Engine temperature vs 90 C",
    "tire_pressure": "Tire pressure",
    "fuel_consumption": "Fuel consumption",
    "battery_status": "Battery health",
    "vibration_levels": "Vibration level",
    "vibration_per_1k_hours": "Vibration per 1k hours",
    "oil_quality": "Oil quality",
    "failure_history": "Failure history",
    "failure_flag": "Prior failure",
    "anomalies_detected": "Anomalies detected",
    "anomaly_flag": "Anomaly flag",
    "diagnostic_trouble_code_count": "Diagnostic codes",
    "dtc_per_10k_hours": "Diagnostic codes per 10k hours",
    "can_message_rate_hz": "CAN message rate",
    "sensor_packet_loss_rate": "Sensor packet loss",
    "packet_loss_pct": "Sensor packet loss (%)",
    "historical_maintenance_cost": "Historical maintenance cost",
    "maintenance_cost_per_hour": "Maintenance cost per hour",
    "brake_condition": "Brake condition",
    "brake_wear_flag": "Brake wear flag",
    "maintenance_type": "Last maintenance type",
    "make_and_model": "Make and model",
    "vehicle_type": "Vehicle type",
    "route_info": "Route",
    "weather_conditions": "Weather",
    "road_conditions": "Road conditions",
    "communication_interface": "Communication interface",
    "telematics_gateway": "Telematics gateway",
    "edge_device_class": "Edge device",
    "weight_kg": "Weight",
    "distance_miles": "Distance",
    "weight_per_mile": "Weight per mile",
    "carrier": "Carrier",
    "origin_warehouse": "Origin",
    "destination": "Destination",
    "lane": "Lane",
    "distance_band": "Distance band",
    "weight_band": "Weight band",
}

SIGNAL_SOURCE = {
    "vehicle_age": "year_of_manufacture",
    "usage_per_year": "usage_hours",
    "load_ratio": "actual_load",
    "temp_delta": "engine_temperature",
    "vibration_per_1k_hours": "vibration_levels",
    "dtc_per_10k_hours": "diagnostic_trouble_code_count",
    "packet_loss_pct": "sensor_packet_loss_rate",
    "maintenance_cost_per_hour": "historical_maintenance_cost",
    "brake_wear_flag": "brake_condition",
    "failure_flag": "failure_history",
    "anomaly_flag": "anomalies_detected",
}


def _model_info(model: LoadedModel) -> dict[str, Any]:
    return {
        "name": model.metadata.get("model_name", model.model_key),
        "version": model.version,
        "run_id": model.run_id,
        "source": model.source,
        "trained_at": model.metadata.get("trained_at"),
    }


def _risk_level(probability: float, threshold: float) -> str:
    """Translate a calibrated probability into an operational band.

    Using a ratio to the decision threshold breaks when the threshold sits near
    1.0, which is exactly what happens on a separable dataset. The bands below
    are anchored to the threshold but still readable for a human reviewer.
    """
    if probability >= threshold:
        return "critical"
    if probability >= threshold * 0.75:
        return "high"
    if probability >= threshold * 0.35:
        return "moderate"
    return "low"


def _action(level: str, brake_condition: str | None, note: dict[str, Any]) -> str:
    if note.get("urgency") == "high":
        return "Review the service note immediately and book a diagnostic bay before dispatch."
    if brake_condition and brake_condition.lower() == "poor":
        return "Hold the vehicle for a brake inspection before the next loaded trip."
    if level == "critical":
        return (
            "Schedule a service bay within 48 hours and inspect thermal, brake "
            "and drivetrain systems."
        )
    if level == "high":
        return (
            "Book an inspection within the next 7 days; avoid long-haul assignments until cleared."
        )
    if level == "moderate":
        return "Add the vehicle to the next scheduled service window and monitor telemetry."
    return "No immediate action. Keep the normal service interval and continue monitoring."


class MaintenancePredictor:
    def __init__(self, model: LoadedModel):
        self.model = model

    def _signals(self, raw: pd.DataFrame, features: pd.DataFrame) -> list[dict[str, Any]]:
        importance = {
            item["feature"]: float(item["importance"])
            for item in self.model.metadata.get("global_importance", [])
        }
        numeric_stats = self.model.reference_stats.get("numeric", {})
        categorical_stats = self.model.reference_stats.get("categorical", {})
        signals: list[dict[str, Any]] = []
        seen: dict[str, int] = {}

        for feature, weight in importance.items():
            if weight <= 0:
                continue
            source = SIGNAL_SOURCE.get(feature, feature)
            label = FEATURE_LABELS.get(source, source.replace("_", " ").title())
            value = features[feature].iloc[0] if feature in features.columns else None
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                continue

            if feature in categorical_stats:
                frequency = categorical_stats[feature].get("frequency", {})
                share = float(frequency.get(str(value), 0.0))
                if share >= 0.15:
                    continue
                impact = weight * (1.0 - share)
                detail = f"seen in {share * 100:.1f}% of the training fleet"
                direction = "uncommon category"
            else:
                stats = numeric_stats.get(feature)
                if not stats:
                    continue
                std = float(stats.get("std") or 0.0)
                mean = float(stats.get("mean") or 0.0)
                if std <= 0:
                    continue
                z_score = abs(float(value) - mean) / std
                if z_score < 1.0:
                    continue
                impact = weight * min(z_score, 5.0)
                direction = "above fleet median" if float(value) > mean else "below fleet median"
                detail = f"{z_score:.1f} standard deviations from the training median"

            if source in seen:
                previous = signals[seen[source]]
                if impact > previous["impact"]:
                    signals[seen[source]] = {
                        "feature": source,
                        "label": label,
                        "value": raw[source].iloc[0] if source in raw.columns else value,
                        "impact": round(impact, 4),
                        "direction": direction,
                        "detail": detail,
                    }
                continue

            seen[source] = len(signals)
            signals.append(
                {
                    "feature": source,
                    "label": label,
                    "value": raw[source].iloc[0] if source in raw.columns else value,
                    "impact": round(impact, 4),
                    "direction": direction,
                    "detail": detail,
                }
            )

        if not signals:
            for item in self.model.metadata.get("global_importance", [])[:3]:
                source = SIGNAL_SOURCE.get(item["feature"], item["feature"])
                signals.append(
                    {
                        "feature": source,
                        "label": FEATURE_LABELS.get(source, source.replace("_", " ").title()),
                        "value": raw[source].iloc[0] if source in raw.columns else None,
                        "impact": round(float(item["importance"]), 4),
                        "direction": "fleet-wide driver",
                        "detail": "top model driver, but this record is close to the fleet median",
                    }
                )

        return sorted(signals, key=lambda item: item["impact"], reverse=True)[:4]

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = pd.DataFrame([payload])
        features = prepare_maintenance_features(raw)
        probability = float(self.model.pipeline.predict_proba(features)[0, 1])
        threshold = self.model.threshold
        required = bool(probability >= threshold)
        level = _risk_level(probability, threshold)
        note = parse_service_note(payload.get("service_note"))
        brake_condition = payload.get("brake_condition")

        return {
            "status": "success",
            "model": _model_info(self.model),
            "risk": {
                "probability": round(probability, 6),
                "level": level,
                "threshold": round(threshold, 6),
                "maintenance_required": required,
            },
            "recommended_action": _action(level, brake_condition, note),
            "signals": self._signals(raw, features),
            "service_note": note if note["keywords"] or note["error_codes"] else None,
        }


class FreightPredictor:
    def __init__(self, model: LoadedModel):
        self.model = model

    def predict(
        self, payload: dict[str, Any], lane_benchmark: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raw = pd.DataFrame([payload])
        features = prepare_freight_features(raw)
        estimate = max(0.0, float(self.model.pipeline.predict(features)[0]))
        quantile = self.model.residual_quantiles.get("0.9", 0.0)
        low = max(0.0, estimate - quantile)
        high = estimate + quantile
        distance = float(payload.get("distance_miles") or 0.0)
        cost_per_mile = estimate / distance if distance > 0 else None

        shipment_date = pd.to_datetime(payload.get("shipment_date"), errors="coerce")
        valid_until = (
            (shipment_date + timedelta(days=7)).date().isoformat()
            if not pd.isna(shipment_date)
            else None
        )

        return {
            "status": "success",
            "model": _model_info(self.model),
            "estimated_cost_usd": round(estimate, 2),
            "interval_usd": {
                "low": round(low, 2),
                "high": round(high, 2),
                "coverage": 0.90,
            },
            "cost_per_mile_usd": round(cost_per_mile, 4) if cost_per_mile is not None else None,
            "quote_valid_until": valid_until,
            "lane_benchmark": lane_benchmark,
        }

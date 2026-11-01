# API Reference

Base URL locally: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`  
OpenAPI JSON: `http://localhost:8000/openapi.json`

All canonical routes are under `/api`. Compatibility aliases exist at `/predict`, `/predict/freight`, `/health`, `/models`, and `/metrics`; they are intentionally hidden from the schema.

---

## Conventions

- Requests and responses are JSON unless noted.
- Unknown request fields are rejected with `422`. This is deliberate: it prevents clients from sending leakage columns such as `maintenance_cost`.
- Batch endpoints accept up to `MAX_BATCH_SIZE` records (default 500). Larger batches return `413`.
- A missing model returns `503` for that model. `/api/health` stays `200`; `/api/ready` returns `503` until both models load.
- `POST /api/admin/reload` requires `X-Admin-Token` when `ADMIN_TOKEN` is set. Locally it is open.

---

## System

### `GET /api/health`

Liveness. Reports database connectivity, per-model availability, and uptime.

```bash
curl -s http://localhost:8000/api/health
```

```json
{
  "status": "ok",
  "app": "FleetIQ",
  "version": "1.0.0",
  "environment": "local",
  "database": true,
  "models": {"maintenance": true, "freight": true},
  "uptime_seconds": 79561.617
}
```

### `GET /api/ready`

Readiness. Returns `200` with `{"ready": true}` when both models are loaded, otherwise `503`.

### `GET /api/models`

Promoted model metadata, including metrics, threshold, feature counts, and global importance.

```bash
curl -s http://localhost:8000/api/models
```

Each item contains:

```json
{
  "key": "maintenance",
  "available": true,
  "source": "local",
  "name": "fleetiq-maintenance-risk",
  "version": "1",
  "run_id": "13a23254c21e4a20a3b542e756362793",
  "trained_at": "2026-10-31T03:21:38.688589+00:00",
  "threshold": 0.9999975252703134,
  "metrics": {"roc_auc": 1.0, "pr_auc": 1.0, "f1": 1.0},
  "global_importance": [{"feature": "maintenance_type", "importance": 0.5006}]
}
```

### `GET /api/models/{key}/card`

Returns the Markdown model card for `maintenance` or `freight`. Unknown keys return `404`.

```bash
curl -s http://localhost:8000/api/models/maintenance/card
```

### `POST /api/admin/reload`

Reloads bundles from disk or from the MLflow registry.

```bash
# local bundles
curl -s -X POST http://localhost:8000/api/admin/reload

# registry champion, with a token
curl -s -X POST 'http://localhost:8000/api/admin/reload?source=mlflow' \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

```json
{
  "status": "reloaded",
  "source": "local",
  "loaded_at": "2026-10-31T21:44:02.123456+00:00",
  "models": [{"key": "maintenance", "available": true}],
  "errors": {}
}
```

### `GET /api/metrics`

Prometheus text exposition. Includes app info, model availability, prediction event totals, and average latency by endpoint.

```bash
curl -s http://localhost:8000/api/metrics
```

```text
# HELP fleetiq_info Application metadata
# TYPE fleetiq_info gauge
fleetiq_info{app="FleetIQ",version="1.0.0",environment="local"} 1
fleetiq_model_available{model="maintenance"} 1
fleetiq_model_available{model="freight"} 1
fleetiq_prediction_events_total 4
fleetiq_prediction_latency_ms_avg{endpoint="/api/predict/maintenance"} 31.09
```

---

## Maintenance prediction

### `POST /api/predict/maintenance`

Scores one vehicle.

Every field is optional; the schema provides sensible defaults. The response includes a logged `prediction_id`.

```bash
curl -s http://localhost:8000/api/predict/maintenance \
  -H 'content-type: application/json' \
  -d '{
    "vehicle_id": 82468,
    "make_and_model": "Volvo FH",
    "vehicle_type": "Heavy Truck",
    "year_of_manufacture": 2015,
    "route_info": "Highway Freight",
    "weather_conditions": "Clear",
    "road_conditions": "Moderate",
    "usage_hours": 18400,
    "load_capacity": 10000,
    "actual_load": 10105,
    "engine_temperature": 124.3,
    "tire_pressure": 23.9,
    "fuel_consumption": 14.4,
    "battery_status": 43.9,
    "vibration_levels": 3.32,
    "oil_quality": 48.6,
    "failure_history": 10,
    "anomalies_detected": 7,
    "diagnostic_trouble_code_count": 8,
    "can_message_rate_hz": 60,
    "sensor_packet_loss_rate": 0.0196,
    "historical_maintenance_cost": 2946,
    "maintenance_type": "Electrical Repair",
    "brake_condition": "Poor",
    "service_note": "Driver reports brake vibration and an overheat smell near the wheel, ERR-431. Pads last replaced at 184,000 miles."
  }'
```

Response:

```json
{
  "status": "success",
  "model": {
    "name": "fleetiq-maintenance-risk",
    "version": "1",
    "run_id": "13a23254c21e4a20a3b542e756362793",
    "source": "local",
    "trained_at": "2026-10-31T03:21:38.688589+00:00"
  },
  "risk": {
    "probability": 1.0,
    "level": "critical",
    "threshold": 0.999998,
    "maintenance_required": true
  },
  "recommended_action": "Hold the vehicle for a brake inspection before the next loaded trip.",
  "signals": [
    {
      "feature": "maintenance_type",
      "label": "Last maintenance type",
      "value": "Electrical Repair",
      "impact": 0.5006,
      "direction": "fleet-wide driver",
      "detail": "top model driver, but this record is close to the fleet median"
    }
  ],
  "service_note": {
    "error_codes": ["ERR-431"],
    "part_numbers": [],
    "mileage": 184000,
    "keywords": ["brake", "vibration", "overheat"],
    "urgency": "high",
    "urgency_score": 5
  },
  "prediction_id": "d2dfd4dd-3a5d-46bf-84c8-c687bc221523"
}
```

Risk levels are anchored to the tuned threshold:

```text
critical  p >= threshold
high      p >= 0.75 * threshold
moderate  p >= 0.35 * threshold
low       otherwise
```

### `POST /api/predict/maintenance/batch`

Accepts a JSON array of maintenance requests.

```bash
curl -s http://localhost:8000/api/predict/maintenance/batch \
  -H 'content-type: application/json' \
  -d '[{"usage_hours": 12000}, {"usage_hours": 18000, "brake_condition": "Poor"}]'
```

```json
{
  "count": 2,
  "succeeded": 2,
  "failed": 0,
  "results": [{"status": "success"}, {"status": "success"}],
  "errors": []
}
```

A bad row is reported in `errors` with its index; the rest of the batch still succeeds. More than 500 rows returns `413`.

---

## Freight prediction

### `POST /api/predict/freight`

Quotes a shipment using booking-time attributes only.

```bash
curl -s http://localhost:8000/api/predict/freight \
  -H 'content-type: application/json' \
  -d '{
    "shipment_id": "SHIP-1001",
    "origin_warehouse": "Warehouse_CHI",
    "destination": "Dallas",
    "carrier": "UPS",
    "shipment_date": "2025-03-18",
    "weight_kg": 38.9,
    "distance_miles": 1225
  }'
```

```json
{
  "status": "success",
  "model": {
    "name": "fleetiq-freight-pricing",
    "version": "1",
    "run_id": "792c772cfb53478fab3cd443fcd5e50e",
    "source": "local",
    "trained_at": "2026-10-31T03:21:42.871600+00:00"
  },
  "estimated_cost_usd": 182.96,
  "interval_usd": {"low": 154.53, "high": 211.39, "coverage": 0.9},
  "cost_per_mile_usd": 0.1906,
  "quote_valid_until": "2026-11-07",
  "lane_benchmark": {
    "origin": "Warehouse_CHI",
    "destination": "Dallas",
    "shipments": 6,
    "median_cost": 156.16,
    "p25_cost": 124.19,
    "p75_cost": 195.98,
    "avg_distance": 1323.83
  },
  "prediction_id": "439e34a9-6a0f-4505-8bfc-72e136b78622"
}
```

`lane_benchmark` is `null` when the lane has fewer than two observed shipments.

### `POST /api/predict/freight/batch`

Same contract as the maintenance batch endpoint.

```bash
curl -s http://localhost:8000/api/predict/freight/batch \
  -H 'content-type: application/json' \
  -d '[{"origin_warehouse": "Warehouse_CHI", "destination": "Dallas"}, {"origin_warehouse": "Warehouse_LA", "destination": "New York"}]'
```

---

## Analytics

### `GET /api/analytics/overview`

Fleet and shipment KPIs plus the last 24 hours of prediction traffic.

```bash
curl -s http://localhost:8000/api/analytics/overview
```

```json
{
  "maintenance": {
    "total_vehicles": 250000,
    "maintenance_rate": 0.3,
    "avg_usage_hours": 11599.7704,
    "avg_engine_temperature": 93.2021,
    "avg_historical_cost": 765.3092,
    "avg_model_year": 2017.0015,
    "distinct_models": 8,
    "poor_brake_vehicles": 31189,
    "repeat_failure_vehicles": 35974
  },
  "shipments": {
    "total_shipments": 1959,
    "avg_cost": 205.1616,
    "total_spend": 401911.57,
    "avg_distance": 1278.5855,
    "avg_transit_days": 4.1884
  },
  "shipment_status": [
    {"status": "Delivered", "count": 1648},
    {"status": "Delayed", "count": 199}
  ],
  "predictions": {
    "window_hours": 24,
    "total": 4,
    "by_endpoint": [
      {"endpoint": "/api/predict/freight", "count": 2, "avg_latency_ms": 19.63}
    ]
  }
}
```

### `GET /api/analytics/segments`

Risk rate by vehicle type.

### `GET /api/analytics/maintenance-trend`

Monthly service volume and risk rate.

### `GET /api/analytics/lanes?min_shipments=2`

Busiest lanes with shipment count, average cost, and average distance.

### `GET /api/analytics/carriers`

Carrier cost comparison.

### `GET /api/analytics/brake-condition`

Risk rate by brake condition.

### `GET /api/analytics/lane-benchmark?origin=Warehouse_CHI&destination=Dallas`

Observed cost distribution for one lane. Returns `404` when the lane has no history.

```json
{
  "origin": "Warehouse_CHI",
  "destination": "Dallas",
  "shipments": 6,
  "median_cost": 156.16,
  "p25_cost": 124.19,
  "p75_cost": 195.98,
  "avg_distance": 1323.83
}
```

### `GET /api/analytics/options?limit=50`

Categorical values used by the dashboard forms.

### `GET /api/shipments/recent?limit=10`

Recent completed shipments with cost.

### `GET /api/vehicles/priority?limit=30`

Scores a sample of vehicles with the live model and returns the highest-risk queue.

```json
[
  {
    "vehicle_id": 200055,
    "make_and_model": "Isuzu NPR",
    "vehicle_type": "Light Truck",
    "usage_hours": 10038.5706,
    "brake_condition": "Poor",
    "failure_history": 10,
    "anomalies_detected": 7,
    "probability": 0.999998,
    "level": "critical",
    "threshold": 0.999998,
    "maintenance_required": true,
    "actual_maintenance_required": 1,
    "top_signal": "Last maintenance type",
    "recommended_action": "Hold the vehicle for a brake inspection before the next loaded trip."
  }
]
```

`actual_maintenance_required` is included because this is a portfolio demo with a labelled dataset. A production queue would not expose the ground truth.

### `GET /api/vehicles/{vehicle_id}`

Vehicle profile plus a live score. Returns `404` for an unknown ID.

```json
{
  "vehicle_id": 200055,
  "record": {"vehicle_id": 200055, "make_and_model": "Isuzu NPR"},
  "prediction": {"risk": {"probability": 0.999998, "level": "critical"}},
  "error": null
}
```

---

## Monitoring

### `GET /api/monitoring/drift?model=maintenance&window=500`

PSI drift report for the last N logged requests.

```bash
curl -s 'http://localhost:8000/api/monitoring/drift?model=maintenance&window=500'
```

```json
{
  "generated_at": "2026-11-01T01:41:30.226860+00:00",
  "window_limit": 500,
  "reports": {
    "maintenance": {
      "model": "maintenance",
      "model_version": "1",
      "window": 3,
      "min_window": 25,
      "status": "insufficient_data",
      "max_psi": 12.43439,
      "thresholds": {"watch": 0.1, "alert": 0.25},
      "features": [
        {"feature": "usage_hours", "kind": "numeric", "psi": 8.90175, "status": "alert"}
      ],
      "top_drifted": [],
      "prediction_stats": {"count": 3, "mean": 0.0, "max": 0.0}
    }
  }
}
```

Status is `stable`, `watch`, `alert`, or `insufficient_data`. A window below 25 requests is never reported as an alert.

### `GET /api/monitoring/traffic?hours=24`

Prediction volume, average latency, p95 latency, and errors by endpoint.

### `GET /api/monitoring/feedback?limit=500`

Accuracy from labelled predictions.

```json
{
  "labeled_events": 12,
  "maintenance": {"count": 8, "accuracy_at_0_5": 0.875},
  "freight": {"count": 4, "mae": 11.2, "mape": 6.8}
}
```

### `GET /api/monitoring/events?limit=50`

Most recent logged prediction events.

### `POST /api/feedback/{event_id}`

Attach the observed outcome to a prediction.

```bash
curl -s -X POST http://localhost:8000/api/feedback/d2dfd4dd-3a5d-46bf-84c8-c687bc221523 \
  -H 'content-type: application/json' \
  -d '{"actual_value": 1, "note": "Confirmed by the workshop"}'
```

```json
{
  "status": "recorded",
  "event_id": "d2dfd4dd-3a5d-46bf-84c8-c687bc221523",
  "actual_value": 1.0,
  "feedback_at": "2026-10-31T21:45:00+00:00"
}
```

For maintenance, `actual_value` is `0` or `1`. For freight, it is the observed cost. Unknown event IDs return `404`.

---

## Errors

| Status | Meaning |
| --- | --- |
| `404` | Unknown vehicle, lane, model card, or prediction event |
| `413` | Batch larger than `MAX_BATCH_SIZE` |
| `422` | Schema violation, including unknown fields |
| `503` | Model bundle missing or registry unavailable |

Example validation error:

```json
{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["body", "maintenance_cost"],
      "msg": "Extra inputs are not permitted",
      "input": 999
    }
  ]
}
```

The rejection is intentional. `maintenance_cost` is a post-service field and is not part of the inference contract.

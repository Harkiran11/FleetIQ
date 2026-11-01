from __future__ import annotations

from datetime import date


def test_health_reports_loaded_models(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database"] is True
    assert payload["models"] == {"maintenance": True, "freight": True}
    assert payload["uptime_seconds"] >= 0


def test_ready_probe(client):
    response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_models_endpoint_returns_metadata(client):
    response = client.get("/api/models")

    assert response.status_code == 200
    models = {item["key"]: item for item in response.json()}
    assert set(models) == {"maintenance", "freight"}
    assert all(item["available"] for item in models.values())
    assert models["maintenance"]["threshold"] > 0
    assert models["maintenance"]["metrics"]
    assert models["freight"]["global_importance"]


def test_model_card_endpoint(client):
    response = client.get("/api/models/maintenance/card")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "# FleetIQ Maintenance Risk" in response.text

    missing = client.get("/api/models/unknown/card")
    assert missing.status_code == 404


def test_analytics_overview(client):
    response = client.get("/api/analytics/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["maintenance"]["total_vehicles"] == 5000
    assert payload["shipments"]["total_shipments"] == 243
    assert "predictions" in payload
    assert payload["predictions"]["window_hours"] == 24


def test_analytics_options_and_lanes(client):
    options = client.get("/api/analytics/options")
    assert options.status_code == 200
    assert options.json()["brake_condition"]

    lanes = client.get("/api/analytics/lanes?min_shipments=2")
    assert lanes.status_code == 200
    assert lanes.json()

    segments = client.get("/api/analytics/segments")
    assert segments.status_code == 200
    assert segments.json()


def test_lane_benchmark_endpoint(client):
    lane = client.get("/api/analytics/lanes?min_shipments=2").json()[0]

    response = client.get(
        "/api/analytics/lane-benchmark",
        params={
            "origin": lane["origin_warehouse"],
            "destination": lane["destination"],
        },
    )

    assert response.status_code == 200
    benchmark = response.json()
    assert benchmark["shipments"] >= 2
    assert benchmark["p25_cost"] <= benchmark["median_cost"] <= benchmark["p75_cost"]

    missing = client.get(
        "/api/analytics/lane-benchmark",
        params={"origin": "Nowhere", "destination": "Elsewhere"},
    )
    assert missing.status_code == 404


def test_priority_queue_and_vehicle_profile(client):
    queue = client.get("/api/vehicles/priority?limit=3")
    assert queue.status_code == 200
    rows = queue.json()
    assert len(rows) == 3
    assert rows == sorted(rows, key=lambda row: row["probability"], reverse=True)
    assert all(row["vehicle_id"] for row in rows)

    vehicle_id = rows[0]["vehicle_id"]
    profile = client.get(f"/api/vehicles/{vehicle_id}")
    assert profile.status_code == 200
    payload = profile.json()
    assert payload["vehicle_id"] == vehicle_id
    assert payload["prediction"]["risk"]["probability"] >= 0
    assert payload["error"] is None

    missing = client.get("/api/vehicles/999999999")
    assert missing.status_code == 404


def test_predict_maintenance(client, maintenance_payload):
    response = client.post("/api/predict/maintenance", json=maintenance_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["risk"]["probability"] >= 0
    assert payload["risk"]["level"] in {"low", "moderate", "high", "critical"}
    assert payload["service_note"]["error_codes"] == ["ERR-431"]
    assert payload["prediction_id"]


def test_predict_maintenance_rejects_unknown_fields(client, maintenance_payload):
    payload = dict(maintenance_payload)
    payload["maintenance_cost"] = 999

    response = client.post("/api/predict/maintenance", json=payload)

    assert response.status_code == 422


def test_predict_maintenance_batch(client, maintenance_payload):
    response = client.post(
        "/api/predict/maintenance/batch",
        json=[maintenance_payload, maintenance_payload],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["succeeded"] == 2
    assert payload["failed"] == 0
    assert len(payload["results"]) == 2


def test_predict_maintenance_batch_rejects_oversized_payload(client, maintenance_payload):
    response = client.post(
        "/api/predict/maintenance/batch",
        json=[maintenance_payload] * 501,
    )

    assert response.status_code == 413


def test_predict_freight_includes_lane_benchmark(client):
    lane = client.get("/api/analytics/lanes?min_shipments=2").json()[0]
    payload = {
        "origin_warehouse": lane["origin_warehouse"],
        "destination": lane["destination"],
        "carrier": "DHL",
        "shipment_date": date.today().isoformat(),
        "weight_kg": 38.9,
        "distance_miles": 1225.0,
    }

    response = client.post("/api/predict/freight", json=payload)

    assert response.status_code == 200
    quote = response.json()
    assert quote["estimated_cost_usd"] >= 0
    assert quote["interval_usd"]["low"] <= quote["estimated_cost_usd"]
    assert quote["interval_usd"]["high"] >= quote["estimated_cost_usd"]
    assert quote["lane_benchmark"]["shipments"] >= 2
    assert quote["prediction_id"]


def test_predict_freight_without_lane_history(client, freight_payload):
    payload = dict(freight_payload)
    payload["origin_warehouse"] = "Warehouse_NOWHERE"
    payload["destination"] = "Nowhere"

    response = client.post("/api/predict/freight", json=payload)

    assert response.status_code == 200
    assert response.json()["lane_benchmark"] is None


def test_feedback_flow_and_event_log(client, maintenance_payload):
    prediction = client.post("/api/predict/maintenance", json=maintenance_payload).json()
    event_id = prediction["prediction_id"]

    feedback = client.post(
        f"/api/feedback/{event_id}",
        json={"actual_value": 1.0, "note": "Confirmed by the workshop"},
    )
    assert feedback.status_code == 200
    assert feedback.json()["event_id"] == event_id

    summary = client.get("/api/monitoring/feedback")
    assert summary.status_code == 200
    assert summary.json()["maintenance"]["count"] >= 1

    events = client.get("/api/monitoring/events?limit=50")
    assert events.status_code == 200
    assert any(event["id"] == event_id for event in events.json())

    missing = client.post("/api/feedback/not-a-real-id", json={"actual_value": 1.0})
    assert missing.status_code == 404


def test_monitoring_traffic_and_drift(client, maintenance_payload):
    client.post("/api/predict/maintenance", json=maintenance_payload)

    traffic = client.get("/api/monitoring/traffic?hours=24")
    assert traffic.status_code == 200
    assert traffic.json()["total"] >= 1

    drift = client.get("/api/monitoring/drift?model=maintenance&window=25")
    assert drift.status_code == 200
    payload = drift.json()
    assert payload["window_limit"] == 25
    assert payload["reports"]["maintenance"]["model"] == "maintenance"
    assert "features" in payload["reports"]["maintenance"]


def test_prometheus_metrics(client):
    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "fleetiq_info{" in body
    assert 'fleetiq_model_available{model="maintenance"} 1' in body
    assert "fleetiq_prediction_events_total{" in body


def test_admin_reload_without_token(client):
    response = client.post("/api/admin/reload")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "reloaded"
    assert payload["source"] == "local"
    assert all(item["available"] for item in payload["models"])


def test_dashboard_and_static_assets(client):
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "FleetIQ" in dashboard.text

    stylesheet = client.get("/static/styles.css")
    assert stylesheet.status_code == 200
    assert "--accent" in stylesheet.text

    script = client.get("/static/app.js")
    assert script.status_code == 200

    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 204


def test_api_index_and_compatibility_aliases(client, maintenance_payload, freight_payload):
    index = client.get("/api")
    assert index.status_code == 200
    assert index.json()["docs"] == "/docs"

    health = client.get("/health")
    assert health.status_code == 200

    models = client.get("/models")
    assert models.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200

    maintenance = client.post("/predict", json=maintenance_payload)
    assert maintenance.status_code == 200

    freight = client.post("/predict/freight", json=freight_payload)
    assert freight.status_code == 200


def test_openapi_schema_is_available(client):
    schema = client.get("/openapi.json")

    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/api/predict/maintenance" in paths
    assert "/api/predict/freight" in paths
    assert "/predict" not in paths

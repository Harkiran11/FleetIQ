"""Command line entry point for every FleetIQ operation.

Examples::

    python -m src.cli ingest --force
    python -m src.cli train --model all
    python -m src.cli predict-maintenance --demo
    python -m src.cli predict-freight --input quote.json
    python -m src.cli monitor --window 500
    python -m src.cli info
    python -m src.cli serve --reload
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import settings


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _read_payload(path: str | None, inline: str | None) -> dict[str, Any]:
    if inline:
        return json.loads(inline)
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


DEMO_MAINTENANCE = {
    "vehicle_id": 200055,
    "make_and_model": "Isuzu NPR",
    "vehicle_type": "Light Truck",
    "year_of_manufacture": 2018,
    "route_info": "Highway Freight",
    "weather_conditions": "Hot",
    "road_conditions": "Congested",
    "communication_interface": "OBD-II/CAN",
    "telematics_gateway": "Fleet_IoT_Node",
    "edge_device_class": "Automotive_ECU",
    "usage_hours": 10038.57,
    "load_capacity": 10000.0,
    "actual_load": 10105.3,
    "engine_temperature": 124.26,
    "tire_pressure": 23.88,
    "fuel_consumption": 14.4,
    "battery_status": 43.94,
    "vibration_levels": 3.32,
    "oil_quality": 48.59,
    "failure_history": 10,
    "anomalies_detected": 7,
    "diagnostic_trouble_code_count": 8,
    "can_message_rate_hz": 60.05,
    "sensor_packet_loss_rate": 0.0196,
    "historical_maintenance_cost": 2945.74,
    "maintenance_type": "Electrical Repair",
    "brake_condition": "Poor",
    "service_note": (
        "Driver reports brake vibration and an overheat smell near the wheel, ERR-431. "
        "Pads last replaced at 184,000 miles."
    ),
}

DEMO_FREIGHT = {
    "shipment_id": "SH10001",
    "origin_warehouse": "Warehouse_MIA",
    "destination": "Atlanta",
    "carrier": "DHL",
    "shipment_date": "2025-03-18",
    "weight_kg": 38.9,
    "distance_miles": 1225.0,
}


def _predict_maintenance(payload: dict[str, Any]) -> dict[str, Any]:
    from src.predict import MaintenancePredictor
    from src.registry import ModelRegistry

    registry = ModelRegistry()
    registry.load_all()
    return MaintenancePredictor(registry.get("maintenance")).predict(payload)


def _predict_freight(payload: dict[str, Any]) -> dict[str, Any]:
    from src.db import get_engine, init_db
    from src.features import lane_benchmark
    from src.predict import FreightPredictor
    from src.registry import ModelRegistry

    registry = ModelRegistry()
    registry.load_all()
    engine = get_engine()
    init_db(engine)
    benchmark = None
    if payload.get("origin_warehouse") and payload.get("destination"):
        benchmark = lane_benchmark(
            engine, str(payload["origin_warehouse"]), str(payload["destination"])
        )
    return FreightPredictor(registry.get("freight")).predict(payload, benchmark)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from src.ingest import run_ingest

    report = run_ingest(force=args.force, sample=args.sample)
    print(json.dumps(report, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from src.train import evaluate_gate, train_freight, train_maintenance

    log_to_mlflow = not args.no_mlflow
    promote = not args.no_promote
    results: dict[str, Any] = {}
    if args.model in {"all", "maintenance"}:
        results["maintenance"] = train_maintenance(
            sample=args.sample, promote=promote, log_to_mlflow=log_to_mlflow
        )
    if args.model in {"all", "freight"}:
        results["freight"] = train_freight(
            sample=args.sample, promote=promote, log_to_mlflow=log_to_mlflow
        )
    if len(results) == 2:
        evaluate_gate(results)
    print(json.dumps(results, indent=2, default=str))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from src.registry import ModelRegistry
    from src.train import evaluate_gate

    registry = ModelRegistry()
    registry.load_all()
    results = {
        item["key"]: {"metrics": item.get("metrics", {})}
        for item in registry.info()
        if item.get("available")
    }
    evaluate_gate(results)
    print(json.dumps({"status": "pass", "models": results}, indent=2))
    return 0


def _cmd_predict_maintenance(args: argparse.Namespace) -> int:
    payload = _read_payload(args.input, args.payload) or DEMO_MAINTENANCE
    print(json.dumps(_predict_maintenance(payload), indent=2, default=str))
    return 0


def _cmd_predict_freight(args: argparse.Namespace) -> int:
    payload = _read_payload(args.input, args.payload) or DEMO_FREIGHT
    print(json.dumps(_predict_freight(payload), indent=2, default=str))
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    from src.db import get_session_factory, init_db
    from src.monitoring import drift_from_events, feedback_summary, prediction_summary
    from src.registry import ModelRegistry

    init_db()
    registry = ModelRegistry()
    registry.load_all()
    session = get_session_factory()()
    try:
        reports = {}
        for key in ("maintenance", "freight"):
            try:
                reports[key] = drift_from_events(registry.get(key), session, window=args.window)
            except Exception as exc:
                reports[key] = {"model": key, "status": "model_unavailable", "error": str(exc)}
        output = {
            "reports": reports,
            "traffic": prediction_summary(session),
            "feedback": feedback_summary(session),
        }
    finally:
        session.close()
    print(json.dumps(output, indent=2, default=str))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from sqlalchemy import func, select

    from src.db import get_engine, init_db
    from src.db_models import MaintenanceRecord, PredictionEvent, Shipment
    from src.registry import ModelRegistry
    from src.storage import local_inventory

    engine = get_engine()
    init_db(engine)
    with engine.connect() as conn:
        maintenance_rows = int(
            conn.execute(select(func.count()).select_from(MaintenanceRecord)).scalar_one()
        )
        shipment_rows = int(conn.execute(select(func.count()).select_from(Shipment)).scalar_one())
        prediction_rows = int(
            conn.execute(select(func.count()).select_from(PredictionEvent)).scalar_one()
        )
    registry = ModelRegistry()
    registry.load_all()
    payload = {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "env": settings.app_env,
        },
        "database": {
            "url": settings.database_url,
            "maintenance_rows": maintenance_rows,
            "shipment_rows": shipment_rows,
            "prediction_events": prediction_rows,
        },
        "models": registry.info(),
        "local_inventory": local_inventory(),
        "s3": {"bucket": settings.s3_bucket, "prefix": settings.s3_prefix},
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def _cmd_s3(args: argparse.Namespace) -> int:
    from src.storage import list_remote, sync_models

    if args.direction == "list":
        print(json.dumps(list_remote(), indent=2))
        return 0
    keys = tuple(args.model) if args.model else ("maintenance", "freight")
    result = sync_models(direction=args.direction, keys=keys)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    from src.pipeline import main as pipeline_main

    argv: list[str] = ["--flow", args.flow]
    if args.sample:
        argv += ["--sample", str(args.sample)]
    if args.force:
        argv.append("--force-ingest")
    return pipeline_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fleetiq",
        description="FleetIQ MLOps command line",
    )
    parser.add_argument("--log-level", default=settings.log_level)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load CSVs into the warehouse")
    ingest.add_argument("--force", action="store_true", help="Replace existing rows")
    ingest.add_argument("--sample", type=int, default=None)
    ingest.set_defaults(func=_cmd_ingest)

    train = subparsers.add_parser("train", help="Train and register models")
    train.add_argument("--model", choices=["all", "maintenance", "freight"], default="all")
    train.add_argument("--sample", type=int, default=None)
    train.add_argument("--no-mlflow", action="store_true")
    train.add_argument("--no-promote", action="store_true")
    train.set_defaults(func=_cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Apply the quality gate to current bundles")
    evaluate.set_defaults(func=_cmd_evaluate)

    maintenance = subparsers.add_parser("predict-maintenance", help="Score one vehicle")
    maintenance.add_argument("--input", help="JSON file with a raw maintenance record")
    maintenance.add_argument("--payload", help="Inline JSON record")
    maintenance.add_argument("--demo", action="store_true", help="Use the built-in demo record")
    maintenance.set_defaults(func=_cmd_predict_maintenance)

    freight = subparsers.add_parser("predict-freight", help="Quote one shipment")
    freight.add_argument("--input", help="JSON file with a raw shipment record")
    freight.add_argument("--payload", help="Inline JSON record")
    freight.add_argument("--demo", action="store_true", help="Use the built-in demo shipment")
    freight.set_defaults(func=_cmd_predict_freight)

    monitor = subparsers.add_parser("monitor", help="Drift and feedback report")
    monitor.add_argument("--window", type=int, default=None)
    monitor.set_defaults(func=_cmd_monitor)

    info = subparsers.add_parser("info", help="Show database, model and storage status")
    info.set_defaults(func=_cmd_info)

    serve = subparsers.add_parser("serve", help="Run the FastAPI application")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    s3 = subparsers.add_parser("s3-sync", help="Mirror model bundles to or from S3")
    s3.add_argument("--direction", choices=["upload", "download", "list"], default="upload")
    s3.add_argument("--model", action="append", choices=["maintenance", "freight"])
    s3.set_defaults(func=_cmd_s3)

    pipeline = subparsers.add_parser("pipeline", help="Run a Prefect flow")
    pipeline.add_argument(
        "--flow", choices=["refresh", "ingest", "train", "monitor"], default="refresh"
    )
    pipeline.add_argument("--sample", type=int, default=None)
    pipeline.add_argument("--force", action="store_true")
    pipeline.set_defaults(func=_cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    try:
        return int(args.func(args))
    except Exception as exc:
        logging.getLogger("fleetiq").error("%s", exc)
        if settings.app_env == "local":
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

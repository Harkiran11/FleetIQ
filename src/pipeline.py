"""Prefect 3 orchestration for the FleetIQ pipeline.

Flows are thin wrappers around the plain-Python modules so the same code can be
run by Prefect, by the CLI, or directly in a notebook. That keeps orchestration
out of the business logic and makes the pipeline testable without a Prefect
server.

Typical use::

    python -m src.pipeline                 # full refresh (ingest -> train -> monitor)
    python -m src.pipeline --flow ingest   # one flow
    python -m src.pipeline --serve         # register cron deployments

A local Prefect UI is available with ``prefect server start``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Any

from prefect import flow, task

from src.config import settings
from src.db import get_session_factory, init_db
from src.ingest import run_ingest
from src.monitoring import drift_from_events
from src.registry import ModelRegistry
from src.train import train_all

logger = logging.getLogger(__name__)


@task(name="ingest-csv", retries=2, retry_delay_seconds=10)
def ingest_task(force: bool = False, sample: int | None = None) -> dict[str, Any]:
    return run_ingest(force=force, sample=sample)


@task(name="train-models", retries=1, retry_delay_seconds=30)
def train_task(
    sample: int | None = None,
    promote: bool = True,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    return train_all(sample=sample, promote=promote, log_to_mlflow=log_to_mlflow)


@task(name="write-monitoring-report")
def monitor_task(window: int | None = None) -> dict[str, Any]:
    registry = ModelRegistry()
    registry.load_all()
    session = get_session_factory()()
    try:
        reports = {}
        for key in ("maintenance", "freight"):
            try:
                model = registry.get(key)
            except Exception as exc:  # pragma: no cover - degraded mode
                reports[key] = {"model": key, "status": "model_unavailable", "error": str(exc)}
                continue
            reports[key] = drift_from_events(model, session, window=window)
    finally:
        session.close()

    output = settings.processed_data_path / "monitoring_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "report": str(output),
        "statuses": {key: value.get("status") for key, value in reports.items()},
    }


@flow(name="fleetiq-ingest", log_prints=True)
def ingest_flow(force: bool = False, sample: int | None = None) -> dict[str, Any]:
    return ingest_task(force=force, sample=sample)


@flow(name="fleetiq-train", log_prints=True)
def train_flow(
    sample: int | None = None,
    promote: bool = True,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    init_db()
    return train_task(sample=sample, promote=promote, log_to_mlflow=log_to_mlflow)


@flow(name="fleetiq-monitor", log_prints=True)
def monitor_flow(window: int | None = None) -> dict[str, Any]:
    return monitor_task(window=window)


@flow(name="fleetiq-refresh", log_prints=True)
def refresh_flow(
    sample: int | None = None,
    force_ingest: bool = False,
    promote: bool = True,
) -> dict[str, Any]:
    """End-to-end refresh used by the scheduled deployment."""
    ingest_result = ingest_task(force=force_ingest, sample=sample)
    train_result = train_task(sample=sample, promote=promote, log_to_mlflow=True)
    monitor_result = monitor_task()
    return {
        "ingest": {
            "maintenance": ingest_result["maintenance"]["rows_written"],
            "shipments": ingest_result["shipments"]["rows_written"],
        },
        "train": {
            key: {
                "run_id": value.get("run_id"),
                "version": value.get("version"),
                "metrics": value.get("metrics"),
            }
            for key, value in train_result.items()
        },
        "monitor": monitor_result,
    }


def serve_schedules() -> None:
    """Register cron deployments on the configured Prefect API."""
    from prefect.schedules import Cron

    refresh_flow.serve(
        name="nightly-refresh",
        schedule=Cron("0 6 * * *", timezone="America/Toronto"),
        tags=["fleetiq", "production"],
        parameters={"force_ingest": False, "promote": True},
    )
    monitor_flow.serve(
        name="hourly-drift-check",
        schedule=Cron("15 * * * *", timezone="America/Toronto"),
        tags=["fleetiq", "monitoring"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FleetIQ Prefect flows")
    parser.add_argument(
        "--flow",
        choices=["refresh", "ingest", "train", "monitor"],
        default="refresh",
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--serve", action="store_true", help="Register cron deployments")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.serve:
        serve_schedules()
        return 0

    if args.flow == "ingest":
        result = ingest_flow(force=args.force_ingest, sample=args.sample)
    elif args.flow == "train":
        result = train_flow(sample=args.sample, promote=not args.no_promote)
    elif args.flow == "monitor":
        result = monitor_flow(window=args.window)
    else:
        result = refresh_flow(
            sample=args.sample,
            force_ingest=args.force_ingest,
            promote=not args.no_promote,
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

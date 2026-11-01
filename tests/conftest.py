"""Shared pytest fixtures.

The test suite must not depend on the full 250k-row production database or on
the trained models in ``models/``. Before any project module is imported we
point the settings at the small fixtures created during development:

* ``data/test_fleetiq.db`` (5,000 maintenance rows, 250 shipments)
* ``models_test/`` (small promoted bundles with the current feature schema)

The database is copied to a throwaway directory so tests can write prediction
events and feedback without touching the fixture on disk.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DB = ROOT / "data" / "test_fleetiq.db"
WORK_DIR = Path(tempfile.mkdtemp(prefix="fleetiq-tests-"))
WORK_DB = WORK_DIR / "fleetiq-test.db"

if not FIXTURE_DB.exists():  # pragma: no cover - developer setup guard
    raise RuntimeError(
        f"Missing test database {FIXTURE_DB}. Recreate it with "
        "`python -m src.cli ingest --sample 5000` against a temporary DATABASE_URL."
    )

shutil.copyfile(FIXTURE_DB, WORK_DB)

os.environ["DATABASE_URL"] = f"sqlite:///{WORK_DB}"
os.environ["MODEL_DIR"] = str(ROOT / "models_test")
os.environ["DATA_DIR"] = str(ROOT / "data")
os.environ["APP_ENV"] = "test"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MLFLOW_ENABLED"] = "false"
os.environ["PREDICTION_LOG_ENABLED"] = "true"
os.environ["ADMIN_TOKEN"] = ""


@pytest.fixture(scope="session")
def engine():
    from src.db import get_engine, init_db

    value = get_engine()
    init_db(value)
    return value


@pytest.fixture()
def session(engine):
    from src.db import get_session_factory

    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def maintenance_model():
    from src.registry import load_bundle

    return load_bundle("maintenance")


@pytest.fixture(scope="session")
def freight_model():
    from src.registry import load_bundle

    return load_bundle("freight")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def maintenance_payload() -> dict:
    """A high-risk record used by the predictor and API tests."""
    from src.cli import DEMO_MAINTENANCE

    return dict(DEMO_MAINTENANCE)


@pytest.fixture()
def freight_payload() -> dict:
    return {
        "shipment_id": "TEST-1001",
        "origin_warehouse": "Warehouse_MIA",
        "destination": "Atlanta",
        "carrier": "DHL",
        "shipment_date": "2025-03-18",
        "weight_kg": 38.9,
        "distance_miles": 1225.0,
    }

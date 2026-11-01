#!/usr/bin/env sh
# Container entrypoint.
#
# The image already contains promoted model bundles, so the API can boot
# without a training run. When the relational warehouse is empty (for example
# on the first `docker compose up`) the raw CSVs mounted at /app/data/raw are
# ingested before Gunicorn starts. Ingestion failures are logged but do not
# stop the API: health and model metadata remain available for debugging.
set -eu

if [ "${FLEETIQ_SKIP_BOOTSTRAP:-0}" != "1" ]; then
    python - <<'PY'
import logging
import sys

from sqlalchemy import func, select

from src.db import get_engine, init_db
from src.db_models import MaintenanceRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fleetiq.bootstrap")

engine = get_engine()
init_db(engine)
with engine.connect() as connection:
    rows = int(
        connection.execute(
            select(func.count()).select_from(MaintenanceRecord)
        ).scalar_one()
    )

if rows:
    logger.info("Warehouse already contains %s maintenance rows; skipping ingest", rows)
    sys.exit(0)

try:
    from src.ingest import run_ingest

    logger.info("Empty warehouse detected; ingesting raw CSVs")
    run_ingest(force=False)
except Exception as error:  # pragma: no cover - container bootstrap guard
    logger.warning("Automatic ingest skipped: %s", error)
    logger.warning("Mount data/raw or run `docker compose run --rm init` to load data")
PY
fi

exec "$@"

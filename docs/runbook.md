# Operations Runbook

This is the practical guide for running FleetIQ: starting it, retraining it, rolling it back, and responding when the data moves.

---

## 1. Local service

### Start

```bash
source .venv/bin/activate
python -m src.cli serve --reload
```

The API is at <http://localhost:8000>, the dashboard at `/`, and OpenAPI at `/docs`.

### Stop

`Ctrl+C` in the terminal running the server.

### Check

```bash
curl -s http://localhost:8000/api/health | python -m json.tool
curl -s http://localhost:8000/api/ready
curl -s http://localhost:8000/api/models | python -m json.tool
```

Expected: `status: ok`, both models `true`, `ready: true`.

### Logs

The CLI configures structured logging through `src/logging_utils.py`. Set `LOG_JSON=true` for one-line JSON records suitable for a log shipper.

---

## 2. Docker Compose

### First start

```bash
docker compose up --build -d
docker compose logs -f api
```

The API entrypoint:

1. Creates the schema.
2. Checks `maintenance_records`.
3. Ingests `data/raw/` if the table is empty.
4. Starts Gunicorn.

The image contains the promoted bundles, so model serving does not wait for ingestion.

### Health

```bash
curl -s http://localhost:8000/api/health
docker compose ps
```

### Retrain inside the stack

```bash
docker compose run --rm init
```

This runs `ingest --force` and `train --model all` against PostgreSQL and the MLflow service, and writes bundles to the bind-mounted `./models`.

### Stop

```bash
docker compose down        # keep volumes
docker compose down -v     # delete Postgres and MLflow artifacts
```

### Skip automatic ingestion

Set `FLEETIQ_SKIP_BOOTSTRAP=1` on the API service if you want to load data separately.

---

## 3. Retraining

### Full retrain

```bash
python -m src.cli ingest --force
python -m src.cli train --model all
```

This creates new MLflow runs, registers new versions, and promotes them to `models/<key>/current/` when `AUTO_PROMOTE=true`.

### One model

```bash
python -m src.cli train --model maintenance
python -m src.cli train --model freight
```

### Without MLflow

```bash
python -m src.cli train --model all --no-mlflow
```

Local bundles and model cards are still written. Use this when the tracking server is unavailable.

### Sample run

```bash
python -m src.cli ingest --sample 5000 --force
python -m src.cli train --model all
```

`--sample` is deterministic. It is useful for smoke tests, not for production promotion.

### Verify the gate

Training raises if:

- Maintenance ROC AUC < `MIN_MAINTENANCE_AUC` (default 0.75)
- Freight MAPE > `MAX_FREIGHT_MAPE` (default 25.0)

Run the gate against the current bundles:

```bash
python -m src.cli evaluate
```

Expected output: `{"status": "pass", "models": {...}}`.

---

## 4. Promotion and rollback

### How promotion works

Each run writes `models/<key>/runs/<run_id>/`. Promotion:

1. Copies the run directory to `models/<key>/current/`.
2. Writes `models/<key>/current.json` with the run ID and timestamp.
3. Sets the MLflow `champion` alias when MLflow is enabled.

### Promote a specific local run

```bash
python - <<'PY'
from src.registry import promote_run

promote_run("maintenance", "13a23254c21e4a20a3b542e756362793")
PY
```

Then reload the API:

```bash
curl -s -X POST http://localhost:8000/api/admin/reload
```

### Roll back to a previous MLflow version

```python
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///./mlflow.db")
client.set_registered_model_alias("fleetiq-maintenance-risk", "champion", version="1")
```

Then reload with `source=mlflow`:

```bash
curl -s -X POST 'http://localhost:8000/api/admin/reload?source=mlflow'
```

### Confirm the active version

```bash
curl -s http://localhost:8000/api/models | python -m json.tool
```

Check `version`, `run_id`, and `trained_at`.

### Rollback checklist

- [ ] Identify the last known-good run ID or registry version.
- [ ] Promote the bundle or move the `champion` alias.
- [ ] Reload the API.
- [ ] Confirm `/api/models` shows the expected run ID.
- [ ] Send a known-good prediction and compare it to the expected output.
- [ ] Watch `/api/monitoring/traffic` and `/api/monitoring/feedback` for 24 hours.

---

## 5. Drift response

### Read the report

```bash
curl -s 'http://localhost:8000/api/monitoring/drift?model=maintenance&window=500' | python -m json.tool
```

Status meanings:

| Status | Action |
| --- | --- |
| `stable` | No action |
| `watch` | Inspect `top_drifted`. Check upstream data quality. |
| `alert` | Investigate immediately. Consider retraining or a rollback. |
| `insufficient_data` | Wait for at least 25 requests |

### Triage order

1. **Is the input real?** Check the raw request in `/api/monitoring/events`. A sudden PSI spike is often a bad client or a units change (psi vs bar, miles vs km).
2. **Is it one feature or many?** One feature suggests a broken integration. Many features suggest a population shift or a new lane/vehicle mix.
3. **Is accuracy falling?** Check `/api/monitoring/feedback`. Drift without accuracy loss is not automatically an incident.
4. **Is the model still calibrated?** If labelled feedback is available, compare accuracy and MAE to the model card.
5. **Retrain or roll back?** Retrain when the new distribution is the new normal. Roll back when the new model is worse than the old one on the same window.

### Useful SQL

SQLite:

```bash
sqlite3 data/fleetiq.db "
  SELECT endpoint, COUNT(*), ROUND(AVG(latency_ms), 2)
  FROM prediction_events
  WHERE created_at >= datetime('now', '-24 hours')
  GROUP BY endpoint;
"
```

PostgreSQL:

```sql
SELECT endpoint, COUNT(*), ROUND(AVG(latency_ms)::numeric, 2)
FROM prediction_events
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY endpoint;
```

Recent drift window:

```sql
SELECT id, created_at, model_name, model_version, risk_probability
FROM prediction_events
WHERE model_name = 'fleetiq-maintenance-risk'
ORDER BY created_at DESC
LIMIT 500;
```

### Feedback loop

```bash
curl -s -X POST http://localhost:8000/api/feedback/<event_id> \
  -H 'content-type: application/json' \
  -d '{"actual_value": 1, "note": "Confirmed by the workshop"}'
```

For freight, send the observed cost as `actual_value`. Labels are optional but they are the only way to measure live accuracy.

---

## 6. Data refresh

### Replace the raw files

1. Put the new CSVs in `data/raw/` with the same names, or set `MAINTENANCE_CSV` and `SHIPMENTS_CSV`.
2. Run `python -m src.cli ingest --force`.
3. Check `data/processed/ingest_report.json` for rejected rows and validation status.
4. Retrain if the schema or distribution changed.

### Schema changes

The loader adds missing columns as `NaN` and fills defaults, so a new column is tolerated. A renamed column is not: update `MAINTENANCE_TABLE_COLUMNS`, the SQLAlchemy model, and the feature contract together.

After a schema change:

```bash
python -m src.cli ingest --force
python -m src.cli train --model all
pytest tests -q
```

### Ingest report

```bash
cat data/processed/ingest_report.json
```

Look for:

- `validation.ok: false`
- `invalid_target_rows > 0`
- `duplicate_vehicle_ids > 0`
- `negative_distance_rows > 0`
- A large jump in `missing_filled`

---

## 7. Backup and recovery

### SQLite

Stop the API, then copy:

```bash
cp data/fleetiq.db data/fleetiq.db.bak
```

For a live database, use SQLite's backup command:

```bash
sqlite3 data/fleetiq.db ".backup 'data/fleetiq.db.bak'"
```

### PostgreSQL

```bash
docker compose exec postgres pg_dump -U fleetiq fleetiq > fleetiq.sql
```

Restore:

```bash
cat fleetiq.sql | docker compose exec -T postgres psql -U fleetiq -d fleetiq
```

### MLflow

The tracking store is `mlflow.db` (SQLite) or the Postgres database. Artifacts are in `mlruns/` locally or the `mlartifacts` volume in Compose.

Back up both the store and the artifacts. A registry version without its artifacts cannot be loaded.

### Model bundles

`models/<key>/current/` is self-contained. Copying that directory is enough to serve the model on another host. `models/<key>/runs/` is history. It is excluded from git but useful for rollback.

---

## 8. Troubleshooting

### `503` from a prediction endpoint

The model bundle is missing or failed to load.

```bash
curl -s http://localhost:8000/api/models | python -m json.tool
```

Look at `available` and `error`. Then:

```bash
ls -la models/maintenance/current
python -m src.cli info
```

### `ready` is 503 but `health` is 200

One model loaded and the other did not. This is intentional: the service can still serve the healthy model. Fix the missing bundle or reload.

### `422` on a prediction

The payload has an unknown field or an out-of-range value. Read the `loc` in the error. A common cause is sending `maintenance_cost` or a post-delivery freight field. Both are rejected by design.

### `413` on a batch

The batch exceeds `MAX_BATCH_SIZE` (default 500). Split the request or raise the limit in `.env`.

### MLflow model registration fails

- Check `MLFLOW_TRACKING_URI`.
- If using the registry, make sure the tracking server is reachable from the training process.
- The project uses `serialization_format="cloudpickle"` because skops rejects the histogram gradient boosting types. Do not switch to the default without testing.
- If the registry is unavailable, retrain with `--no-mlflow` and promote locally.

### Docker API cannot write models

The `init` service runs as root so it can write the bind-mounted `./models` directory. If files become root-owned on Linux:

```bash
sudo chown -R "$USER":"$USER" models
```

### Postgres connection refused

```bash
docker compose ps postgres
docker compose logs postgres
```

The API waits for the Postgres health check. If the password or database name changed, update both the service and `DATABASE_URL`.

### MLflow health check fails

```bash
docker compose logs mlflow
```

The MLflow server needs the Postgres backend to be ready. The Compose health check waits for it. If you changed the backend URL, restart both services.

### Frontend shows stale assets

The dashboard is served with normal caching. Hard-refresh, or append a query string:

```text
http://localhost:8000/?v=6
```

### Tests fail because fixtures are missing

The suite expects:

```text
data/test_fleetiq.db
models_test/maintenance/current/
models_test/freight/current/
```

These are small committed fixtures. If they are missing, regenerate them from the full data:

```bash
DATABASE_URL=sqlite:///./data/test_fleetiq.db \
  python -m src.cli ingest --sample 5000 --force
```

Then retrain with `MODEL_DIR=models_test` and promote the bundles. The test fixture is deterministic because `--sample` uses a fixed seed.

---

## 9. Security notes

- Set `ADMIN_TOKEN` in any shared environment. Without it, `/api/admin/reload` is open.
- The API does not implement authentication for prediction or analytics endpoints. Put it behind an authenticating proxy if it is exposed.
- CORS is restricted by `CORS_ORIGINS`. The default is local development.
- The container runs as a non-root user (UID 10001).
- Request bodies are bounded by the batch limit and Pydantic field constraints, but there is no global request-size limit. Add one at the proxy for public deployments.
- The feedback endpoint trusts the caller's label. In production, labels should come from an authenticated warehouse job.

---

## 10. On-call checklist

When something looks wrong:

1. `curl /api/health` and `/api/ready`.
2. `curl /api/models` and confirm the run IDs.
3. Check the last 20 events in `/api/monitoring/events`.
4. Check `/api/monitoring/traffic` for latency or error spikes.
5. Check `/api/monitoring/drift` for input movement.
6. Check `/api/monitoring/feedback` if labels exist.
7. Inspect the container logs.
8. If the model is the problem, roll back. If the data is the problem, fix the feed and retrain.
9. Record the incident: model version, window, PSI, accuracy, and the action taken.

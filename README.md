# FleetIQ MLOps

FleetIQ is an end-to-end MLOps reference project for a logistics operator. It scores every vehicle for maintenance risk, quotes freight lanes with a calibrated cost interval, logs every prediction, and watches its own inputs for drift.

The interesting part is not the model. It is the path from raw CSV to a promoted, monitored, reproducible service: ingestion, leakage-safe feature engineering, experiment tracking, a model registry, orchestration, a typed API, a dependency-free dashboard, tests, and container deployment.

**Stack:** Python 3.13 · scikit-learn · MLflow · Prefect · FastAPI · SQLAlchemy · SQLite/PostgreSQL · Docker · pytest · vanilla HTML/CSS/JS

---

## What it does

| Capability | Implementation |
| --- | --- |
| Warehouse | 250,000 maintenance records + 2,000 shipments in SQLite locally, PostgreSQL in Compose |
| Maintenance risk | Histogram gradient boosting classifier, tuned decision threshold, risk bands, explainability signals |
| Freight pricing | Histogram gradient boosting regressor with absolute-error loss and split-conformal 90% intervals |
| Experiment tracking | MLflow runs, metrics, params, plots, signatures, model registry versions and `champion` aliases |
| Orchestration | Prefect 3 flows for refresh, ingest, train, monitor, plus cron deployments |
| Serving | FastAPI, batch endpoints, Pydantic contracts, compatibility aliases, Gunicorn in Docker |
| Monitoring | PSI drift against stored training distributions, traffic/latency summaries, human feedback loop |
| Dashboard | Editorial operations console with no frontend framework or build step |
| Quality | 111 pytest tests, ruff lint/format, coverage, CI workflow, Docker health checks |

---

## Screenshots

| Overview | Maintenance risk |
| --- | --- |
| ![Overview dashboard](docs/images/overview.png) | ![Maintenance risk result](docs/images/maintenance.png) |

| Freight quote | Monitoring |
| --- | --- |
| ![Freight quote with interval and lane benchmark](docs/images/freight.png) | ![PSI drift and traffic monitoring](docs/images/monitoring.png) |

The dashboard is served by FastAPI at `/`; no Node build step is involved.

---

## Architecture

```text
                    ┌──────────────────────────────────────────────┐
                    │              data/raw/*.csv                  │
                    │  maintenance telemetry · shipment history    │
                    └──────────────────────┬───────────────────────┘
                                           │
                                    src/ingest.py
                         clean · validate · batch insert · report
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │        SQLite (local) / PostgreSQL (Docker)  │
                    │  maintenance_records · shipments             │
                    │  prediction_events · ingest_runs             │
                    └───────────┬──────────────────────┬───────────┘
                                │                      │
                         src/features.py          src/analytics.py
                    shared train/serve features   SQL summaries
                                │                      │
                         src/train.py                  │
              HistGradientBoosting + MLflow            │
              metrics · plots · registry               │
                                │                      │
                     models/<key>/current/             │
                                │                      │
                    ┌───────────▼──────────────────────▼───────────┐
                    │                api/main.py                   │
                    │  FastAPI · Pydantic · prediction logging     │
                    │  /api/predict · /api/analytics · /monitoring │
                    └───────────┬──────────────────────┬───────────┘
                                │                      │
                     frontend/ dashboard        Prometheus /metrics
                     (HTML/CSS/JS, no build)     drift + feedback
```

Prefect wraps the same functions the CLI calls, so a flow is never a second implementation of the pipeline.

---

## Results

These are the committed bundles in `models/`, trained on the full dataset. MLflow tracking lives in `mlflow.db`; the model registry holds both runs.

### Maintenance risk

| Metric | Value |
| --- | --- |
| MLflow run | `13a23254c21e4a20a3b542e756362793` |
| Registry version | `1` (alias `champion`) |
| Train / validation / test | 175,000 / 25,000 / 50,000 |
| ROC AUC | 1.0000 |
| PR AUC | 1.0000 |
| F1 / precision / recall | 1.0000 / 1.0000 / 1.0000 |
| Brier score | 2.63e-12 |
| Log loss | 1.49e-06 |
| Baseline ROC AUC / F1 | 0.5000 / 0.0000 |
| Test confusion (TN / FP / FN / TP) | 35,000 / 0 / 0 / 15,000 |
| Tuned threshold | 0.9999975253 |
| Features | 36 (26 numeric + 10 categorical) |

The public maintenance dataset is synthetic and encodes the target as a deterministic function of service state. Near-perfect metrics are expected, not evidence of production readiness. The pipeline deliberately drops `days_since_last_maintenance`, `maintenance_severity`, `predictive_score`, `maintenance_cost`, `downtime_maintenance`, `data_split`, and the derived reliability indices to avoid direct leakage. The remaining signal is dominated by `maintenance_type`; the model card says so.

### Freight pricing

| Metric | Value |
| --- | --- |
| MLflow run | `792c772cfb53478fab3cd443fcd5e50e` |
| Registry version | `1` (alias `champion`) |
| Train / validation / test | 1,371 / 294 / 294 |
| MAE | $13.15 |
| Median absolute error | $10.62 |
| RMSE | $18.01 |
| MAPE | 7.42% |
| R² | 0.9616 |
| 90% interval coverage | 90.14% |
| Baseline MAE / MAPE | $30.02 / 16.64% |
| Residual quantiles (0.8 / 0.9 / 0.95) | 18.92 / 28.43 / 34.31 |
| Features | 13 booking-time features |

The freight model only sees information available when the quote is made. `cost`, `delivery_date`, `status`, `transit_days`, and `shipment_id` are excluded. The interval is a split-conformal residual quantile, not a Gaussian assumption.

---

## Quickstart (local)

### 1. Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If you use `uv` (the reference environment did):

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

### 2. Put the data in place

Place the two source CSVs in `data/raw/`:

```text
data/raw/logistics_predictive_maintenanceV2.csv   # 250k maintenance records
data/raw/logistics_shipments_dataset.csv          # 2k shipments
```

The loader also searches `archive/` and the project root, but `data/raw/` is the expected location. The CSVs are not committed because of size.

### 3. Configure

```bash
cp .env.example .env
```

The defaults use SQLite and a local MLflow store, so no edits are required for a local run.

### 4. Ingest, train, serve

```bash
python -m src.cli ingest --force
python -m src.cli train --model all
python -m src.cli serve --reload
```

Open <http://localhost:8000>. The dashboard is served by FastAPI at `/`; OpenAPI docs are at `/docs`.

Training writes:

- MLflow runs and metrics to `mlflow.db` / `mlruns/`
- Registry versions and the `champion` alias
- Local serving bundles under `models/maintenance/current/` and `models/freight/current/`
- Model cards, metadata, reference distributions, and evaluation plots

### 5. Inspect MLflow

```bash
mlflow ui --backend-store-uri sqlite:///./mlflow.db --port 5001
```

Open <http://localhost:5001>. Both experiments and registered models are visible.

---

## Quickstart (Docker Compose)

```bash
docker compose up --build -d
docker compose logs -f api
```

Open <http://localhost:8000>. The API entrypoint detects an empty warehouse and ingests `data/raw/` into PostgreSQL on first boot. The image already contains the promoted bundles, so serving does not wait for training.

Services:

| Service | Port | Purpose |
| --- | --- | --- |
| `api` | 8000 | FastAPI + Gunicorn + dashboard |
| `mlflow` | 5000 | MLflow tracking server and registry (Postgres backend) |
| `postgres` | 5432 | Warehouse and MLflow backend store |
| `init` | — | Optional one-shot ingest + train job (`--profile bootstrap`) |

Retrain and promote inside the stack:

```bash
docker compose run --rm init
```

Stop everything, keeping volumes:

```bash
docker compose down
```

Stop and delete the warehouse and MLflow state:

```bash
docker compose down -v
```

---

## Commands

The CLI is the operational surface. Every command is also callable from Python or Prefect.

```bash
# Data
python -m src.cli ingest --force
python -m src.cli ingest --sample 5000 --force

# Models
python -m src.cli train --model all
python -m src.cli train --model maintenance --no-mlflow
python -m src.cli evaluate

# Inference
python -m src.cli predict-maintenance --demo
python -m src.cli predict-freight --demo
python -m src.cli predict-maintenance --payload '{"usage_hours": 18000, "brake_condition": "Poor"}'

# Operations
python -m src.cli info
python -m src.cli monitor --window 500
python -m src.cli serve --reload
python -m src.cli pipeline --flow refresh
python -m src.cli s3-sync --direction list
```

Useful `make` targets:

```bash
make install      # create .venv and install dependencies
make ingest       # load raw CSVs
make train        # train and register both models
make serve        # API with autoreload
make test         # pytest
make cov          # pytest + coverage
make lint         # ruff check
make format       # ruff format + safe fixes
make check        # lint + test
make docker-up    # Compose stack
make docker-init  # retrain inside Compose
```

---

## API

Canonical routes live under `/api`. Compatibility aliases (`/predict`, `/predict/freight`, `/health`, `/models`, `/metrics`) exist for simple demos and are hidden from the OpenAPI schema.

### Maintenance

```bash
curl -s http://localhost:8000/api/predict/maintenance \
  -H 'content-type: application/json' \
  -d '{
    "vehicle_id": 82468,
    "year_of_manufacture": 2015,
    "usage_hours": 18400,
    "load_capacity": 10000,
    "actual_load": 10105,
    "engine_temperature": 124.3,
    "tire_pressure": 23.9,
    "battery_status": 43.9,
    "vibration_levels": 3.32,
    "oil_quality": 48.6,
    "failure_history": 10,
    "anomalies_detected": 7,
    "diagnostic_trouble_code_count": 8,
    "maintenance_type": "Electrical Repair",
    "brake_condition": "Poor",
    "service_note": "Driver reports brake vibration and an overheat smell near the wheel, ERR-431. Pads last replaced at 184,000 miles."
  }'
```

The response contains the calibrated probability, the threshold, a risk band (`low`, `moderate`, `high`, `critical`), the recommended action, the top contributing signals, parsed error codes and mileage from the free-text note, and the logged `prediction_id`.

### Freight

```bash
curl -s http://localhost:8000/api/predict/freight \
  -H 'content-type: application/json' \
  -d '{
    "origin_warehouse": "Warehouse_CHI",
    "destination": "Dallas",
    "carrier": "UPS",
    "shipment_date": "2025-03-18",
    "weight_kg": 38.9,
    "distance_miles": 1225
  }'
```

The response contains the point estimate, 90% interval, cost per mile, quote validity date, and the observed lane benchmark when the lane has history.

### Other endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness, DB and model availability |
| GET | `/api/ready` | Readiness (503 until both models load) |
| GET | `/api/models` | Promoted model metadata and metrics |
| GET | `/api/models/{key}/card` | Markdown model card |
| POST | `/api/predict/maintenance` | Single vehicle score |
| POST | `/api/predict/maintenance/batch` | Up to 500 vehicles |
| POST | `/api/predict/freight` | Single quote |
| POST | `/api/predict/freight/batch` | Up to 500 quotes |
| GET | `/api/analytics/overview` | Fleet, shipment and traffic KPIs |
| GET | `/api/analytics/segments` | Risk by vehicle type |
| GET | `/api/analytics/maintenance-trend` | Monthly service volume and risk rate |
| GET | `/api/analytics/lanes` | Busiest freight lanes |
| GET | `/api/analytics/lane-benchmark` | Observed cost distribution for a lane |
| GET | `/api/vehicles/priority` | Model-scored maintenance queue |
| GET | `/api/vehicles/{id}` | Vehicle profile plus live score |
| GET | `/api/monitoring/drift` | PSI report from recent requests |
| GET | `/api/monitoring/traffic` | Prediction volume and latency |
| GET | `/api/monitoring/feedback` | Accuracy from labelled predictions |
| POST | `/api/feedback/{event_id}` | Attach the observed outcome |
| GET | `/api/metrics` | Prometheus text exposition |
| POST | `/api/admin/reload` | Reload bundles or MLflow champion |

Full request/response examples are in [docs/api.md](docs/api.md).

---

## Project layout

```text
fleet-iq-mlops/
├── api/
│   ├── main.py                 # app factory, lifespan, static dashboard
│   ├── deps.py                 # registry, DB session, prediction logging
│   ├── schemas.py              # Pydantic contracts
│   └── routers/                # predictions, analytics, monitoring, system
├── src/
│   ├── config.py               # pydantic-settings configuration
│   ├── db.py                   # engine/session lifecycle
│   ├── db_models.py            # SQLAlchemy 2 models
│   ├── ingest.py               # CSV discovery, cleaning, batch inserts
│   ├── features.py             # shared train/serve features, regex parser, SQL analytics
│   ├── train.py                # training, evaluation, MLflow, bundles, plots
│   ├── predict.py              # maintenance and freight inference
│   ├── registry.py             # local bundles, promotion, MLflow loading
│   ├── monitoring.py           # PSI drift, traffic, feedback metrics
│   ├── analytics.py            # dashboard SQL
│   ├── pipeline.py             # Prefect flows and deployments
│   ├── storage.py              # optional S3 artefact mirror
│   └── cli.py                  # operational CLI
├── frontend/                   # dependency-free dashboard and bundled fonts
├── models/                     # promoted bundles + run history
├── tests/                      # 111 tests
├── docs/                       # architecture, walkthrough, API, runbook, UI, portfolio
├── docker/entrypoint.sh        # container bootstrap
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml              # pytest, coverage, ruff
└── requirements*.txt
```

---

## Design decisions that matter

**Shared feature builders.** Training and serving call the same `prepare_maintenance_features` / `prepare_freight_features` functions. There is no notebook-to-production rewrite and no train/serve skew.

**Leakage is a first-class concern.** The maintenance dataset contains post-service fields that would make the task trivial. The exclusion list is explicit in `src/features.py`, asserted by tests, and recorded in every model card. The synthetic-data caveat is documented rather than hidden.

**Local bundles, optional registry.** The default serving path reads `models/<key>/current/`, so the API boots without MLflow. `MODEL_SOURCE=mlflow` switches to the registry's `champion` alias. Both paths are covered by tests.

**Monitoring uses training references.** Each bundle stores numeric bin edges, bin shares, and categorical frequencies. PSI compares live requests against those references. Watch at 0.10, alert at 0.25, with a minimum window of 25 requests so the dashboard does not overreact to a handful of predictions.

**The dashboard has no build step.** It is plain HTML, CSS, and JavaScript served by FastAPI. That removes Node from the deployment, keeps the image small, and makes the UI easy to inspect.

**Tests do not need the production warehouse.** A 5,000-row fixture database and small model bundles under `models_test/` keep the suite fast (about 3 seconds) and independent of the full training run.

---

## Testing and quality

```bash
pytest tests -q
pytest tests -q --cov=src --cov=api --cov-report=term-missing
ruff check src api tests
ruff format --check src api tests
```

Current state:

- `111 passed`
- Coverage: 69% overall, with the serving, feature, analytics, monitoring, and registry paths covered
- Ruff: clean, line length 100, target py311
- Lighthouse on the overview: Accessibility 1.0, Best Practices 1.0, SEO 1.0

The two pytest warnings are upstream deprecations from Starlette's TestClient and FastAPI's `HTTP_413` constant.

---

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Component-by-component architecture, data flow, schema, design trade-offs |
| [docs/walkthrough.md](docs/walkthrough.md) | How each module was built, with the reasoning behind the code |
| [docs/api.md](docs/api.md) | Endpoint reference and curl examples |
| [docs/runbook.md](docs/runbook.md) | Operations: retraining, rollback, drift response, troubleshooting |
| [docs/ui.md](docs/ui.md) | Dashboard structure, design system, accessibility notes |
| [docs/portfolio.md](docs/portfolio.md) | Resume bullets and interview talking points |

---

## Caveats

- The maintenance dataset is synthetic. The near-perfect metrics are a property of the data, not a claim about real fleet performance.
- The freight dataset covers 2023 US lanes with 2,000 shipments. The model is a demonstration of the workflow, not a production pricing engine.
- SQLite is the local default. PostgreSQL is used in Docker; the SQLAlchemy layer is portable but the test suite exercises SQLite.
- MLflow uses `cloudpickle` serialization because skops blocks the histogram gradient boosting types used here.
- The dashboard is intentionally framework-free. If this grew into a multi-team product, the next step would be a typed client and a component library, not a rewrite of the serving contract.

---

## License

MIT. The datasets retain their original licenses; check the source distribution before redistributing them.

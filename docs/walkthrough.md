# Build Walkthrough

This is the long-form account of how FleetIQ was built: the order of work, the decisions that shaped the code, the bugs that were worth fixing, and the commands used to verify each layer. It is written for someone who wants to understand the project rather than just run it.

---

## 0. Starting point

The repository began with four CSVs and no application code:

```text
logistics_predictive_maintenanceV2.csv       94 MB, 250,000 rows
logistics_dataset_with_maintenance_required.csv
revised_logistics_dataset_V2.csv
logistics_shipments_dataset.csv              172 KB, 2,000 rows
```

The first pass was exploratory: column names, dtypes, missingness, target balance, duplicates, and the relationship between the target and the post-service columns. Two facts changed the design immediately.

1. **The maintenance data is synthetic and separable.** `maintenance_required` is essentially a deterministic function of service state. A model can reach ROC AUC 1.0 without trying.
2. **The dataset contains leakage.** `days_since_last_maintenance`, `maintenance_severity`, `predictive_score`, `maintenance_cost`, and the reliability indices are recorded after the service decision. Including them would produce an impressive and useless model.

The project therefore treats leakage prevention as a feature, not a footnote. The exclusion list is explicit, tested, and repeated in every model card.

The shipments file is a different shape: 2,000 rows, 2023 dates, 41 missing costs, and a heavy-tailed cost distribution. That became the second model.

---

## 1. Project skeleton

The first commits created the directory structure and configuration layer:

```text
src/        domain code
api/        FastAPI app and routers
frontend/   dashboard
tests/      pytest suite
models/     serving bundles
data/       raw, processed, SQLite warehouse
docs/       this documentation
docker/     container entrypoint
```

`src/config.py` uses `pydantic-settings` so every runtime value is an environment variable with a sane default. The same settings object is used by the CLI, the API, training, and the Prefect flows. There is no second configuration file for Docker.

Key defaults:

```python
database_url = "sqlite:///./data/fleetiq.db"
model_dir = Path("models")
mlflow_tracking_uri = "sqlite:///./mlflow.db"
model_source = "local"          # or "mlflow"
reference_year = 2025
drift_window = 500
psi_watch = 0.10
psi_alert = 0.25
max_batch_size = 500
```

`settings.path()` resolves relative paths against the project root, not the current working directory. That matters when the API is started from a different directory or from inside a container.

---

## 2. Database layer

`src/db.py` builds a SQLAlchemy engine with `pool_pre_ping`, a session factory, a `session_scope()` context manager, and an `init_db()` helper. SQLite gets `check_same_thread=False`. In-memory SQLite uses `StaticPool` so tests share one connection.

`src/db_models.py` defines four tables with SQLAlchemy 2 typed mappings:

- `maintenance_records`
- `shipments`
- `prediction_events`
- `ingest_runs`

The schema intentionally stores post-service columns even though the model cannot use them. Analytics and evaluation need them, and keeping them visible makes the leakage boundary explicit rather than accidental.

The database URL is the only difference between local and Docker. The application code does not branch on the backend except for SQLite-specific connect arguments.

---

## 3. Ingestion

`src/ingest.py` was built next because every later step depends on a trustworthy warehouse.

### Column normalization

The raw headers are inconsistent (`Vehicle ID`, `Maintenance-Required`, `CANMessageRateHz`). `normalize_column_name` applies four regex passes:

```python
value = re.sub(r"[\s\-]+", "_", str(name).strip())
value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", value)
return value.lower().strip("_")
```

That converts `Vehicle ID` → `vehicle_id`, `Maintenance-Required` → `maintenance_required`, and `CANMessageRateHz` → `can_message_rate_hz`. Columns starting with `unnamed` are dropped.

### Cleaning

`_clean_maintenance`:

- Adds missing columns as `NaN` so older CSV variants still load.
- Coerces dates with `pd.to_datetime(..., errors="coerce")`.
- Coerces numeric columns with `pd.to_numeric(..., errors="coerce")`.
- Drops rows without `vehicle_id` or `maintenance_required`.
- Casts IDs to integers and clips the target to `{0, 1}`.
- Fills count-like integers with zero or the median, other numerics with the median.
- Fills categorical gaps with `unknown` and strips whitespace.
- Records missing counts before and after filling.

`_clean_shipments` does the analogous work, but keeps rows with missing `cost` because they are still valid for analytics. `rows_without_cost` is reported separately.

### Validation

`validate_maintenance_frame` and `validate_shipment_frame` return a JSON-safe report:

```json
{
  "rows": 250000,
  "missing_columns": [],
  "invalid_target_rows": 0,
  "duplicate_vehicle_ids": 0,
  "ok": true
}
```

The report is written to `data/processed/ingest_report.json` and mirrored into `ingest_runs.details`.

### Insert strategy

A naive `DataFrame.to_sql` is slow and opaque. The loader converts rows to dictionaries with `_records`, normalizes NumPy types and dates with `_json_safe`, and inserts in batches of 4,000 using SQLAlchemy Core. The full load is 250,000 maintenance rows and 2,000 shipments.

The first full ingest produced:

```text
maintenance: 250,000 rows, 30% positive, 2023-01-02 → 2024-12-02
shipments:   2,000 rows, 41 without cost, 2023 calendar year
total:       252,000 rows
```

### Tests

`tests/test_ingest.py` covers normalization, CSV reading, cleaning, validation, JSON conversion, and source discovery. The source-discovery test uses `tmp_path` and monkeypatches `settings.data_dir`, so CI does not need the 209 MB raw CSVs.

---

## 4. Feature engineering

`src/features.py` is the most important module for correctness. It defines the model contract once and is imported by both training and serving.

### Maintenance contract

```python
MAINTENANCE_NUMERIC = [
    "usage_hours", "load_capacity", "actual_load", "engine_temperature",
    "tire_pressure", "fuel_consumption", "battery_status", "vibration_levels",
    "oil_quality", "failure_history", "anomalies_detected",
    "diagnostic_trouble_code_count", "can_message_rate_hz",
    "sensor_packet_loss_rate", "historical_maintenance_cost",
    "vehicle_age", "usage_per_year", "load_ratio", "temp_delta",
    "vibration_per_1k_hours", "dtc_per_10k_hours", "packet_loss_pct",
    "maintenance_cost_per_hour", "brake_wear_flag", "failure_flag", "anomaly_flag",
]
```

The derived features encode domain intuition: load ratio catches overload, usage per year separates old low-use vehicles from new high-use vehicles, and per-hour rates capture wear density rather than absolute counts.

`prepare_maintenance_features` is defensive. It adds missing source columns, coerces types, computes derived columns, and returns a frame whose columns exactly match `MAINTENANCE_FEATURES`. A test passes a one-column frame and asserts the output still has the full schema.

### Leakage list

`MAINTENANCE_EXCLUDED_COLUMNS` is a frozenset. The test suite asserts that no excluded column appears in `MAINTENANCE_FEATURES`. This is the guardrail that prevents a future contributor from "improving" the model by adding `days_since_last_maintenance`.

### Freight contract

`prepare_freight_features` builds calendar features from `shipment_date` and lane/band features from origin, destination, weight, and distance. It never reads `cost`, `transit_days`, `delivery_date`, `status`, or `shipment_id`, even when those columns are present.

### Regex / NLP note parser

`parse_service_note` is a deterministic parser, not a transformer. It handles:

- Error codes: `ERR-431`, `err 902`, `E-123`
- Part numbers: `ABC-1234`
- Mileage: `184,000 miles`, `184000 mi`
- Keywords from a curated risk vocabulary
- Urgency scoring based on keyword severity and explicit codes

The parser is tested with a realistic note and a benign note. It returns a structured object that the UI can render and the action logic can use.

### SQL analytics

The same module contains read-only SQL helpers used by the dashboard:

- `segment_risk`: risk rate by vehicle type
- `monthly_maintenance`: service volume and risk rate by month
- `lane_statistics`: busiest lanes with cost and distance
- `lane_benchmark`: p25/median/p75 cost for one lane
- `carrier_statistics`: cost comparison by carrier
- `brake_condition_stats`: risk rate by brake condition
- `fetch_maintenance_frame`: a raw frame for drift windows

Keeping SQL here means the routers stay thin and the queries can be tested against the fixture database without HTTP.

---

## 5. Training

`src/train.py` was built in layers.

### 5.1 Data loading and splitting

`_load_maintenance` reads the warehouse, selects the feature contract, and splits 70/10/20 with a fixed seed. The dataset's own `data_split` column is ignored. `_load_freight` drops rows without cost and splits the same way.

### 5.2 Preprocessing

`_preprocessor` returns a `ColumnTransformer`:

- Numeric: median imputation → standard scaling
- Categorical: ordinal encoding with `handle_unknown="use_encoded_value"` and `unknown_value=NaN`

The baseline uses one-hot encoding instead. The main model uses ordinal encoding because histogram gradient boosting handles categorical codes natively when the categorical mask is passed.

### 5.3 Maintenance model

`HistGradientBoostingClassifier` with 350 iterations, learning rate 0.06, 31 leaves, L2 1.0, and early stopping. The categorical mask marks the categorical columns so the model treats their integer codes as categories, not ordered numbers.

A logistic-regression baseline is trained on the same split. The baseline is logged but never promoted.

### 5.4 Threshold tuning

The validation set produces probabilities. `_tune_threshold` computes the precision-recall curve and selects the threshold with the highest F1. The test set is scored once with that threshold.

On the full data:

```text
threshold = 0.9999975252703134
```

That number looks absurd until you inspect the probabilities. The minimum positive-class probability is exactly the threshold, and the maximum negative-class probability is about `1.06e-06`. The classes are separable. The threshold is the model's way of putting the decision boundary in the empty space between them.

The risk bands were initially written as probability/threshold ratios. That collapsed every level into `critical` when the threshold was near 1. The fix was to anchor the bands to multiples of the threshold:

```text
critical: p >= threshold
high:     p >= 0.75 * threshold
moderate: p >= 0.35 * threshold
low:      otherwise
```

A test pins this behavior for both a 0.5 threshold and a 0.9999 threshold.

### 5.5 Freight model

`HistGradientBoostingRegressor` with `loss="absolute_error"`, 500 iterations, learning rate 0.05, 31 leaves, L2 1.0, and early stopping. Absolute error matches the operational cost function better than squared error on a heavy-tailed target.

The validation residuals produce quantiles at 0.8, 0.9, and 0.95. The 0.9 quantile is stored in metadata and used for the serving interval. Test coverage is 90.14%.

### 5.6 Evaluation

`_classification_metrics` returns ROC AUC, PR AUC, F1, precision, recall, accuracy, Brier score, log loss, and the confusion matrix. `_regression_metrics` returns MAE, median absolute error, RMSE, MAPE, R², interval coverage, and baseline MAE/MAPE.

`_safe_metrics` removes NaN and infinity before logging. `_safe_json` recursively converts NumPy scalars, dates, and non-finite floats. This matters because MLflow and JSON both reject NaN.

### 5.7 Explainability

Two kinds of importance are produced:

- **Permutation importance** on a sample of the test set, using ROC AUC for classification and negative MAE for regression.
- **Native feature importance** where available, plus segment-level metrics by vehicle type, brake condition, route, carrier, and distance band.

The first full run exposed an index-alignment bug. `permutation_importance` returns positions relative to the sampled frame, but the code indexed the original frame. The fix was to reset the sampled frame index and zip the sampled columns with the importance array:

```python
sampled = frame.sample(n=..., random_state=...).reset_index(drop=True)
...
for feature, value in zip(sampled.columns, result.importances_mean, strict=True)
```

A regression test uses a frame with a non-default index to make sure the bug cannot return.

The result is honest: `maintenance_type` has importance 0.50 and every other feature is 0.00. The target is a function of maintenance type in this synthetic dataset. The model card says so instead of pretending the telemetry drives the score.

### 5.8 MLflow

The tracking URI defaults to `sqlite:///./mlflow.db`. Each run logs:

- Params: model hyperparameters, split sizes, feature count
- Metrics: test and validation metrics, baseline metrics, threshold
- Tags: git commit when available, dataset fingerprint, model key
- Artefacts: model, plots, metadata, reference stats, model card

Registration creates a new version and sets the `champion` alias when `AUTO_PROMOTE=true`. The serialization format is `cloudpickle` because MLflow's skops path rejects `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`.

The full maintenance run:

```text
run_id       13a23254c21e4a20a3b542e756362793
version      1
ROC AUC      1.0
PR AUC       1.0
F1           1.0
Brier        2.63e-12
log loss     1.49e-06
test rows    50,000 (15,000 positive)
confusion    TN 35,000 · FP 0 · FN 0 · TP 15,000
```

The full freight run:

```text
run_id       792c772cfb53478fab3cd443fcd5e50e
version      1
MAE          13.15
MedAE        10.62
RMSE         18.01
MAPE         7.42%
R²           0.9616
coverage_90  90.14%
```

### 5.9 Quality gate and bundles

`evaluate_gate` enforces the minimum AUC and maximum MAPE. If it passes, `write_bundle` writes the run directory, reference stats, model card, and metadata, then `promote_run` copies the run to `current/` and writes `current.json`.

The bundle is self-contained. Serving does not need the training frame, MLflow, or the warehouse to compute features or intervals.

---

## 6. Registry and inference

`src/registry.py` defines `LoadedModel` and `ModelRegistry`.

`LoadedModel` is a dataclass with convenience properties:

```python
model.version              # MLflow version or run ID
model.threshold            # maintenance decision threshold
model.residual_quantiles   # freight interval quantiles
model.feature_columns
model.numeric_features
model.categorical_features
```

`ModelRegistry` loads each key independently, records errors per key, and supports `reload(source)`. The API uses `registry.get(key)` and converts `ModelNotFoundError` into a 503.

`src/predict.py` implements the two predictors.

`MaintenancePredictor.predict`:

1. Build features with `prepare_maintenance_features`.
2. Predict probability.
3. Apply the threshold.
4. Assign a risk level.
5. Build up to four signals from a curated list of high-signal features.
6. Parse the service note.
7. Choose an action based on risk, brake condition, and note urgency.

`FreightPredictor.predict`:

1. Build booking-time features.
2. Predict cost and clip at zero.
3. Apply the stored 0.9 residual quantile.
4. Compute cost per mile.
5. Set the quote validity date.
6. Attach the lane benchmark if provided.

The output is JSON-safe and matches the Pydantic response models.

---

## 7. API

The API was built after the domain code so the routers could stay thin.

`api/schemas.py` defines request and response contracts. Request models use `extra="forbid"`, which prevents a client from sneaking in `maintenance_cost` or another leakage column. That choice also caught a frontend bug: a payload key mismatch produced a 422 until the form fields were aligned.

`api/deps.py` provides:

- `get_registry`
- `get_db`
- `client_label` (user agent / forwarded-for, truncated)
- `record_prediction`
- `require_admin`

`api/routers/predictions.py` implements single and batch endpoints for both models. Batch errors are collected per row. The batch limit is enforced before inference.

`api/routers/analytics.py` exposes the SQL helpers, the priority queue, and vehicle profiles. The priority queue scores a sample of vehicles with the live model, sorts by probability, and returns the raw record plus the score.

`api/routers/monitoring.py` exposes drift, traffic, feedback, recent events, and the feedback submission endpoint.

`api/routers/system.py` exposes health, readiness, models, model cards, admin reload, and Prometheus text metrics.

`api/main.py` mounts the static dashboard, adds CORS and GZip, and registers compatibility aliases (`/predict`, `/predict/freight`, `/health`, `/models`, `/metrics`) with `include_in_schema=False`.

The first API smoke test used `TestClient` and checked:

- health, readiness, models, model cards
- overview, options, lanes, segments, lane benchmark
- priority queue and vehicle profile
- maintenance prediction and batch
- freight prediction and lane benchmark
- feedback flow and missing-event 404
- monitoring traffic and drift
- Prometheus metrics
- admin reload
- dashboard and static assets
- compatibility aliases
- OpenAPI schema

All of those became tests in `tests/test_api.py`.

---

## 8. Dashboard

The frontend is deliberately framework-free: `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`, and bundled WOFF2 fonts.

### Structure

- Sticky masthead with API / warehouse / model status
- Overview panel: hero statement, KPI grid, segment chart, maintenance trend, brake-condition cards, carrier chart, registry card
- Maintenance panel: two-column form, sliders, service-note textarea, result card with probability, risk band, signals, parsed note
- Freight panel: quote form, cost interval, lane benchmark, recent shipments
- Priority panel: model-scored queue, filter, vehicle dialog
- Monitoring panel: drift table, traffic summary, feedback metrics, recent events
- Status rail and toast notifications

### Design system

The palette is warm limestone (`#f3efe6`), deep petrol (`#0e1b20`), signal orange (`#ee5a29`), jade (`#0b7f6d`), amber (`#b97c0c`), and red (`#b13a2b`). Typography is Fraunces for display, Manrope for UI, and JetBrains Mono for numbers and metadata. Cards use offset shadows rather than soft blur, which gives the dashboard an editorial, printed feel instead of a generic SaaS look.

The fonts are bundled locally, so the dashboard does not depend on Google Fonts or any external CDN.

### Bugs found during browser verification

1. **Number inputs rejected decimal values.** `m-capacity`, `m-load`, `m-packet`, `m-histcost`, `m-temp`, `m-vibration`, `m-battery`, and `m-oil` defaulted to `step="1"`. The browser refused values such as `10000.5`. Fixed with `step="any"`.
2. **Gauge SVG font attributes.** The gauge used `font-family="var(--mono)"`, which is invalid in SVG presentation attributes. Moved to `style="font-family: var(--mono)"`.
3. **Freight lane benchmark did not populate on first load.** `loadOptions()` populated the lane select but did not call `refreshBenchmark()`. Added the call at the end of `loadOptions()`.
4. **The high-risk example could load before the selects existed.** `fillMaintenanceForm` wrote numeric fields synchronously while the options request was still in flight, so the maintenance-type select kept its default and the score came back low. Added `ensureOptions()` and awaited it in the demo loader and both submit handlers.
5. **Fractional demo values failed HTML validation.** The example record contains `10038.57` usage hours, `23.88` psi and `60.05` Hz, but those inputs still used `step="1"` or `step="0.1"`. The browser silently blocked submission. Set `step="any"` on every numeric input that can receive warehouse values.

### Verification

- Live overview loads 250,000 vehicles.
- Maintenance form returns a `critical` risk with parsed `ERR-431` and `184,000` mileage.
- Freight quote returns a dollar estimate, 90% interval, and lane benchmark.
- No console errors.
- No horizontal overflow.
- Lighthouse on the overview: Accessibility 1.0, Best Practices 1.0, SEO 1.0.

Contrast fixes from Lighthouse:

- Primary buttons: `#c33f12`
- Registry meta: `#5d6d73`
- Model pill ok: `#08695a`

---

## 9. Monitoring

`src/monitoring.py` was built after serving because it needs `prediction_events`.

### PSI

`build_reference_stats` stores:

- For numeric features: mean, std, min, max, bin edges, and bin shares
- For categorical features: frequency table
- Target rate and row count

`_numeric_psi` bins the live values with the stored edges and compares shares. `_categorical_psi` compares the live frequency table with the stored one and groups unseen values into `other`. `population_stability_index` validates equal lengths and clamps tiny floating-point noise.

### Windows

`drift_from_events` reads the last N prediction events for a model, extracts the request payloads, rebuilds features with the same serving function, and computes PSI per feature. If the window is smaller than 25, it returns `insufficient_data`.

### Feedback

`feedback_summary` reads events with `actual_value` set. Maintenance events produce accuracy at 0.5. Freight events produce MAE and MAPE. The feedback endpoint attaches a label to an existing event by ID.

---

## 10. Orchestration and cloud hooks

`src/pipeline.py` defines Prefect flows that call `run_ingest`, `train_all`, `evaluate_gate`, and the monitoring functions. The flows are thin wrappers, not reimplementations.

`src/storage.py` is an optional S3 mirror. `boto3` is imported lazily so the core install stays small. `upload_bundle` and `download_bundle` copy a bundle directory, and `local_inventory` reports what is on disk.

The project does not require AWS. The cloud path is an extension point, not a dependency.

---

## 11. Tests

The test suite was built around fixtures rather than the production warehouse.

`tests/conftest.py`:

- Copies `data/test_fleetiq.db` to a temporary directory.
- Sets `DATABASE_URL` to the copy.
- Sets `MODEL_DIR=models_test`.
- Disables MLflow and the admin token.
- Provides `engine`, `session`, `client`, `maintenance_model`, `freight_model`, and payload fixtures.

The fixture database has 5,000 maintenance rows and 250 shipments (243 with cost). The fixture bundles have the current feature schema. That keeps the suite independent of the full 250k-row training run.

Nine test modules:

| Module | Focus |
| --- | --- |
| `test_features.py` | leakage contract, derived features, freight contract, note parser, SQL analytics |
| `test_ingest.py` | normalization, cleaning, validation, JSON conversion, source discovery |
| `test_predict.py` | risk bands, actions, maintenance output, freight interval, zero distance |
| `test_registry.py` | reference stats, bundle round trip, promotion, model card, missing models |
| `test_monitoring.py` | PSI math, drift report, traffic, feedback |
| `test_analytics.py` | overview, priority queue, vehicle profile, options, shipments |
| `test_api.py` | every route, batch limits, feedback, metrics, aliases, static assets |
| `test_cli.py` | parser, info, demo predictions, evaluate gate |
| `test_train.py` | threshold tuning, metrics, safe JSON, fingerprint, importance, gate, plots |

Result:

```text
111 passed, 2 warnings in 2.64s
```

The warnings are upstream: Starlette's TestClient deprecates the current httpx integration, and FastAPI's `HTTP_413_REQUEST_ENTITY_TOO_LARGE` is deprecated in favor of `HTTP_413_CONTENT_TOO_LARGE`.

Coverage is 69% overall. The lowest-covered modules are `src/train.py` (the full training path), `src/pipeline.py` (Prefect flows), and `src/storage.py` (S3, which requires credentials). Those are exercised manually and by the training/evaluation commands rather than unit tests.

### Lint

`pyproject.toml` configures ruff with line length 100, target py311, and `E,F,I,UP,B,SIM,C4,RUF`. `B008` is ignored for FastAPI and CLI files because `Depends(...)` and `Option(...)` in defaults are the framework idiom.

The cleanup pass applied 73 safe fixes, then resolved the remaining 26 by hand: long lines, two `zip(..., strict=...)` calls, and one collection concatenation.

---

## 12. Deployment

`Dockerfile`:

- `python:3.13-slim`
- `libgomp1` for scikit-learn, `curl` for health checks
- installs `requirements.txt`
- copies `src`, `api`, `frontend`, `models`, and the entrypoint
- creates a non-root `fleetiq` user (UID 10001)
- exposes 8000
- health check on `/api/health`
- Gunicorn with Uvicorn workers

`docker/entrypoint.sh` initializes the schema and ingests raw CSVs if the warehouse is empty. It catches ingest failures and continues, so the API can still start for debugging.

`docker-compose.yml`:

- `postgres` with a health check
- `mlflow` with a Postgres backend and a shared artifact volume
- `api` with the same image, Postgres, MLflow, and a health check
- `init` under the `bootstrap` profile for ingest + training

The image contains the promoted bundles, so `docker compose up --build` serves immediately. The `init` profile is for retraining inside the stack.

`Makefile` wraps the common commands. `.github/workflows/ci.yml` runs ruff, format check, pytest with coverage, a Docker build, and a container health smoke test.

---

## 13. What I would do next

1. **Real labels.** Replace the synthetic maintenance target with work orders and failure events. Rebuild the model card with real base rates and cost-sensitive thresholds.
2. **Feature store.** If the fleet grows, move the shared feature builders behind a versioned feature service so training and serving can evolve independently.
3. **Delayed-label pipeline.** Ingest work-order outcomes on a schedule and compute rolling accuracy per model version.
4. **Shadow deployment.** Serve the candidate and champion in parallel, compare predictions, and promote on a business metric rather than offline AUC.
5. **Pricing guardrails.** Add min/max quote bounds, lane-level fallbacks, and a human approval step for large deviations from the benchmark.
6. **Observability.** Add OpenTelemetry traces and a proper Prometheus client once there is more than one service.

The current system is intentionally small enough that every one of those changes has an obvious insertion point.

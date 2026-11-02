# Architecture

FleetIQ is a small system with a deliberately boring shape: a relational warehouse, a batch training job, a model registry, a stateless API, and a dashboard. Everything that looks like magic is either SQL, a scikit-learn pipeline, or a file on disk.

This document explains the pieces and why they are arranged this way.

---

## 1. Runtime topology

```text
                       ┌────────────────────────────────────────────┐
                       │                Source CSVs                 │
                       │  logistics_predictive_maintenanceV2.csv    │
                       │  logistics_shipments_dataset.csv           │
                       └───────────────────┬────────────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │           src/ingest.py         │
                          │  discover → normalize → clean   │
                          │  validate → batch insert        │
                          └────────────────┬────────────────┘
                                           │
              ┌────────────────────────────▼────────────────────────────┐
              │              SQLAlchemy warehouse                       │
              │  maintenance_records  shipments                        │
              │  prediction_events    ingest_runs                      │
              │  SQLite locally · PostgreSQL in Docker                 │
              └───────────┬───────────────────────────┬────────────────┘
                          │                           │
              ┌───────────▼───────────┐   ┌───────────▼──────────────┐
              │     src/train.py      │   │    src/analytics.py      │
              │  features + model     │   │  dashboard SQL           │
              │  metrics + plots      │   │  segments, lanes, KPIs   │
              │  MLflow + bundles     │   └───────────┬──────────────┘
              └───────────┬───────────┘               │
                          │                           │
              ┌───────────▼───────────┐               │
              │  models/<key>/current │               │
              │  metadata · card ·    │               │
              │  reference stats      │               │
              └───────────┬───────────┘               │
                          │                           │
              ┌───────────▼───────────────────────────▼──────────────┐
              │                    api/main.py                       │
              │  FastAPI · Pydantic · prediction logging             │
              │  /api/predict · /api/analytics · /api/monitoring     │
              │  static dashboard at / · OpenAPI at /docs            │
              └───────────┬───────────────────────────┬──────────────┘
                          │                           │
              ┌───────────▼───────────┐   ┌───────────▼──────────────┐
              │ frontend/ dashboard   │   │  /api/metrics (text)     │
              │ HTML/CSS/JS, no build │   │  Prometheus scrape       │
              └───────────────────────┘   └──────────────────────────┘
```

There is no separate feature store, message bus, or online/offline split. For this problem size, the warehouse and the shared Python feature functions are the source of truth. The design keeps the operational surface small enough to reason about end to end.

---

## 2. Warehouse schema

Defined in `src/db_models.py` with SQLAlchemy 2 typed mappings.

### `maintenance_records`

One row per vehicle, keyed by `vehicle_id`.

| Group | Columns |
| --- | --- |
| Identity | `vehicle_id`, `make_and_model`, `vehicle_type`, `year_of_manufacture` |
| Operating context | `route_info`, `weather_conditions`, `road_conditions`, `communication_interface`, `telematics_gateway`, `edge_device_class` |
| Telemetry | `usage_hours`, `load_capacity`, `actual_load`, `engine_temperature`, `tire_pressure`, `fuel_consumption`, `battery_status`, `vibration_levels`, `oil_quality` |
| Diagnostics | `failure_history`, `anomalies_detected`, `diagnostic_trouble_code_count`, `can_message_rate_hz`, `sensor_packet_loss_rate` |
| Service history | `maintenance_cost`, `historical_maintenance_cost`, `downtime_maintenance`, `impact_on_efficiency`, `days_since_last_maintenance`, `last_maintenance_date`, `maintenance_type`, `brake_condition` |
| Outcome | `maintenance_severity`, `maintenance_severity_id`, `maintenance_required`, `predictive_score`, `data_split` |
| Lineage | `source`, `ingested_at` |

The outcome and post-service columns are stored for analytics and evaluation. They are **not** model inputs. That distinction is enforced in `src/features.py` and tested in `tests/test_features.py`.

### `shipments`

One row per shipment, keyed by `shipment_id`.

`origin_warehouse`, `destination`, `carrier`, `shipment_date`, `delivery_date`, `weight_kg`, `distance_miles`, `cost`, `status`, `transit_days`, `ingested_at`.

The pricing model sees booking-time fields only. `delivery_date`, `status`, `transit_days`, and `cost` are post-delivery facts.

### `prediction_events`

One row per served prediction. This is the observability backbone.

`id` (UUID), `created_at`, `endpoint`, `model_name`, `model_version`, `run_id`, `request` (JSON), `response` (JSON), `latency_ms`, `risk_probability`, `predicted_cost`, `actual_value`, `feedback_at`, `client`.

The request and response JSON make it possible to reconstruct a drift window without a separate feature logging pipeline. The feedback columns turn the table into a lightweight delayed-label store.

### `ingest_runs`

Audit trail for ingestion: source names, rows read/written/rejected, status, JSON details, notes, timestamps.

---

## 3. Ingestion

`src/ingest.py` is intentionally explicit rather than clever.

1. **Discover** the CSVs. `locate_sources()` checks `settings.maintenance_csv` / `settings.shipments_csv` first, then `data/raw/`, `archive/`, and the project root. It picks the largest match for each pattern, which avoids accidentally selecting a small sample file.
2. **Normalize column names.** `normalize_column_name` converts `Vehicle ID`, `Maintenance-Required`, and `CANMessageRateHz` into `vehicle_id`, `maintenance_required`, and `can_message_rate_hz`.
3. **Clean.** `_clean_maintenance` and `_clean_shipments` coerce dates, numeric columns, integer columns, fill categorical gaps with `unknown`, fill numeric gaps with medians (or zero for count-like fields), drop rows without keys or targets, and clip the target to `{0, 1}`.
4. **Validate.** `validate_maintenance_frame` and `validate_shipment_frame` report row counts, missing columns, invalid target rows, duplicate IDs, and negative distances. The report is written to `data/processed/ingest_report.json`.
5. **Insert in batches.** `_insert_in_batches` writes 4,000 rows at a time with SQLAlchemy Core. This is fast enough for 252,000 rows and avoids one giant transaction.
6. **Persist processed frames.** Cleaned data is written to Parquet under `data/processed/` for fast experimentation. The warehouse remains the serving source of truth.
7. **Record the run.** An `ingest_runs` row captures the outcome, including rejected rows and the report path.

`--sample N` takes a deterministic sample for quick local work. The test fixture database was created this way.

---

## 4. Feature engineering

`src/features.py` is the contract between training and serving.

### Maintenance features

36 features: 26 numeric and 10 categorical.

Numeric inputs include the raw telemetry plus derived features:

| Derived feature | Formula | Why |
| --- | --- | --- |
| `vehicle_age` | `reference_year - year_of_manufacture` | Normalizes model year across the fleet |
| `usage_per_year` | `usage_hours / max(vehicle_age, 1)` | Separates high-hours-new vehicles from old low-use vehicles |
| `load_ratio` | `actual_load / load_capacity` | Captures overload, which raw load does not |
| `temp_delta` | `engine_temperature - 90` | Centers the thermal signal |
| `vibration_per_1k_hours` | `vibration_levels / (usage_hours / 1000)` | Wear rate, not absolute vibration |
| `dtc_per_10k_hours` | `diagnostic_trouble_code_count / (usage_hours / 10000)` | Fault density |
| `packet_loss_pct` | `sensor_packet_loss_rate * 100` | Human-readable telemetry quality |
| `maintenance_cost_per_hour` | `historical_maintenance_cost / max(usage_hours, 1)` | Cost efficiency |
| `brake_wear_flag` | `brake_condition == "Poor"` | Boolean for the strongest categorical risk signal |
| `failure_flag` | `failure_history > 0` | Prior failure indicator |
| `anomaly_flag` | `anomalies_detected > 0` | Prior anomaly indicator |

Categorical inputs: `make_and_model`, `vehicle_type`, `route_info`, `weather_conditions`, `road_conditions`, `communication_interface`, `telematics_gateway`, `edge_device_class`, `maintenance_type`, `brake_condition`.

Missing values are handled inside the sklearn pipeline (`SimpleImputer` for numeric, ordinal encoding with an explicit unknown value for categorical). Serving never has to guess how to fill a gap.

### Leakage exclusions

`MAINTENANCE_EXCLUDED_COLUMNS` contains:

```text
days_since_last_maintenance
maintenance_severity
maintenance_severity_id
predictive_score
maintenance_cost
downtime_maintenance
impact_on_efficiency
last_maintenance_date
data_split
pcr · uir · tpi · mbf · ads · ohi · cmes · uer
pre_event_record
```

`days_since_last_maintenance` is effectively the target. `maintenance_severity` and `predictive_score` are outcomes. `maintenance_cost` and downtime are recorded after service. The reliability indices are derived from post-service state. `data_split` is the dataset's own split label. The project re-splits deterministically from the cleaned frame so the test set is never used for tuning.

The exclusion list is asserted by `test_feature_contract_has_no_leakage_columns`.

### Freight features

13 booking-time features:

- Raw: `origin_warehouse`, `destination`, `carrier`, `weight_kg`, `distance_miles`
- Calendar: `month`, `day_of_week`, `day_of_year`, `is_weekend`
- Derived: `lane` (`origin->destination`), `distance_band`, `weight_band`, `weight_per_mile`

`prepare_freight_features` ignores post-delivery columns even if they are present in the input frame. A test passes a frame containing `cost`, `transit_days`, and `delivery_date` and asserts they do not appear in the output.

### Free-text parsing

`parse_service_note` is a small, deterministic NLP layer over driver notes. It extracts:

- Error codes such as `ERR-431` and `err 902`, normalized to uppercase with a hyphen
- Part numbers such as `ABC-1234`
- Mileage from patterns like `184,000 miles`
- Risk keywords (`brake`, `overheat`, `vibration`, `leak`, `smoke`, `steering`, ...)
- An urgency score and band (`low`, `medium`, `high`)

The result is returned alongside the model score and used by `_action()` to escalate the recommendation. It is not fed into the tree model. That keeps the model contract numeric/categorical and makes the text behavior easy to test and explain.

---

## 5. Training

`src/train.py` is the largest module because it owns the full experiment lifecycle.

### Maintenance pipeline

```text
ColumnTransformer
  numeric     → SimpleImputer(median) → StandardScaler
  categorical → OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=NaN)
→ HistGradientBoostingClassifier(
      max_iter=350,
      learning_rate=0.06,
      max_leaf_nodes=31,
      l2_regularization=1.0,
      early_stopping=True
  )
```

A logistic-regression baseline with one-hot encoding is trained on the same split. The baseline is not promoted. It exists to make the lift visible in MLflow and the model card.

### Split strategy

The cleaned frame is split 70/10/20 into train/validation/test with a fixed random seed. The validation set tunes the decision threshold by maximizing F1. The test set is touched once, after the threshold is fixed.

The dataset's own `data_split` column is ignored. Re-splitting avoids trusting a label that may have been generated with knowledge of the target distribution.

### Threshold tuning

`_tune_threshold` walks the precision-recall curve and picks the threshold with the highest F1. On this synthetic dataset the classes are almost perfectly separable, so the optimal threshold lands at `0.9999975253`. That is a data property, not a modelling bug. Risk bands are anchored to the threshold rather than to fixed probability cutoffs so the UI remains readable when the threshold is near 1.

### Freight pipeline

```text
ColumnTransformer
  numeric     → SimpleImputer(median) → StandardScaler
  categorical → OrdinalEncoder(...)
→ HistGradientBoostingRegressor(
      loss="absolute_error",
      max_iter=500,
      learning_rate=0.05,
      max_leaf_nodes=31,
      l2_regularization=1.0,
      early_stopping=True
  )
```

Absolute-error loss is chosen because freight cost is heavy-tailed and the operation cares about median error, not squared error. A mean baseline is reported for comparison.

### Prediction intervals

The residuals on the validation set produce empirical quantiles at 0.8, 0.9, and 0.95. Serving uses the 0.9 quantile as a split-conformal interval:

```text
low  = max(0, prediction - q90)
high = prediction + q90
```

The test coverage is 90.14%, which is close to the nominal 90%. The interval is stored in metadata so serving does not need the validation set.

### Evaluation and quality gate

`evaluate_gate` checks:

- Maintenance ROC AUC ≥ `MIN_MAINTENANCE_AUC` (default 0.75)
- Freight MAPE ≤ `MAX_FREIGHT_MAPE` (default 25.0)

If either gate fails, training raises and the run is not promoted. The gate is deliberately loose enough for the demo data and strict enough to catch a broken feature pipeline.

### Artefacts

Each run writes:

- `model.joblib`: the fitted sklearn pipeline
- `metadata.json`: model name, run ID, feature lists, metrics, params, segments, importance, limitations, intended use, MLflow version
- `reference_stats.json`: numeric bin edges/shares and categorical frequencies for PSI
- `model_card.md`: human-readable governance summary
- Evaluation plots: calibration, confusion matrix, importance, residuals

Bundles live at `models/<key>/runs/<run_id>/`. Promotion copies the chosen run to `models/<key>/current/` and writes `models/<key>/current.json` with the run ID and timestamp.

### MLflow

MLflow logs params, metrics, tags, the model signature, plots, and the sklearn model. Registration creates a new version and sets the `champion` alias when `AUTO_PROMOTE=true`. Serialization uses `cloudpickle` because MLflow's default skops path rejects the histogram gradient boosting types.

The local serving path does not need MLflow at runtime. `MODEL_SOURCE=mlflow` is available for environments that want the registry to be authoritative.

---

## 6. Registry and serving

`src/registry.py` has two responsibilities:

1. Read and write local bundles, including promotion and model cards.
2. Optionally load the `champion` alias from MLflow.

`LoadedModel` exposes `pipeline`, `metadata`, `reference_stats`, `threshold`, `residual_quantiles`, `feature_columns`, `numeric_features`, and `categorical_features`. The API only depends on this interface, not on the storage backend.

`ModelRegistry` loads both keys at startup, records per-key errors, and exposes `available()`, `errors()`, and `info()`. A missing freight bundle does not prevent maintenance scoring. `/api/ready` returns 503 until both are available. `/api/health` stays 200 so the container can report the partial state.

`src/predict.py` contains the inference logic:

- `MaintenancePredictor` builds features, predicts a probability, applies the tuned threshold, assigns a risk band, ranks contributing signals, parses the service note, and recommends an action.
- `FreightPredictor` builds features, predicts cost, clips at zero, applies the conformal interval, computes cost per mile, sets a seven-day quote validity, and attaches a lane benchmark if one exists.

Risk bands:

```text
p >= threshold              → critical
p >= 0.75 * threshold       → high
p >= 0.35 * threshold       → moderate
otherwise                   → low
```

The multipliers keep the bands meaningful when the threshold is near 1. The response rounds the threshold to six decimals so the JSON is readable.

---

## 7. API lifecycle

`api/main.py` uses a FastAPI lifespan:

1. Configure logging.
2. Initialize the database schema.
3. Build the model registry from `MODEL_SOURCE`.
4. Record `started_at` for uptime.

The app mounts `frontend/` at `/static`, serves `index.html` at `/`, and exposes routers under `/api`.

Prediction requests are timed with `perf_counter`. `api/deps.py` writes a `prediction_events` row with the request, response, latency, model name/version/run ID, risk probability or predicted cost, and a coarse client label. If prediction logging is disabled, the endpoint still returns the response. It simply skips the insert.

Batch endpoints process records independently. A bad row becomes an entry in `errors` rather than failing the whole batch. The maximum batch size is `MAX_BATCH_SIZE` (default 500), enforced before any inference.

Admin reload is protected by `X-Admin-Token` when `ADMIN_TOKEN` is set. Locally it is open, which keeps the demo simple.

---

## 8. Monitoring

`src/monitoring.py` measures two things.

### PSI drift

For every numeric feature, the training bundle stores histogram bin edges and the share of training rows in each bin. For every categorical feature, it stores the training frequency table.

At request time:

```text
PSI = Σ (actual_share - expected_share) * ln(actual_share / expected_share)
```

with a small epsilon to avoid division by zero. Actual shares come from the recent prediction request window. Unseen categorical values are grouped into an `other` bucket.

Status thresholds:

| PSI | Status |
| --- | --- |
| < 0.10 | stable |
| 0.10 – 0.25 | watch |
| ≥ 0.25 | alert |

A window below 25 requests returns `insufficient_data` instead of a misleading score.

### Traffic and feedback

`prediction_summary` aggregates the last N hours by endpoint: count, average latency, p95 latency, and error count. `recent_events` returns the latest logged predictions for the dashboard.

`feedback_summary` joins `actual_value` back to the prediction:

- Maintenance: accuracy at 0.5, plus the count of labelled events
- Freight: MAE and MAPE on labelled quotes

The feedback endpoint is intentionally simple. A real system would ingest labels from a warehouse job. The API shape would stay the same.

### Prometheus

`/api/metrics` emits a small text exposition: app info, model availability, prediction event totals, and average latency by endpoint. It is scrape-friendly without pulling in a metrics library.

---

## 9. Orchestration

`src/pipeline.py` defines Prefect 3 flows:

- `refresh_flow`: ingest, train, evaluate
- `ingest_flow`: ingest only
- `train_flow`: train only
- `monitor_flow`: drift and feedback summary

Each flow calls the same functions as the CLI. Deployments use cron schedules and are created with `python -m src.pipeline --serve` or the CLI `pipeline` command.

The orchestration layer is thin on purpose. Prefect owns scheduling, retries, and observability. The domain logic stays testable without a Prefect server.

---

## 10. Deployment

### Local

SQLite, local MLflow store, `uvicorn`/Gunicorn, models from `models/<key>/current/`.

### Docker Compose

- PostgreSQL for the warehouse and MLflow backend
- MLflow server with a shared artifact volume
- API built from the same image, Gunicorn + Uvicorn workers, non-root user, health check
- Optional `init` profile for ingest + training inside the stack

The API image contains the promoted bundles, so a cold start does not require training. The entrypoint ingests raw CSVs only when the warehouse is empty, and it degrades gracefully if the data mount is absent.

### Cloud shape

The same image can run on ECS, Cloud Run, or Kubernetes. The only environment changes are `DATABASE_URL`, `MLFLOW_TRACKING_URI`, and optionally `S3_BUCKET` for the artefact mirror. `src/storage.py` can upload or download a bundle directory without making boto3 a hard dependency at import time.

---

## 11. Failure modes and guardrails

| Failure | Behavior |
| --- | --- |
| Missing model bundle | Registry records the error. `/api/ready` and the affected prediction route return 503. |
| Empty warehouse | API still boots. Analytics return zeros. Entrypoint can ingest. |
| Bad batch row | Row is reported in `errors`. The rest of the batch succeeds. |
| Drift window too small | Status is `insufficient_data`, not a false alert |
| MLflow unavailable | Local bundles still serve. Training with `--no-mlflow` still writes bundles. |
| Quality gate fails | Training raises and nothing is promoted. |
| Admin token set | Reload requires `X-Admin-Token` |
| S3 not configured | `StorageUnavailable` with a clear message. Core serving is unaffected. |

---

## 12. Why not more infrastructure?

A feature store, streaming ingestion, and a separate model server would all be defensible for a larger fleet. They would also make this project harder to verify. The current design keeps every step inspectable: a CSV becomes rows, rows become features, features become a bundle, the bundle becomes a response, and the response becomes a row in `prediction_events`. That traceability is the point.

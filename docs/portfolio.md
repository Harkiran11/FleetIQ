# Portfolio Notes

This page is for the human side of the project: how to describe it on a resume, what to emphasize in an interview, and which details are worth defending.

---

## 1. One-line summary

> Built an end-to-end MLOps system for fleet maintenance risk and freight pricing: 250k-row ingestion, leakage-safe feature engineering, MLflow tracking and registry, Prefect orchestration, FastAPI serving, PSI drift monitoring, a custom operations dashboard, 111 tests, and Docker deployment.

---

## 2. Resume bullets

Pick three or four; do not use all of them.

- Designed and shipped an end-to-end MLOps pipeline for a 250,000-vehicle logistics dataset, covering ingestion, validation, feature engineering, training, evaluation, registry promotion, and FastAPI serving.
- Built a leakage-safe maintenance risk model (HistGradientBoosting) with a tuned decision threshold and explainability signals; documented the synthetic-data caveat instead of presenting near-perfect metrics as production performance.
- Trained a freight pricing model with absolute-error loss and split-conformal 90% prediction intervals, achieving $13.15 MAE and 90.1% empirical coverage on a held-out test set.
- Implemented MLflow experiment tracking and a model registry with versioned runs, `champion` aliases, model cards, and local bundle promotion; added a quality gate that blocks promotion on AUC or MAPE regressions.
- Built PSI-based drift monitoring against stored training distributions, a prediction event log, a human feedback loop, and Prometheus text metrics.
- Delivered a dependency-free operations dashboard (HTML/CSS/JS, bundled fonts) with a custom design system, achieving Lighthouse 1.0 for accessibility, best practices, and SEO.
- Containerized the stack with PostgreSQL, MLflow, Gunicorn/Uvicorn, health checks, and a non-root image; added a Compose bootstrap job for ingest and retraining.
- Wrote 111 pytest tests with fixtures isolated from the production warehouse, plus ruff lint/format and a GitHub Actions CI workflow.

---

## 3. The 90-second walkthrough

> "FleetIQ is a maintenance and pricing system for a logistics fleet. The raw data is 250,000 vehicle records and 2,000 shipments. I built the whole path: ingestion into a SQLAlchemy warehouse, shared feature builders used by both training and serving, MLflow tracking and a model registry, Prefect flows, a FastAPI service, and a dashboard.
>
> The maintenance model is a histogram gradient boosting classifier. The public dataset is synthetic, so the metrics are near-perfect; the interesting work was making sure the model does not cheat. I explicitly excluded the post-service fields like days since last maintenance, severity, cost, and predictive score, and I documented that caveat in the model card. The freight model is a gradient boosting regressor with absolute-error loss and a split-conformal interval; it gets about 7.4% MAPE and 90% interval coverage.
>
> Operationally, every prediction is logged with its request, response, latency, and model version. Drift is measured with PSI against the training distribution, with a minimum window so the dashboard does not overreact. The API can serve local bundles or pull the champion from MLflow. The whole thing runs locally with SQLite or in Docker with PostgreSQL and an MLflow server. There are 111 tests and the suite runs in about three seconds."

---

## 4. Questions to expect

### "Your AUC is 1.0. Isn't that suspicious?"

Yes, and that is the right reaction. The dataset is synthetic and the target is a deterministic function of service state. I kept the near-perfect metrics visible because hiding them would be worse, but I also:

- Excluded the obvious leakage columns and tested the exclusion.
- Re-split the data instead of trusting the dataset's `data_split` label.
- Reported a baseline so the lift is visible.
- Put the caveat in the model card, the README, and the API metadata.
- Showed that `maintenance_type` alone carries most of the permutation importance.

The honest conclusion is that the pipeline is correct, not that the model would achieve AUC 1.0 on real fleet data.

### "Why histogram gradient boosting instead of XGBoost or a neural network?"

The dataset is tabular, the signal is mostly monotonic and interaction-heavy, and the rows fit comfortably in memory. HistGradientBoosting is fast, handles missing values natively, supports categorical features, and has no extra native dependency. XGBoost would also work; a neural network would add tuning and preprocessing cost without an obvious benefit at this size.

### "Why not a feature store?"

At this scale, the shared Python feature builders are enough. A feature store becomes worth the operational cost when multiple teams need the same features with different freshness requirements. The project is structured so that the feature contract is centralized in `src/features.py`; moving it behind a service later would not require rewriting training or serving.

### "How do you prevent train/serve skew?"

Training and serving import the same `prepare_maintenance_features` and `prepare_freight_features` functions. The feature lists and derived formulas live in one module. The test suite asserts the output schema, including the missing-column case. The model bundle stores the feature lists and the reference distribution, so a mismatch is visible at load time.

### "What happens when the model is wrong?"

Every prediction is logged. A human can attach the observed outcome through `/api/feedback/{event_id}`. The feedback summary computes maintenance accuracy and freight MAE/MAPE. If the model is wrong because the inputs moved, PSI flags it. If it is wrong because the world changed, the next retrain picks up the new labels. If a new model is worse, the registry alias or local bundle can be rolled back.

### "How do you know the interval is calibrated?"

The 0.9 residual quantile is computed on the validation set, and the test coverage is 90.14% against a nominal 90%. That is an empirical check, not a proof. The model card states the assumption: future shipments resemble the 2023 US lane mix.

### "What would you change for production?"

- Real work-order labels and cost-sensitive thresholds.
- A delayed-label pipeline instead of an API endpoint.
- Shadow deployment and promotion on a business metric.
- Authentication and a global request-size limit at the proxy.
- A proper metrics library and tracing.
- A feature service if multiple teams need the same features.
- A pricing guardrail with human approval for large deviations.

### "Why is the dashboard not React?"

The API is the product surface; the dashboard is a thin client over typed endpoints. A framework would add a build step, a Node dependency, and a second deployment artifact without changing the user experience. The current UI is about 3,000 lines of HTML/CSS/JS and loads instantly. If the frontend grew into a multi-team product, the next step would be a typed client and a component library, not a rewrite of the serving contract.

---

## 5. STAR stories

### Story: the index-alignment bug

**Situation:** The first full training run crashed during permutation importance with a `KeyError` because the sampled frame's index did not match the original frame.

**Task:** Fix the importance computation without masking the problem or dropping explainability.

**Action:** Reproduced it with a non-default index, reset the sampled frame index, zipped the sampled columns with the importance array using `strict=True`, and added a regression test that uses an index starting at 500.

**Result:** Full training completed, importance is stable, and the bug cannot silently return.

**What it shows:** Debugging discipline and the value of a small regression test for a subtle pandas issue.

### Story: the threshold that broke the UI

**Situation:** The tuned maintenance threshold was `0.9999975`. The original risk bands divided probability by threshold, so every score above the threshold landed in `critical` and everything below landed in `low`.

**Task:** Keep the tuned threshold but make the risk bands readable.

**Action:** Anchored the bands to multiples of the threshold (0.75x, 0.35x) instead of ratios, rounded the threshold in the response, and added tests for both a 0.5 threshold and a 0.9999 threshold.

**Result:** The UI shows a meaningful spread of risk levels without changing the model's decision boundary.

**What it shows:** Model behavior and product behavior are different concerns; a correct number can still produce a bad interface.

### Story: the form that would not submit

**Situation:** The maintenance form silently failed in the browser. The payload looked correct, but the API rejected it.

**Task:** Find the mismatch without weakening the API contract.

**Action:** Kept `extra="forbid"` and traced the form field names against the Pydantic schema. The number inputs also defaulted to `step="1"`, so decimal telemetry values were invalid HTML. Fixed the field names and set `step="any"` where appropriate.

**Result:** The form submits, the result card renders, and the API still rejects leakage fields.

**What it shows:** A strict contract is useful precisely because it surfaces client bugs early.

---

## 6. Mapping to roles

### ML / AI infrastructure (Huawei Noah's Ark style)

Emphasize:

- The full lifecycle: ingest → features → train → register → serve → monitor.
- MLflow tracking and registry, model cards, promotion gates, rollback.
- Prefect orchestration and the separation between flows and domain code.
- Containerization, health checks, non-root image, PostgreSQL.
- Monitoring and drift, not just model accuracy.
- The honest treatment of a synthetic dataset.

### ML co-op (Carfax style)

Emphasize:

- SQL analytics and warehouse modeling.
- Feature engineering with domain reasoning.
- Model evaluation beyond a single metric: MAE, MAPE, coverage, calibration, baseline lift.
- Typed API contracts and test coverage.
- The ability to explain trade-offs to non-ML stakeholders.

---

## 7. Demo script

1. Open the overview. Point out the 250,000 vehicles, the maintenance rate, and the model version in the registry card.
2. Open Maintenance. Click "Load high-risk example." Submit. Walk through the probability, the threshold, the risk band, the signals, and the parsed `ERR-431` / `184,000 miles`.
3. Open Freight. Submit the default lane. Point out the point estimate, the 90% interval, the cost per mile, and the observed lane benchmark.
4. Open Priority. Show the scored queue and the vehicle dialog.
5. Open Monitoring. Show the drift table and explain the minimum window. Open `/api/metrics` to show the Prometheus surface.
6. Open MLflow at `:5001`. Show the two runs, the metrics, the model versions, and the `champion` alias.
7. Run `pytest tests -q` and `ruff check src api tests` in the terminal.

Keep the caveat ready: the maintenance metrics are a property of the synthetic data, and the project is designed to make that obvious rather than impressive.

---

## 8. What not to claim

- Do not claim the maintenance model is production-ready.
- Do not claim the freight model generalizes beyond the 2023 US lane mix.
- Do not claim the PSI implementation is a full drift platform; it is a focused, tested monitor.
- Do not claim the dashboard is a design system for a team; it is a deliberate single-product UI.
- Do not claim the project handles streaming or online learning. It is batch training plus online serving.

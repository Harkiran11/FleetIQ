"""Drift and operational monitoring for served models.

The module is deliberately dependency-light: it only needs numpy, pandas and
SQLAlchemy. Two things are measured:

* Population Stability Index (PSI) for every model input, comparing the live
  request window against the training reference distribution stored next to
  each model bundle.
* Prediction and feedback summaries read from the ``prediction_events`` table.

PSI thresholds follow the usual industry convention: below 0.10 is stable,
0.10-0.25 deserves a look, above 0.25 means the input distribution has moved
enough that the model should be re-validated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.config import settings
from src.db_models import PredictionEvent
from src.features import prepare_freight_features, prepare_maintenance_features
from src.registry import LoadedModel

EPSILON = 1e-6


def _shares(counts: Sequence[float]) -> list[float]:
    total = float(sum(counts))
    if total <= 0:
        return [0.0 for _ in counts]
    return [float(count) / total for count in counts]


def population_stability_index(expected: Sequence[float], actual: Sequence[float]) -> float:
    """Symmetric PSI between two probability vectors of equal length."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same length")
    psi = 0.0
    for expected_share, actual_share in zip(expected, actual, strict=True):
        e = max(float(expected_share), EPSILON)
        a = max(float(actual_share), EPSILON)
        psi += (a - e) * math.log(a / e)
    return float(max(psi, 0.0))


def _numeric_psi(values: pd.Series, stats: dict[str, Any]) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    edges = stats.get("bin_edges") or []
    if len(edges) < 2:
        return None
    edges = np.asarray(edges, dtype=float)
    clipped = np.clip(numeric.to_numpy(dtype=float), edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    actual = _shares(counts)

    expected = stats.get("bin_share")
    if not expected or len(expected) != len(actual):
        # Older bundles only stored quantile edges; quantile bins are close to
        # equal by construction, so fall back to a uniform expectation.
        expected = [1.0 / len(actual)] * len(actual)
    return population_stability_index(expected, actual)


def _categorical_psi(values: pd.Series, stats: dict[str, Any]) -> float | None:
    frequency: dict[str, float] = stats.get("frequency", {})
    if not frequency:
        return None
    cleaned = values.astype("string").fillna("unknown")
    known = set(frequency)
    counts: dict[str, int] = dict.fromkeys(known, 0)
    other = 0
    for value in cleaned:
        text = str(value)
        if text in known:
            counts[text] += 1
        else:
            other += 1
    actual_counts = [*counts.values(), other]
    expected_counts = [float(frequency[key]) for key in known] + [
        max(0.0, 1.0 - sum(frequency.values()))
    ]
    return population_stability_index(_shares(expected_counts), _shares(actual_counts))


def _status(psi: float | None, watch: float, alert: float) -> str:
    if psi is None:
        return "no_data"
    if psi >= alert:
        return "alert"
    if psi >= watch:
        return "watch"
    return "stable"


def _prepare_frame(model: LoadedModel, payloads: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(payloads))
    if frame.empty:
        return frame
    if model.model_key == "maintenance":
        return prepare_maintenance_features(frame)
    if model.model_key == "freight":
        return prepare_freight_features(frame)
    raise ValueError(f"Unsupported model key: {model.model_key}")


def build_drift_report(
    model: LoadedModel,
    payloads: Iterable[dict[str, Any]],
    scores: Iterable[float] | None = None,
    watch: float | None = None,
    alert: float | None = None,
) -> dict[str, Any]:
    """Compare a live request window with the training reference."""
    watch = settings.psi_watch if watch is None else watch
    alert = settings.psi_alert if alert is None else alert
    payload_list = list(payloads)
    features = _prepare_frame(model, payload_list)
    reference = model.reference_stats

    rows: list[dict[str, Any]] = []
    if not features.empty:
        for column in model.numeric_features:
            if column not in features.columns:
                continue
            psi = _numeric_psi(features[column], reference.get("numeric", {}).get(column, {}))
            rows.append(
                {
                    "feature": column,
                    "kind": "numeric",
                    "psi": round(psi, 5) if psi is not None else None,
                    "status": _status(psi, watch, alert),
                }
            )
        for column in model.categorical_features:
            if column not in features.columns:
                continue
            psi = _categorical_psi(
                features[column], reference.get("categorical", {}).get(column, {})
            )
            rows.append(
                {
                    "feature": column,
                    "kind": "categorical",
                    "psi": round(psi, 5) if psi is not None else None,
                    "status": _status(psi, watch, alert),
                }
            )

    scored = [float(value) for value in (scores or []) if value is not None]
    prediction_stats: dict[str, Any] = {}
    if scored:
        series = pd.Series(scored, dtype="float64")
        prediction_stats = {
            "count": len(series),
            "mean": round(float(series.mean()), 4),
            "p50": round(float(series.median()), 4),
            "p90": round(float(series.quantile(0.90)), 4),
            "p99": round(float(series.quantile(0.99)), 4),
            "max": round(float(series.max()), 4),
        }

    ranked = sorted(
        [row for row in rows if row["psi"] is not None],
        key=lambda row: row["psi"],
        reverse=True,
    )
    max_psi = ranked[0]["psi"] if ranked else None
    overall = _status(max_psi, watch, alert)
    if len(payload_list) < 25:
        overall = "insufficient_data"

    return {
        "model": model.model_key,
        "model_version": model.version,
        "generated_at": datetime.now(UTC).isoformat(),
        "window": len(payload_list),
        "min_window": 25,
        "status": overall,
        "max_psi": max_psi,
        "thresholds": {"watch": watch, "alert": alert},
        "features": rows,
        "top_drifted": ranked[:8],
        "prediction_stats": prediction_stats,
    }


def recent_events(session: Session, limit: int = 50) -> list[PredictionEvent]:
    statement = select(PredictionEvent).order_by(PredictionEvent.created_at.desc()).limit(limit)
    return list(session.execute(statement).scalars().all())


def drift_from_events(
    model: LoadedModel,
    session: Session,
    window: int | None = None,
) -> dict[str, Any]:
    """Build a drift report from logged prediction requests for one model."""
    window = window or settings.drift_window
    statement = (
        select(PredictionEvent)
        .where(PredictionEvent.model_name == model.metadata.get("model_name", model.model_key))
        .order_by(PredictionEvent.created_at.desc())
        .limit(window)
    )
    events = list(session.execute(statement).scalars().all())
    payloads = [event.request for event in events if event.request]
    scores = [
        event.risk_probability if event.risk_probability is not None else event.predicted_cost
        for event in events
    ]
    report = build_drift_report(model, payloads, scores)
    report["source"] = "prediction_events"
    return report


def prediction_summary(session: Session, hours: int = 24) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=hours)
    rows = session.execute(
        select(
            PredictionEvent.endpoint,
            func.count(PredictionEvent.id),
            func.avg(PredictionEvent.latency_ms),
        )
        .where(PredictionEvent.created_at >= since)
        .group_by(PredictionEvent.endpoint)
    ).all()
    total = int(
        session.execute(
            select(func.count(PredictionEvent.id)).where(PredictionEvent.created_at >= since)
        ).scalar_one()
    )
    return {
        "window_hours": hours,
        "total": total,
        "by_endpoint": [
            {
                "endpoint": endpoint,
                "count": int(count),
                "avg_latency_ms": round(float(latency or 0.0), 2),
            }
            for endpoint, count, latency in rows
        ],
    }


def feedback_summary(session: Session, limit: int = 2000) -> dict[str, Any]:
    """Error metrics from events where an operator supplied the actual value."""
    statement = (
        select(PredictionEvent)
        .where(PredictionEvent.actual_value.is_not(None))
        .order_by(PredictionEvent.feedback_at.desc())
        .limit(limit)
    )
    events = list(session.execute(statement).scalars().all())
    maintenance = [
        event
        for event in events
        if event.risk_probability is not None and event.actual_value is not None
    ]
    freight = [
        event
        for event in events
        if event.predicted_cost is not None and event.actual_value is not None
    ]

    payload: dict[str, Any] = {
        "labeled_events": len(events),
        "maintenance": None,
        "freight": None,
    }
    if maintenance:
        predicted = np.array([event.risk_probability for event in maintenance], dtype=float)
        actual = np.array([event.actual_value for event in maintenance], dtype=float)
        correct = ((predicted >= 0.5) == (actual >= 0.5)).mean()
        payload["maintenance"] = {
            "count": len(maintenance),
            "accuracy_at_0_5": round(float(correct), 4),
            "mean_probability": round(float(predicted.mean()), 4),
            "observed_rate": round(float(actual.mean()), 4),
        }
    if freight:
        predicted = np.array([event.predicted_cost for event in freight], dtype=float)
        actual = np.array([event.actual_value for event in freight], dtype=float)
        mask = actual > 0
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
        payload["freight"] = {
            "count": len(freight),
            "mae": round(float(np.mean(np.abs(actual - predicted))), 2),
            "mape": round(mape, 2),
        }
    return payload

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.db_models import PredictionEvent
from src.features import fetch_maintenance_frame
from src.monitoring import (
    _categorical_psi,
    _numeric_psi,
    build_drift_report,
    feedback_summary,
    population_stability_index,
    prediction_summary,
    recent_events,
)


def test_population_stability_index_is_zero_for_identical_shares():
    assert population_stability_index([0.25] * 4, [0.25] * 4) == pytest.approx(0.0)


def test_population_stability_index_detects_a_shift():
    shifted = population_stability_index([0.5, 0.5], [0.9, 0.1])
    assert shifted > 0.25


def test_population_stability_index_requires_equal_lengths():
    with pytest.raises(ValueError):
        population_stability_index([0.5, 0.5], [1.0])


def test_numeric_psi_uses_stored_bin_share():
    stats = {
        "bin_edges": [0.0, 1.0, 2.0, 3.0, 4.0],
        "bin_share": [0.25, 0.25, 0.25, 0.25],
    }
    stable = pd.Series([0.5, 1.5, 2.5, 3.5] * 25)
    shifted = pd.Series([3.5] * 100)

    assert _numeric_psi(stable, stats) == pytest.approx(0.0, abs=1e-6)
    assert _numeric_psi(shifted, stats) > 0.25


def test_numeric_psi_falls_back_to_uniform_expectation():
    stats = {"bin_edges": [0.0, 1.0, 2.0, 3.0, 4.0]}

    psi = _numeric_psi(pd.Series([0.5, 1.5, 2.5, 3.5] * 25), stats)

    assert psi == pytest.approx(0.0, abs=1e-6)


def test_numeric_psi_returns_none_without_data():
    assert _numeric_psi(pd.Series([], dtype=float), {"bin_edges": [0, 1]}) is None
    assert _numeric_psi(pd.Series([1.0]), {}) is None


def test_categorical_psi_tracks_unseen_values_as_other():
    stats = {"frequency": {"Good": 0.5, "Poor": 0.5}}

    assert _categorical_psi(pd.Series(["Good", "Poor"] * 50), stats) == pytest.approx(0.0, abs=1e-6)
    assert _categorical_psi(pd.Series(["Unknown"] * 100), stats) > 0.25


def test_categorical_psi_returns_none_without_frequency():
    assert _categorical_psi(pd.Series(["Good"]), {}) is None


def test_build_drift_report_on_a_real_training_sample(engine, maintenance_model):
    frame = fetch_maintenance_frame(engine, limit=40)
    payloads = frame.to_dict("records")

    report = build_drift_report(maintenance_model, payloads)

    assert report["model"] == "maintenance"
    assert report["window"] == 40
    assert report["min_window"] == 25
    assert report["status"] in {"stable", "watch", "alert"}
    assert report["max_psi"] is not None
    assert len(report["features"]) == len(maintenance_model.numeric_features) + len(
        maintenance_model.categorical_features
    )
    assert len(report["top_drifted"]) <= 8
    assert all({"feature", "kind", "psi", "status"} <= set(row) for row in report["features"])


def test_build_drift_report_marks_small_windows_as_insufficient(
    maintenance_model, maintenance_payload
):
    report = build_drift_report(maintenance_model, [maintenance_payload] * 3)

    assert report["status"] == "insufficient_data"
    assert report["window"] == 3


def test_build_drift_report_summarises_scores(maintenance_model, maintenance_payload):
    report = build_drift_report(
        maintenance_model,
        [maintenance_payload] * 30,
        scores=[0.1] * 20 + [0.9] * 10,
    )

    stats = report["prediction_stats"]
    assert stats["count"] == 30
    assert stats["mean"] == pytest.approx(0.3667, abs=1e-4)
    assert stats["max"] == pytest.approx(0.9)


def test_prediction_summary_and_recent_events(session):
    endpoint = "test://monitoring/summary"
    events = [
        PredictionEvent(
            endpoint=endpoint,
            model_name="test-model",
            model_version="1",
            run_id="run-test",
            request={"vehicle_id": index},
            response={"status": "success"},
            latency_ms=12.5,
            risk_probability=0.4,
            client="pytest",
        )
        for index in range(3)
    ]
    session.add_all(events)
    session.commit()

    summary = prediction_summary(session, hours=24)
    assert summary["total"] >= 3
    row = next(item for item in summary["by_endpoint"] if item["endpoint"] == endpoint)
    assert row["count"] == 3
    assert row["avg_latency_ms"] == pytest.approx(12.5)

    recent = recent_events(session, limit=5)
    assert recent
    assert all(event.created_at is not None for event in recent)


def test_feedback_summary_computes_maintenance_and_freight_errors(session):
    now = datetime.now(UTC)
    events = [
        PredictionEvent(
            endpoint="test://feedback/maintenance",
            model_name="test-maintenance",
            model_version="1",
            request={},
            response={},
            latency_ms=1.0,
            risk_probability=0.9,
            actual_value=1.0,
            feedback_at=now,
        ),
        PredictionEvent(
            endpoint="test://feedback/maintenance",
            model_name="test-maintenance",
            model_version="1",
            request={},
            response={},
            latency_ms=1.0,
            risk_probability=0.1,
            actual_value=0.0,
            feedback_at=now,
        ),
        PredictionEvent(
            endpoint="test://feedback/freight",
            model_name="test-freight",
            model_version="1",
            request={},
            response={},
            latency_ms=1.0,
            predicted_cost=100.0,
            actual_value=110.0,
            feedback_at=now,
        ),
        PredictionEvent(
            endpoint="test://feedback/freight",
            model_name="test-freight",
            model_version="1",
            request={},
            response={},
            latency_ms=1.0,
            predicted_cost=200.0,
            actual_value=190.0,
            feedback_at=now,
        ),
    ]
    session.add_all(events)
    session.commit()

    summary = feedback_summary(session)

    assert summary["labeled_events"] >= 4
    assert summary["maintenance"]["count"] >= 2
    assert summary["maintenance"]["accuracy_at_0_5"] == pytest.approx(1.0)
    assert summary["freight"]["count"] >= 2
    assert summary["freight"]["mae"] >= 0
    assert summary["freight"]["mape"] >= 0


def test_feedback_summary_without_labels_returns_empty_payload(session):
    session.query(PredictionEvent).filter(PredictionEvent.endpoint.like("test://empty%")).delete(
        synchronize_session=False
    )
    session.commit()

    summary = feedback_summary(session, limit=0)

    assert summary["labeled_events"] == 0
    assert summary["maintenance"] is None
    assert summary["freight"] is None


def test_numeric_psi_is_non_negative_for_numpy_shares():
    # Regression guard: the implementation clamps tiny floating point noise.
    assert population_stability_index(np.array([0.2, 0.8]), np.array([0.2, 0.8])) >= 0.0

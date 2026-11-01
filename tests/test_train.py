from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.config import settings
from src.train import (
    _classification_metrics,
    _fingerprint,
    _importance,
    _plot_confusion,
    _plot_importance,
    _preprocessor,
    _regression_metrics,
    _safe_json,
    _safe_metrics,
    _tune_threshold,
    evaluate_gate,
)


def test_tune_threshold_maximises_f1():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    assert _tune_threshold(y_true, probabilities) == pytest.approx(0.6)


def test_classification_metrics_reports_confusion_and_scores():
    metrics = _classification_metrics(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), threshold=0.5
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert metrics["brier"] < 0.05


def test_regression_metrics_reports_error_and_interval_coverage():
    metrics = _regression_metrics(
        np.array([100.0, 200.0]),
        np.array([110.0, 190.0]),
        interval=20.0,
        baseline_predictions=np.array([120.0, 180.0]),
    )

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["medae"] == pytest.approx(10.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["mape"] == pytest.approx(7.5)
    assert metrics["r2"] == pytest.approx(0.96)
    assert metrics["coverage_90"] == pytest.approx(1.0)
    assert metrics["baseline_mae"] == pytest.approx(20.0)
    assert metrics["baseline_mape"] == pytest.approx(15.0)


def test_safe_metrics_drops_non_finite_values():
    cleaned = _safe_metrics(
        {
            "good": 1,
            "numpy": np.float64(2.5),
            "nan": float("nan"),
            "inf": float("inf"),
            "none": None,
        }
    )

    assert cleaned == {"good": 1.0, "numpy": 2.5}


def test_safe_json_converts_nested_numpy_and_nan():
    payload = _safe_json(
        {
            "nested": {"value": np.int64(3), "bad": np.float64("nan")},
            "items": [np.float32(1.5), float("inf")],
        }
    )

    assert payload == {"nested": {"value": 3, "bad": None}, "items": [1.5, None]}


def test_fingerprint_is_stable_and_order_sensitive():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)

    assert _fingerprint(frame, ["a", "b"]) == _fingerprint(frame, ["a", "b"])
    assert _fingerprint(frame, ["a", "b"]) != _fingerprint(reversed_frame, ["a", "b"])


def test_importance_handles_a_non_default_index():
    frame = pd.DataFrame(
        {
            "usage_hours": np.linspace(0, 100, 80),
            "brake_condition": ["Good"] * 40 + ["Poor"] * 40,
        },
        index=range(500, 580),
    )
    target = pd.Series((frame["usage_hours"] > 50).astype(int).to_numpy(), index=frame.index)
    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor(["usage_hours"], ["brake_condition"])),
            (
                "model",
                HistGradientBoostingClassifier(max_iter=10, early_stopping=False, random_state=42),
            ),
        ]
    )
    pipeline.fit(frame, target)

    importance = _importance(pipeline, frame, target, scoring="roc_auc", sample_size=40)

    assert len(importance) == 2
    assert {item["feature"] for item in importance} == {"usage_hours", "brake_condition"}
    assert all(item["importance"] >= 0 for item in importance)


def test_evaluate_gate_passes_good_metrics(monkeypatch):
    monkeypatch.setattr(settings, "min_maintenance_auc", 0.75)
    monkeypatch.setattr(settings, "max_freight_mape", 25.0)

    evaluate_gate(
        {
            "maintenance": {"metrics": {"roc_auc": 0.9}},
            "freight": {"metrics": {"mape": 12.0}},
        }
    )


def test_evaluate_gate_raises_on_bad_metrics(monkeypatch):
    monkeypatch.setattr(settings, "min_maintenance_auc", 0.75)
    monkeypatch.setattr(settings, "max_freight_mape", 25.0)

    with pytest.raises(RuntimeError) as error:
        evaluate_gate(
            {
                "maintenance": {"metrics": {"roc_auc": 0.5}},
                "freight": {"metrics": {"mape": 40.0}},
            }
        )

    message = str(error.value)
    assert "maintenance ROC AUC" in message
    assert "freight MAPE" in message


def test_plot_helpers_write_png_files(tmp_path):
    importance = [
        {"feature": "usage_hours", "importance": 0.2},
        {"feature": "brake_condition", "importance": 0.1},
    ]
    confusion = {"confusion": {"tn": 10, "fp": 1, "fn": 2, "tp": 7}}

    importance_path = tmp_path / "importance.png"
    confusion_path = tmp_path / "confusion.png"
    _plot_importance(importance, importance_path, "Drivers")
    _plot_confusion(confusion, confusion_path)

    assert importance_path.stat().st_size > 0
    assert confusion_path.stat().st_size > 0

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import settings
from src.db import get_engine, init_db
from src.features import (
    FREIGHT_CATEGORICAL,
    FREIGHT_FEATURES,
    FREIGHT_NUMERIC,
    MAINTENANCE_CATEGORICAL,
    MAINTENANCE_EXCLUDED_COLUMNS,
    MAINTENANCE_FEATURES,
    MAINTENANCE_NUMERIC,
    fetch_maintenance_frame,
    fetch_shipment_frame,
    prepare_freight_features,
    prepare_maintenance_features,
)
from src.registry import (
    build_reference_stats,
    render_model_card,
    write_bundle,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
INK = "#16212a"
ACCENT = "#d95f2b"
TEAL = "#1f7a6d"
GRID = "#d8d3c7"


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    sample = frame[columns].head(20000)
    hashed = pd.util.hash_pandas_object(sample, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()[:16]


def _safe_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            continue
        cleaned[key] = number
    return cleaned


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if np.isnan(number) or np.isinf(number) else number
    if isinstance(value, float):
        return None if np.isnan(value) or np.isinf(value) else value
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _preprocessor(
    numeric: list[str], categorical: list[str], one_hot: bool = False
) -> ColumnTransformer:
    if one_hot:
        categorical_step = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    else:
        categorical_step = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("categorical", categorical_step, categorical),
        ],
        verbose_feature_names_out=False,
    )


def _categorical_mask(numeric: list[str], categorical: list[str]) -> list[bool]:
    return [False] * len(numeric) + [True] * len(categorical)


def _maintenance_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    config = {
        "max_iter": 350,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 25,
        "random_state": RANDOM_STATE,
        "categorical_features": _categorical_mask(MAINTENANCE_NUMERIC, MAINTENANCE_CATEGORICAL),
    }
    config.update(params or {})
    classifier = HistGradientBoostingClassifier(**config)
    return Pipeline(
        [
            ("preprocess", _preprocessor(MAINTENANCE_NUMERIC, MAINTENANCE_CATEGORICAL)),
            ("model", classifier),
        ]
    )


def _freight_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    config = {
        "loss": "absolute_error",
        "max_iter": 500,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 30,
        "random_state": RANDOM_STATE,
        "categorical_features": _categorical_mask(FREIGHT_NUMERIC, FREIGHT_CATEGORICAL),
    }
    config.update(params or {})
    regressor = HistGradientBoostingRegressor(**config)
    return Pipeline(
        [
            ("preprocess", _preprocessor(FREIGHT_NUMERIC, FREIGHT_CATEGORICAL)),
            ("model", regressor),
        ]
    )


def _logistic_baseline() -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                _preprocessor(MAINTENANCE_NUMERIC, MAINTENANCE_CATEGORICAL, one_hot=True),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def _tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    best = float(np.max(f1))
    candidates = thresholds[f1 >= best - 1e-9]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def _classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, np.clip(probabilities, 1e-6, 1 - 1e-6))),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _regression_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    interval: float,
    baseline_predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    predictions = np.clip(predictions, 0, None)
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100)
    coverage = float(np.mean(np.abs(y_true - predictions) <= interval))
    metrics: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "medae": float(median_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "mape": mape,
        "r2": float(r2_score(y_true, predictions)),
        "coverage_90": coverage,
    }
    if baseline_predictions is not None:
        baseline_predictions = np.clip(baseline_predictions, 0, None)
        metrics["baseline_mae"] = float(mean_absolute_error(y_true, baseline_predictions))
        metrics["baseline_mape"] = float(
            np.mean(np.abs((y_true[mask] - baseline_predictions[mask]) / y_true[mask])) * 100
        )
    return metrics


def _importance(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    target: pd.Series,
    scoring: str,
    sample_size: int = 4000,
) -> list[dict[str, Any]]:
    work = frame.reset_index(drop=True)
    target_series = pd.Series(np.asarray(target)).reset_index(drop=True)
    if len(work) > sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        positions = rng.choice(len(work), size=sample_size, replace=False)
        sampled = work.iloc[positions]
        target_sample = target_series.iloc[positions]
    else:
        sampled = work
        target_sample = target_series
    result = permutation_importance(
        pipeline,
        sampled,
        target_sample,
        scoring=scoring,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    ranked = [
        {"feature": feature, "importance": float(max(0.0, value))}
        for feature, value in zip(sampled.columns, result.importances_mean, strict=True)
    ]
    return sorted(ranked, key=lambda item: item["importance"], reverse=True)[:15]


def _plot_importance(importance: list[dict[str, Any]], path: Path, title: str) -> None:
    top = importance[:12][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = [item["feature"].replace("_", " ") for item in top]
    values = [item["importance"] for item in top]
    ax.barh(labels, values, color=TEAL, height=0.62)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel("Permutation importance")
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(metrics: dict[str, Any], path: Path) -> None:
    matrix = np.array(
        [
            [metrics["confusion"]["tn"], metrics["confusion"]["fp"]],
            [metrics["confusion"]["fn"], metrics["confusion"]["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap="Oranges")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color=INK,
                fontsize=12,
            )
    ax.set_xticks([0, 1], ["Predicted clear", "Predicted due"])
    ax.set_yticks([0, 1], ["Actual clear", "Actual due"])
    ax.set_title("Confusion matrix", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_calibration(y_true: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    fraction_positive, mean_predicted = calibration_curve(y_true, probabilities, n_bins=10)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRID, linewidth=1)
    ax.plot(mean_predicted, fraction_positive, marker="o", color=ACCENT, linewidth=2)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed maintenance rate")
    ax.set_title("Calibration", loc="left", fontsize=12, color=INK)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_residuals(y_true: np.ndarray, predictions: np.ndarray, path: Path) -> None:
    residuals = y_true - predictions
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(predictions, residuals, s=12, alpha=0.45, color=TEAL, edgecolors="none")
    ax.axhline(0, color=ACCENT, linewidth=1.2)
    ax.set_xlabel("Predicted cost (USD)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals on holdout set", loc="left", fontsize=12, color=INK)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_prediction_actual(y_true: np.ndarray, predictions: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(y_true, predictions, s=12, alpha=0.45, color=TEAL, edgecolors="none")
    upper = float(max(np.percentile(y_true, 99), np.percentile(predictions, 99)))
    ax.plot([0, upper], [0, upper], linestyle="--", color=ACCENT, linewidth=1.2)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Actual cost (USD)")
    ax.set_ylabel("Predicted cost (USD)")
    ax.set_title("Holdout predictions", loc="left", fontsize=12, color=INK)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _maintenance_segments(
    frame: pd.DataFrame, y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    work = frame.reset_index(drop=True).copy()
    work["_y"] = y_true
    work["_p"] = probabilities
    segments: dict[str, Any] = {}
    for column in ("vehicle_type", "brake_condition", "route_info"):
        rows: list[dict[str, Any]] = []
        for value, group in work.groupby(column, dropna=False):
            y_group = group["_y"].to_numpy()
            p_group = group["_p"].to_numpy()
            row = {
                "segment": str(value),
                "rows": len(group),
                "positive_rate": float(y_group.mean()),
                "mean_prediction": float(p_group.mean()),
                "f1": float(f1_score(y_group, (p_group >= threshold).astype(int), zero_division=0)),
                "auc": None,
            }
            if len(np.unique(y_group)) > 1:
                row["auc"] = float(roc_auc_score(y_group, p_group))
            rows.append(row)
        segments[column] = sorted(rows, key=lambda item: item["rows"], reverse=True)
    return segments


def _freight_segments(
    frame: pd.DataFrame, y_true: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    work = frame.reset_index(drop=True).copy()
    work["_y"] = y_true
    work["_p"] = predictions
    work["_distance_band"] = pd.cut(
        work["distance_miles"],
        bins=[0, 250, 500, 1000, 1500, 2500, np.inf],
        labels=["<250", "250-500", "500-1000", "1000-1500", "1500-2500", "2500+"],
        right=False,
    ).astype("string")
    segments: dict[str, Any] = {}
    for column in ("carrier", "_distance_band"):
        rows: list[dict[str, Any]] = []
        for value, group in work.groupby(column, dropna=False):
            actual = group["_y"].to_numpy()
            predicted = np.clip(group["_p"].to_numpy(), 0, None)
            mask = actual > 0
            rows.append(
                {
                    "segment": str(value),
                    "rows": len(group),
                    "mae": float(mean_absolute_error(actual, predicted)),
                    "mape": float(
                        np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
                    ),
                    "mean_actual": float(actual.mean()),
                    "mean_predicted": float(predicted.mean()),
                }
            )
        key = "distance_band" if column == "_distance_band" else column
        segments[key] = sorted(rows, key=lambda item: item["rows"], reverse=True)
    return segments


def _maintenance_metadata(
    run_id: str,
    version: str | None,
    threshold: float,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    params: dict[str, Any],
    rows: dict[str, int],
    target_rate: float,
    fingerprint: str,
    importance: list[dict[str, Any]],
    segments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_key": "maintenance",
        "model_name": settings.mlflow_registered_model_maintenance,
        "title": "FleetIQ Maintenance Risk",
        "run_id": run_id,
        "trained_at": datetime.now(UTC).isoformat(),
        "reference_year": settings.reference_year,
        "training_rows": rows["train"],
        "validation_rows": rows["validation"],
        "test_rows": rows["test"],
        "target_rate": target_rate,
        "feature_columns": MAINTENANCE_FEATURES,
        "numeric_features": MAINTENANCE_NUMERIC,
        "categorical_features": MAINTENANCE_CATEGORICAL,
        "excluded_columns": sorted(MAINTENANCE_EXCLUDED_COLUMNS),
        "threshold": threshold,
        "metrics": metrics,
        "baseline_metrics": baseline,
        "params": params,
        "segments": segments,
        "global_importance": importance,
        "dataset": {
            "source": "logistics_predictive_maintenanceV2.csv",
            "fingerprint": fingerprint,
            "target": "maintenance_required",
            "target_rate": target_rate,
        },
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment": settings.mlflow_experiment_maintenance,
            "registered_model": settings.mlflow_registered_model_maintenance,
            "version": version,
        },
        "intended_use": (
            "Rank vehicles for scheduled maintenance using telemetry and the last "
            "known service state."
        ),
        "limitations": (
            "The public dataset is synthetic and encodes maintenance need as a deterministic "
            "function of service age and telemetry. Discrimination is therefore near-perfect. "
            "Days since last maintenance, severity, cost and post-service fields are excluded "
            "to avoid direct leakage. Re-validate on real fleet data before operational use."
        ),
    }


def _freight_metadata(
    run_id: str,
    version: str | None,
    metrics: dict[str, Any],
    params: dict[str, Any],
    rows: dict[str, int],
    fingerprint: str,
    importance: list[dict[str, Any]],
    segments: dict[str, Any],
    residual_quantiles: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_key": "freight",
        "model_name": settings.mlflow_registered_model_freight,
        "title": "FleetIQ Freight Pricing",
        "run_id": run_id,
        "trained_at": datetime.now(UTC).isoformat(),
        "reference_year": settings.reference_year,
        "training_rows": rows["train"],
        "validation_rows": rows["validation"],
        "test_rows": rows["test"],
        "feature_columns": FREIGHT_FEATURES,
        "numeric_features": FREIGHT_NUMERIC,
        "categorical_features": FREIGHT_CATEGORICAL,
        "excluded_columns": [
            "shipment_id",
            "delivery_date",
            "status",
            "transit_days",
            "cost",
        ],
        "metrics": metrics,
        "params": params,
        "segments": segments,
        "global_importance": importance,
        "residual_quantiles": residual_quantiles,
        "dataset": {
            "source": "logistics_shipments_dataset.csv",
            "fingerprint": fingerprint,
            "target": "cost",
        },
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment": settings.mlflow_experiment_freight,
            "registered_model": settings.mlflow_registered_model_freight,
            "version": version,
        },
        "intended_use": (
            "Produce a booking-time freight quote and a calibrated 90% cost interval for a lane."
        ),
        "limitations": (
            "Freight cost is heavy-tailed. The point model optimises absolute error, so R2 is low "
            "while MAE and MAPE are the useful operational metrics. The 90% interval is a split-"
            "conformal estimate and assumes future shipments resemble the 2023 US lane mix."
        ),
    }


def _log_common(
    experiment: str,
    registered_name: str,
    pipeline: Pipeline,
    signature,
    input_example: pd.DataFrame,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Path],
    metadata: dict[str, Any],
    reference_stats: dict[str, Any],
) -> str | None:
    """Log a trained pipeline to the active MLflow run and register it."""
    mlflow.set_experiment(experiment)
    mlflow.log_params({key: value for key, value in params.items() if value is not None})
    mlflow.log_metrics(_safe_metrics(metrics))
    mlflow.set_tags(
        {
            "git_commit": _git_commit() or "not-a-git-repo",
            "pipeline": "fleetiq",
            "model_family": "hist_gradient_boosting",
        }
    )
    for path in artifacts.values():
        mlflow.log_artifact(str(path), artifact_path="plots")
    mlflow.log_dict(_safe_json(metadata.get("segments", {})), "segments.json")
    mlflow.log_dict(_safe_json(metadata.get("global_importance", [])), "global_importance.json")
    mlflow.log_dict(_safe_json(metadata.get("error_analysis", [])), "error_analysis.json")

    info = mlflow.sklearn.log_model(
        pipeline,
        name="model",
        signature=signature,
        input_example=input_example,
        serialization_format="cloudpickle",
    )
    version: str | None = None
    try:
        model_version = mlflow.register_model(info.model_uri, registered_name)
        version = str(model_version.version)
        client = MlflowClient()
        client.set_registered_model_alias(registered_name, "champion", version)
        client.update_model_version(
            name=registered_name,
            version=version,
            description=metadata.get("intended_use", ""),
        )
    except Exception as exc:
        logger.warning("Model registration skipped: %s", exc)

    metadata.setdefault("mlflow", {})["version"] = version
    mlflow.log_text(render_model_card(metadata), "model_card.md")
    mlflow.log_dict(_safe_json(metadata), "metadata.json")
    mlflow.log_dict(_safe_json(reference_stats), "reference_stats.json")
    return version


def train_maintenance(
    sample: int | None = None,
    promote: bool | None = None,
    log_to_mlflow: bool | None = None,
) -> dict[str, Any]:
    engine = get_engine()
    init_db(engine)
    frame = fetch_maintenance_frame(engine)
    if frame.empty:
        raise RuntimeError("maintenance_records is empty. Run `make ingest` first.")
    if sample:
        frame = frame.sample(n=min(sample, len(frame)), random_state=RANDOM_STATE)

    if {"Train", "Validation", "Test"}.issubset(set(frame["data_split"].dropna())):
        train = frame[frame["data_split"] == "Train"].copy()
        validation = frame[frame["data_split"] == "Validation"].copy()
        test = frame[frame["data_split"] == "Test"].copy()
    else:
        train, rest = train_test_split(
            frame, test_size=0.30, random_state=RANDOM_STATE, stratify=frame["maintenance_required"]
        )
        validation, test = train_test_split(
            rest,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=rest["maintenance_required"],
        )

    X_train = prepare_maintenance_features(train)
    X_val = prepare_maintenance_features(validation)
    X_test = prepare_maintenance_features(test)
    y_train = train["maintenance_required"].astype(int).reset_index(drop=True)
    y_val = validation["maintenance_required"].astype(int).reset_index(drop=True)
    y_test = test["maintenance_required"].astype(int).reset_index(drop=True)

    params = {
        "max_iter": 350,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "reference_year": settings.reference_year,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "feature_count": len(MAINTENANCE_FEATURES),
    }
    pipeline = _maintenance_pipeline(
        {
            "max_iter": params["max_iter"],
            "learning_rate": params["learning_rate"],
            "max_leaf_nodes": params["max_leaf_nodes"],
            "l2_regularization": params["l2_regularization"],
        }
    )
    pipeline.fit(X_train, y_train)
    val_probability = pipeline.predict_proba(X_val)[:, 1]
    threshold = _tune_threshold(y_val.to_numpy(), val_probability)
    test_probability = pipeline.predict_proba(X_test)[:, 1]
    metrics = _classification_metrics(y_test.to_numpy(), test_probability, threshold)
    val_metrics = _classification_metrics(y_val.to_numpy(), val_probability, threshold)

    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    baseline_probability = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = _classification_metrics(y_test.to_numpy(), baseline_probability, 0.5)

    importance = _importance(pipeline, X_test, y_test, scoring="roc_auc")
    segments = _maintenance_segments(test, y_test.to_numpy(), test_probability, threshold)

    test_work = test.reset_index(drop=True).copy()
    test_work["_y"] = y_test.to_numpy()
    test_work["_p"] = test_probability
    errors = test_work[(test_work["_y"] == 1) != (test_work["_p"] >= threshold)].copy()
    errors["_error"] = (errors["_p"] - errors["_y"]).abs()
    error_analysis = (
        errors.sort_values("_error", ascending=False)
        .head(25)[
            [
                "vehicle_id",
                "vehicle_type",
                "make_and_model",
                "brake_condition",
                "usage_hours",
                "engine_temperature",
                "diagnostic_trouble_code_count",
                "_y",
                "_p",
            ]
        ]
        .rename(columns={"_y": "actual", "_p": "probability"})
        .to_dict("records")
    )

    reference_stats = build_reference_stats(
        X_train, MAINTENANCE_NUMERIC, MAINTENANCE_CATEGORICAL, target=y_train
    )
    fingerprint = _fingerprint(X_train, MAINTENANCE_FEATURES)
    local_run_id = f"local-{_utc_stamp()}-{uuid.uuid4().hex[:6]}"
    metadata = _maintenance_metadata(
        local_run_id,
        None,
        threshold,
        {key: value for key, value in metrics.items() if key != "confusion"},
        {key: value for key, value in baseline_metrics.items() if key != "confusion"},
        params,
        {"train": len(train), "validation": len(validation), "test": len(test)},
        float(y_train.mean()),
        fingerprint,
        importance,
        segments,
    )
    metadata["error_analysis"] = _safe_json(error_analysis)

    plot_dir = settings.processed_data_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "importance": plot_dir / "maintenance_importance.png",
        "confusion": plot_dir / "maintenance_confusion.png",
        "calibration": plot_dir / "maintenance_calibration.png",
    }
    _plot_importance(importance, artifacts["importance"], "Maintenance model drivers")
    _plot_confusion(metrics, artifacts["confusion"])
    _plot_calibration(y_test.to_numpy(), test_probability, artifacts["calibration"])

    should_log = settings.mlflow_enabled if log_to_mlflow is None else log_to_mlflow
    version: str | None = None
    if should_log:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_maintenance)
            with mlflow.start_run(run_name=f"maintenance-{_utc_stamp()}") as run:
                run_id = run.info.run_id
                metadata = _maintenance_metadata(
                    run_id,
                    None,
                    threshold,
                    {key: value for key, value in metrics.items() if key != "confusion"},
                    {key: value for key, value in baseline_metrics.items() if key != "confusion"},
                    params,
                    {"train": len(train), "validation": len(validation), "test": len(test)},
                    float(y_train.mean()),
                    fingerprint,
                    importance,
                    segments,
                )
                metadata["error_analysis"] = _safe_json(error_analysis)
                signature = infer_signature(
                    X_train.head(50), pipeline.predict_proba(X_train.head(50))[:, 1]
                )
                version = _log_common(
                    settings.mlflow_experiment_maintenance,
                    settings.mlflow_registered_model_maintenance,
                    pipeline,
                    signature,
                    X_train.head(3),
                    params,
                    {
                        **{
                            f"test_{key}": value
                            for key, value in metrics.items()
                            if key != "confusion"
                        },
                        **{
                            f"val_{key}": value
                            for key, value in val_metrics.items()
                            if key != "confusion"
                        },
                        **{
                            f"baseline_test_{key}": value
                            for key, value in baseline_metrics.items()
                            if key != "confusion"
                        },
                        "threshold": threshold,
                    },
                    artifacts,
                    metadata,
                    reference_stats,
                )
                metadata["mlflow"]["version"] = version
        except Exception:
            logger.exception("MLflow logging failed; writing a local bundle only")
            metadata = _maintenance_metadata(
                local_run_id,
                None,
                threshold,
                {key: value for key, value in metrics.items() if key != "confusion"},
                {key: value for key, value in baseline_metrics.items() if key != "confusion"},
                params,
                {"train": len(train), "validation": len(validation), "test": len(test)},
                float(y_train.mean()),
                fingerprint,
                importance,
                segments,
            )
            metadata["error_analysis"] = _safe_json(error_analysis)

    run_dir = write_bundle(
        "maintenance",
        metadata["run_id"],
        pipeline,
        metadata,
        reference_stats,
        promote=promote,
    )
    return {
        "model": "maintenance",
        "run_id": metadata["run_id"],
        "version": metadata["mlflow"]["version"],
        "threshold": threshold,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "bundle": str(run_dir),
    }


def train_freight(
    sample: int | None = None,
    promote: bool | None = None,
    log_to_mlflow: bool | None = None,
) -> dict[str, Any]:
    engine = get_engine()
    init_db(engine)
    frame = fetch_shipment_frame(engine)
    if frame.empty:
        raise RuntimeError("shipments is empty. Run `make ingest` first.")
    if sample:
        frame = frame.sample(n=min(sample, len(frame)), random_state=RANDOM_STATE)

    train, rest = train_test_split(frame, test_size=0.30, random_state=RANDOM_STATE)
    validation, test = train_test_split(rest, test_size=0.50, random_state=RANDOM_STATE)

    X_train = prepare_freight_features(train)
    X_val = prepare_freight_features(validation)
    X_test = prepare_freight_features(test)
    y_train = train["cost"].astype(float).reset_index(drop=True)
    y_val = validation["cost"].astype(float).reset_index(drop=True)
    y_test = test["cost"].astype(float).reset_index(drop=True)

    params = {
        "loss": "absolute_error",
        "max_iter": 500,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "feature_count": len(FREIGHT_FEATURES),
    }
    pipeline = _freight_pipeline(
        {
            "max_iter": params["max_iter"],
            "learning_rate": params["learning_rate"],
            "max_leaf_nodes": params["max_leaf_nodes"],
            "l2_regularization": params["l2_regularization"],
        }
    )
    pipeline.fit(X_train, y_train)
    val_prediction = np.clip(pipeline.predict(X_val), 0, None)
    test_prediction = np.clip(pipeline.predict(X_test), 0, None)
    residuals = np.abs(y_val.to_numpy() - val_prediction)
    residual_quantiles = {
        str(quantile): float(np.quantile(residuals, quantile)) for quantile in (0.80, 0.90, 0.95)
    }
    metrics = _regression_metrics(y_test.to_numpy(), test_prediction, residual_quantiles["0.9"])
    baseline_rate = float((train["cost"] / train["distance_miles"].replace(0, np.nan)).median())
    baseline_predictions = baseline_rate * test["distance_miles"].to_numpy()
    metrics.update(
        _regression_metrics(
            y_test.to_numpy(), test_prediction, residual_quantiles["0.9"], baseline_predictions
        )
    )

    importance = _importance(pipeline, X_test, y_test, scoring="neg_mean_absolute_error")
    segments = _freight_segments(test, y_test.to_numpy(), test_prediction)
    test_work = test.reset_index(drop=True).copy()
    test_work["_actual"] = y_test.to_numpy()
    test_work["_predicted"] = test_prediction
    test_work["_abs_error"] = (test_work["_actual"] - test_work["_predicted"]).abs()
    error_analysis = (
        test_work.sort_values("_abs_error", ascending=False)
        .head(25)[
            [
                "shipment_id",
                "origin_warehouse",
                "destination",
                "carrier",
                "distance_miles",
                "weight_kg",
                "_actual",
                "_predicted",
                "_abs_error",
            ]
        ]
        .rename(
            columns={
                "_actual": "actual_cost",
                "_predicted": "predicted_cost",
                "_abs_error": "absolute_error",
            }
        )
        .to_dict("records")
    )

    reference_stats = build_reference_stats(
        X_train, FREIGHT_NUMERIC, FREIGHT_CATEGORICAL, target=y_train
    )
    fingerprint = _fingerprint(X_train, FREIGHT_FEATURES)
    local_run_id = f"local-{_utc_stamp()}-{uuid.uuid4().hex[:6]}"
    metadata = _freight_metadata(
        local_run_id,
        None,
        metrics,
        params,
        {"train": len(train), "validation": len(validation), "test": len(test)},
        fingerprint,
        importance,
        segments,
        residual_quantiles,
    )
    metadata["error_analysis"] = _safe_json(error_analysis)

    plot_dir = settings.processed_data_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "importance": plot_dir / "freight_importance.png",
        "residuals": plot_dir / "freight_residuals.png",
        "prediction_actual": plot_dir / "freight_prediction_actual.png",
    }
    _plot_importance(importance, artifacts["importance"], "Freight model drivers")
    _plot_residuals(y_test.to_numpy(), test_prediction, artifacts["residuals"])
    _plot_prediction_actual(y_test.to_numpy(), test_prediction, artifacts["prediction_actual"])

    should_log = settings.mlflow_enabled if log_to_mlflow is None else log_to_mlflow
    version: str | None = None
    if should_log:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_freight)
            with mlflow.start_run(run_name=f"freight-{_utc_stamp()}") as run:
                run_id = run.info.run_id
                metadata = _freight_metadata(
                    run_id,
                    None,
                    metrics,
                    params,
                    {"train": len(train), "validation": len(validation), "test": len(test)},
                    fingerprint,
                    importance,
                    segments,
                    residual_quantiles,
                )
                metadata["error_analysis"] = _safe_json(error_analysis)
                signature = infer_signature(X_train.head(50), pipeline.predict(X_train.head(50)))
                version = _log_common(
                    settings.mlflow_experiment_freight,
                    settings.mlflow_registered_model_freight,
                    pipeline,
                    signature,
                    X_train.head(3),
                    params,
                    {f"test_{key}": value for key, value in metrics.items()},
                    artifacts,
                    metadata,
                    reference_stats,
                )
                metadata["mlflow"]["version"] = version
        except Exception:
            logger.exception("MLflow logging failed; writing a local bundle only")
            metadata = _freight_metadata(
                local_run_id,
                None,
                metrics,
                params,
                {"train": len(train), "validation": len(validation), "test": len(test)},
                fingerprint,
                importance,
                segments,
                residual_quantiles,
            )
            metadata["error_analysis"] = _safe_json(error_analysis)

    run_dir = write_bundle(
        "freight",
        metadata["run_id"],
        pipeline,
        metadata,
        reference_stats,
        promote=promote,
    )
    return {
        "model": "freight",
        "run_id": metadata["run_id"],
        "version": metadata["mlflow"]["version"],
        "metrics": metrics,
        "residual_quantiles": residual_quantiles,
        "bundle": str(run_dir),
    }


def evaluate_gate(results: dict[str, dict[str, Any]]) -> None:
    maintenance = results.get("maintenance", {}).get("metrics", {})
    freight = results.get("freight", {}).get("metrics", {})
    failures: list[str] = []
    if maintenance.get("roc_auc", 0) < settings.min_maintenance_auc:
        failures.append(
            f"maintenance ROC AUC {maintenance.get('roc_auc'):.3f} < {settings.min_maintenance_auc}"
        )
    if freight.get("mape", 100) > settings.max_freight_mape:
        failures.append(f"freight MAPE {freight.get('mape'):.1f}% > {settings.max_freight_mape}%")
    if failures:
        raise RuntimeError("Model quality gate failed: " + "; ".join(failures))


def train_all(
    sample: int | None = None,
    promote: bool | None = None,
    log_to_mlflow: bool | None = None,
) -> dict[str, dict[str, Any]]:
    results = {
        "maintenance": train_maintenance(
            sample=sample, promote=promote, log_to_mlflow=log_to_mlflow
        ),
        "freight": train_freight(sample=sample, promote=promote, log_to_mlflow=log_to_mlflow),
    }
    evaluate_gate(results)
    return results

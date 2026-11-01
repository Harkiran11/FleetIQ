from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

MODEL_KEYS = ("maintenance", "freight")


class ModelNotFoundError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model_key: str
    pipeline: Any
    metadata: dict[str, Any]
    reference_stats: dict[str, Any]
    source: str = "local"

    @property
    def version(self) -> str:
        metadata = self.metadata.get("mlflow", {})
        return str(metadata.get("version") or self.metadata.get("run_id", "local"))

    @property
    def run_id(self) -> str:
        return str(self.metadata.get("run_id", "local"))

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def residual_quantiles(self) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in self.metadata.get("residual_quantiles", {}).items()
        }

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata.get("feature_columns", []))

    @property
    def numeric_features(self) -> list[str]:
        return list(self.metadata.get("numeric_features", []))

    @property
    def categorical_features(self) -> list[str]:
        return list(self.metadata.get("categorical_features", []))


def model_root(model_key: str) -> Path:
    return settings.model_path / model_key


def current_dir(model_key: str) -> Path:
    return model_root(model_key) / "current"


def runs_dir(model_key: str) -> Path:
    return model_root(model_key) / "runs"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, sort_keys=False),
        encoding="utf-8",
    )


def build_reference_stats(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target: pd.Series | None = None,
) -> dict[str, Any]:
    numeric: dict[str, Any] = {}
    for column in numeric_features:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        quantiles = np.nanquantile(values, np.linspace(0, 1, 11))
        edges = np.unique(quantiles.astype(float))
        if len(edges) < 3:
            edges = np.array([float(values.min()), float(values.max())])
        counts, _ = np.histogram(values, bins=edges)
        total = int(counts.sum())
        bin_share = [float(count / total) for count in counts] if total else []
        numeric[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "p25": float(values.quantile(0.25)),
            "p50": float(values.quantile(0.50)),
            "p75": float(values.quantile(0.75)),
            "bin_edges": [float(edge) for edge in edges],
            "bin_share": bin_share,
        }

    categorical: dict[str, Any] = {}
    for column in categorical_features:
        values = frame[column].astype("string").fillna("unknown")
        counts = values.value_counts(normalize=True, dropna=False)
        categorical[column] = {
            "frequency": {str(key): float(value) for key, value in counts.head(50).items()},
            "top": [str(key) for key in counts.head(8).index],
        }

    payload: dict[str, Any] = {
        "row_count": len(frame),
        "numeric": numeric,
        "categorical": categorical,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if target is not None:
        payload["target_rate"] = float(pd.to_numeric(target, errors="coerce").mean())
    return payload


def render_model_card(metadata: dict[str, Any]) -> str:
    metrics = metadata.get("metrics", {})
    metric_lines = "\n".join(
        f"| {name} | {value:.4f} |"
        for name, value in sorted(metrics.items())
        if isinstance(value, (int, float))
    )
    excluded = ", ".join(metadata.get("excluded_columns", []))
    return f"""# {metadata.get("title", metadata.get("model_name", "Model"))}

**Version:** {metadata.get("mlflow", {}).get("version", "local")}
**Run:** {metadata.get("run_id", "n/a")}
**Trained:** {metadata.get("trained_at", "n/a")}
**Training rows:** {metadata.get("training_rows", "n/a")}
**Reference year:** {metadata.get("reference_year", "n/a")}

## Intended use

{metadata.get("intended_use", "Batch and online scoring for fleet operations.")}

## Metrics

| Metric | Value |
| --- | --- |
{metric_lines}

## Inputs

{", ".join(metadata.get("feature_columns", []))}

## Excluded columns

{excluded or "None"}

## Limitations

{metadata.get("limitations", "See the project documentation.")}
"""


def write_bundle(
    model_key: str,
    run_id: str,
    pipeline: Any,
    metadata: dict[str, Any],
    reference_stats: dict[str, Any],
    promote: bool | None = None,
) -> Path:
    run_dir = runs_dir(model_key) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, run_dir / "model.joblib", compress=3)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "reference_stats.json", reference_stats)
    (run_dir / "model_card.md").write_text(render_model_card(metadata), encoding="utf-8")

    should_promote = settings.auto_promote if promote is None else promote
    if should_promote:
        promote_run(model_key, run_id)
    return run_dir


def promote_run(model_key: str, run_id: str) -> Path:
    source = runs_dir(model_key) / run_id
    if not source.exists():
        raise FileNotFoundError(f"Run bundle not found: {source}")
    target = current_dir(model_key)
    staging = model_root(model_key) / f".current-{uuid.uuid4().hex[:8]}"
    shutil.copytree(source, staging)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    _write_json(
        model_root(model_key) / "current.json",
        {
            "run_id": run_id,
            "promoted_at": datetime.now(UTC).isoformat(),
            "path": str(target),
        },
    )
    logger.info("Promoted %s run %s to current", model_key, run_id)
    return target


def load_bundle(model_key: str) -> LoadedModel:
    directory = current_dir(model_key)
    model_path = directory / "model.joblib"
    metadata_path = directory / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise ModelNotFoundError(
            f"No promoted {model_key} model in {directory}. Run `make train` first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference_path = directory / "reference_stats.json"
    reference_stats = (
        json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else {}
    )
    return LoadedModel(
        model_key=model_key,
        pipeline=joblib.load(model_path),
        metadata=metadata,
        reference_stats=reference_stats,
        source="local",
    )


def _registered_name(model_key: str) -> str:
    if model_key == "maintenance":
        return settings.mlflow_registered_model_maintenance
    if model_key == "freight":
        return settings.mlflow_registered_model_freight
    raise ValueError(f"Unknown model key: {model_key}")


def load_from_mlflow(model_key: str, alias: str = "champion") -> LoadedModel:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    name = _registered_name(model_key)
    version = client.get_model_version_by_alias(name, alias)
    run_id = version.run_id
    metadata_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/metadata.json")
    reference_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/reference_stats.json")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    reference_stats = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    pipeline = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    return LoadedModel(
        model_key=model_key,
        pipeline=pipeline,
        metadata=metadata,
        reference_stats=reference_stats,
        source="mlflow",
    )


class ModelRegistry:
    """Holds the models the API serves and reloads them as a unit."""

    def __init__(self, source: str | None = None):
        self.source = source or settings.model_source
        self._models: dict[str, LoadedModel] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self.loaded_at: datetime | None = None

    def load_all(self) -> None:
        models: dict[str, LoadedModel] = {}
        errors: dict[str, str] = {}
        for key in MODEL_KEYS:
            try:
                models[key] = load_from_mlflow(key) if self.source == "mlflow" else load_bundle(key)
            except Exception as exc:
                logger.warning("Could not load %s model: %s", key, exc)
                errors[key] = str(exc)
        with self._lock:
            self._models = models
            self._errors = errors
            self.loaded_at = datetime.now(UTC)

    def reload(self, source: str | None = None) -> None:
        if source:
            self.source = source
        self.load_all()

    def get(self, model_key: str) -> LoadedModel:
        with self._lock:
            model = self._models.get(model_key)
            error = self._errors.get(model_key)
        if model is None:
            raise ModelNotFoundError(error or f"Model {model_key} is not loaded")
        return model

    def available(self) -> dict[str, bool]:
        with self._lock:
            return {key: key in self._models for key in MODEL_KEYS}

    def errors(self) -> dict[str, str]:
        with self._lock:
            return dict(self._errors)

    def info(self) -> list[dict[str, Any]]:
        with self._lock:
            models = dict(self._models)
            errors = dict(self._errors)
        payload: list[dict[str, Any]] = []
        for key in MODEL_KEYS:
            model = models.get(key)
            if model is None:
                payload.append(
                    {
                        "key": key,
                        "available": False,
                        "error": errors.get(key),
                        "source": self.source,
                    }
                )
                continue
            metadata = model.metadata
            payload.append(
                {
                    "key": key,
                    "available": True,
                    "source": model.source,
                    "name": metadata.get("model_name", key),
                    "version": model.version,
                    "run_id": model.run_id,
                    "trained_at": metadata.get("trained_at"),
                    "metrics": metadata.get("metrics", {}),
                    "params": metadata.get("params", {}),
                    "threshold": metadata.get("threshold"),
                    "global_importance": metadata.get("global_importance", []),
                    "segments": metadata.get("segments", {}),
                    "limitations": metadata.get("limitations"),
                    "mlflow": metadata.get("mlflow", {}),
                }
            )
        return payload

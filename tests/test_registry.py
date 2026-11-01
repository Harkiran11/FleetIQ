from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.registry import (
    LoadedModel,
    ModelNotFoundError,
    ModelRegistry,
    build_reference_stats,
    load_bundle,
    promote_run,
    render_model_card,
    write_bundle,
)
from src.storage import local_inventory


def _dummy_pipeline() -> DummyClassifier:
    pipeline = DummyClassifier(strategy="prior")
    pipeline.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    return pipeline


def _metadata(run_id: str, version: str = "1") -> dict:
    return {
        "model_key": "maintenance",
        "model_name": "fleetiq-maintenance-risk",
        "title": "FleetIQ Maintenance Risk",
        "run_id": run_id,
        "trained_at": "2026-01-01T00:00:00+00:00",
        "training_rows": 100,
        "reference_year": 2025,
        "feature_columns": ["usage_hours", "brake_condition"],
        "excluded_columns": ["maintenance_cost", "predictive_score"],
        "threshold": 0.62,
        "metrics": {"roc_auc": 0.91, "f1": 0.82},
        "mlflow": {"version": version, "registered_model": "fleetiq-maintenance-risk"},
        "intended_use": "Prioritise maintenance inspections.",
        "limitations": "Synthetic dataset; metrics are optimistic.",
    }


def _reference_stats() -> dict:
    return {
        "row_count": 100,
        "numeric": {"usage_hours": {"mean": 1000.0, "std": 100.0}},
        "categorical": {"brake_condition": {"frequency": {"Good": 0.8, "Poor": 0.2}}},
    }


def test_build_reference_stats_creates_bins_and_frequencies():
    frame = pd.DataFrame(
        {
            "usage_hours": np.arange(100, dtype=float),
            "brake_condition": ["Good"] * 80 + ["Poor"] * 20,
        }
    )

    stats = build_reference_stats(
        frame, ["usage_hours"], ["brake_condition"], target=pd.Series([1] * 40 + [0] * 60)
    )

    numeric = stats["numeric"]["usage_hours"]
    assert numeric["mean"] == pytest.approx(49.5)
    assert len(numeric["bin_edges"]) >= 2
    assert sum(numeric["bin_share"]) == pytest.approx(1.0)
    assert stats["categorical"]["brake_condition"]["frequency"]["Poor"] == pytest.approx(0.2)
    assert stats["target_rate"] == pytest.approx(0.4)
    assert stats["row_count"] == 100


def test_build_reference_stats_handles_constant_numeric_column():
    frame = pd.DataFrame({"usage_hours": [7.0] * 10, "brake_condition": ["Good"] * 10})

    stats = build_reference_stats(frame, ["usage_hours"], ["brake_condition"])

    numeric = stats["numeric"]["usage_hours"]
    assert numeric["std"] == 0.0
    assert numeric["bin_edges"] == [7.0, 7.0]


def test_write_promote_and_load_bundle_round_trip(tmp_path, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "model_dir", tmp_path)
    run_id = "run-001"

    run_dir = write_bundle(
        "maintenance",
        run_id,
        _dummy_pipeline(),
        _metadata(run_id),
        _reference_stats(),
        promote=False,
    )

    assert run_dir == tmp_path / "maintenance" / "runs" / run_id
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "reference_stats.json").exists()

    promote_run("maintenance", run_id)

    loaded = load_bundle("maintenance")
    assert isinstance(loaded, LoadedModel)
    assert loaded.model_key == "maintenance"
    assert loaded.run_id == run_id
    assert loaded.version == "1"
    assert loaded.threshold == pytest.approx(0.62)
    assert loaded.feature_columns == ["usage_hours", "brake_condition"]
    assert loaded.source == "local"
    assert loaded.metadata["title"] == "FleetIQ Maintenance Risk"
    assert loaded.reference_stats["row_count"] == 100

    pointer = json.loads((tmp_path / "maintenance" / "current.json").read_text())
    assert pointer["run_id"] == run_id


def test_promote_run_rejects_missing_bundle(tmp_path, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "model_dir", tmp_path)

    with pytest.raises(FileNotFoundError):
        promote_run("maintenance", "does-not-exist")


def test_write_bundle_auto_promotes_when_requested(tmp_path, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "model_dir", tmp_path)

    write_bundle(
        "freight",
        "run-002",
        _dummy_pipeline(),
        _metadata("run-002", version="2"),
        _reference_stats(),
        promote=True,
    )

    loaded = load_bundle("freight")
    assert loaded.version == "2"
    assert (tmp_path / "freight" / "current" / "model_card.md").exists()


def test_render_model_card_contains_governance_sections():
    card = render_model_card(_metadata("run-003", version="3"))

    assert "# FleetIQ Maintenance Risk" in card
    assert "**Version:** 3" in card
    assert "run-003" in card
    assert "| roc_auc | 0.9100 |" in card
    assert "maintenance_cost" in card
    assert "Synthetic dataset" in card


def test_loaded_model_properties_have_safe_defaults():
    model = LoadedModel(
        model_key="freight",
        pipeline=object(),
        metadata={"run_id": "local-1"},
        reference_stats={},
    )

    assert model.version == "local-1"
    assert model.threshold == 0.5
    assert model.residual_quantiles == {}
    assert model.feature_columns == []
    assert model.numeric_features == []
    assert model.categorical_features == []


def test_registry_reports_missing_models_without_raising(tmp_path, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "model_dir", tmp_path)
    registry = ModelRegistry(source="local")

    registry.load_all()

    assert registry.available() == {"maintenance": False, "freight": False}
    assert set(registry.errors()) == {"maintenance", "freight"}
    with pytest.raises(ModelNotFoundError):
        registry.get("maintenance")
    info = registry.info()
    assert [item["key"] for item in info] == ["maintenance", "freight"]
    assert all(item["available"] is False for item in info)
    assert all(item["error"] for item in info)


def test_registry_loads_the_fixture_bundles():
    registry = ModelRegistry(source="local")
    registry.load_all()

    assert registry.available() == {"maintenance": True, "freight": True}
    assert registry.errors() == {}
    assert registry.get("maintenance").model_key == "maintenance"
    assert registry.get("freight").model_key == "freight"
    assert registry.loaded_at is not None


def test_local_inventory_describes_fixture_bundles():
    inventory = local_inventory()

    assert {item["model"] for item in inventory} == {"maintenance", "freight"}
    assert all(item["exists"] for item in inventory)
    assert all(item["size_bytes"] > 0 for item in inventory)
    assert all(item["run_id"] for item in inventory)

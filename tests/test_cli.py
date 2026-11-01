from __future__ import annotations

import json

import pytest

from src.cli import _read_payload, build_parser, main


def test_parser_exposes_every_operational_command():
    parser = build_parser()

    assert parser.parse_args(["ingest", "--force"]).force is True
    assert parser.parse_args(["train", "--model", "freight", "--no-mlflow"]).no_mlflow is True
    assert parser.parse_args(["predict-maintenance", "--demo"]).command == "predict-maintenance"
    assert parser.parse_args(["predict-freight", "--demo"]).command == "predict-freight"
    assert parser.parse_args(["monitor", "--window", "250"]).window == 250
    assert parser.parse_args(["s3-sync", "--direction", "list"]).direction == "list"
    assert parser.parse_args(["pipeline", "--flow", "monitor"]).flow == "monitor"


def test_parser_requires_a_command():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_read_payload_prefers_inline_json(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"source": "file"}), encoding="utf-8")

    assert _read_payload(str(path), None) == {"source": "file"}
    assert _read_payload(str(path), '{"source": "inline"}') == {"source": "inline"}
    assert _read_payload(None, None) == {}


def test_info_command_prints_database_and_model_status(capsys):
    code = main(["info"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["database"]["maintenance_rows"] == 5000
    assert payload["database"]["shipment_rows"] == 250
    assert {item["key"] for item in payload["models"]} == {"maintenance", "freight"}
    assert all(item["available"] for item in payload["models"])
    assert all(item["exists"] for item in payload["local_inventory"])


def test_predict_maintenance_demo_command(capsys):
    code = main(["predict-maintenance", "--demo"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["risk"]["level"] in {"low", "moderate", "high", "critical"}
    assert payload["service_note"]["error_codes"] == ["ERR-431"]


def test_predict_freight_demo_command(capsys):
    code = main(["predict-freight", "--demo"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["estimated_cost_usd"] >= 0
    assert payload["interval_usd"]["low"] <= payload["estimated_cost_usd"]


def test_evaluate_command_applies_quality_gate(capsys):
    code = main(["evaluate"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert set(payload["models"]) == {"maintenance", "freight"}


def test_predict_maintenance_accepts_an_inline_payload(capsys):
    code = main(
        [
            "predict-maintenance",
            "--payload",
            json.dumps(
                {
                    "year_of_manufacture": 2020,
                    "usage_hours": 5000,
                    "load_capacity": 12000,
                    "actual_load": 6000,
                    "engine_temperature": 92,
                    "brake_condition": "Good",
                }
            ),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["risk"]["probability"] >= 0

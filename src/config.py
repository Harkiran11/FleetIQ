from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Values come from environment variables or a local .env file. Names are
    intentionally unprefixed so the same variables work for the CLI, the API
    container, and the MLflow client.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "FleetIQ"
    app_version: str = "1.0.0"
    app_env: str = "local"
    log_level: str = "INFO"
    log_json: bool = False

    database_url: str = "sqlite:///./data/fleetiq.db"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")

    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment_maintenance: str = "fleetiq-maintenance-risk"
    mlflow_experiment_freight: str = "fleetiq-freight-pricing"
    mlflow_registered_model_maintenance: str = "fleetiq-maintenance-risk"
    mlflow_registered_model_freight: str = "fleetiq-freight-pricing"
    mlflow_enabled: bool = True

    model_source: Literal["local", "mlflow"] = "local"
    auto_promote: bool = True
    reference_year: int = 2025

    min_maintenance_auc: float = 0.75
    max_freight_mape: float = 25.0

    drift_window: int = 500
    psi_watch: float = 0.10
    psi_alert: float = 0.25

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:8000,http://localhost:5173"
    admin_token: str | None = None
    prediction_log_enabled: bool = True
    max_batch_size: int = 500

    s3_bucket: str | None = None
    s3_prefix: str = "fleetiq"
    aws_region: str = "ca-central-1"

    maintenance_csv: Path | None = None
    shipments_csv: Path | None = None

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def path(self, value: Path) -> Path:
        return value if value.is_absolute() else self.project_root / value

    @property
    def data_path(self) -> Path:
        return self.path(self.data_dir)

    @property
    def model_path(self) -> Path:
        return self.path(self.model_dir)

    @property
    def raw_data_path(self) -> Path:
        return self.data_path / "raw"

    @property
    def processed_data_path(self) -> Path:
        return self.data_path / "processed"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def connect_args(self) -> dict:
        return {"check_same_thread": False} if self.is_sqlite else {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

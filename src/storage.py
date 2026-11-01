"""Optional S3 artefact synchronisation.

The application is fully functional without AWS: local bundles under
``models/<key>/current`` are the default serving source. When an S3 bucket is
configured this module mirrors the bundle directory so a container or a second
host can pull the exact model that was promoted in training.

boto3 is imported lazily so the core install stays small and the API can boot
on a machine that has no AWS credentials.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import settings
from src.registry import MODEL_KEYS, current_dir

logger = logging.getLogger(__name__)


class StorageUnavailable(RuntimeError):
    pass


def _require_bucket(bucket: str | None) -> str:
    target = bucket or settings.s3_bucket
    if not target:
        raise StorageUnavailable("No S3 bucket configured. Set S3_BUCKET or pass --bucket.")
    return target


def _client():
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise StorageUnavailable(
            "boto3 is not installed. Run `pip install boto3` to enable S3 sync."
        ) from exc
    return boto3.client("s3", region_name=settings.aws_region)


def _iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def upload_directory(
    local_dir: Path,
    bucket: str | None = None,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Upload every file under ``local_dir`` to ``s3://bucket/prefix``."""
    bucket_name = _require_bucket(bucket)
    base_prefix = (prefix or settings.s3_prefix).strip("/")
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    client = _client()
    uploaded: list[str] = []
    for path in _iter_files(local_dir):
        relative = path.relative_to(local_dir).as_posix()
        key = (
            f"{base_prefix}/{local_dir.name}/{relative}"
            if base_prefix
            else f"{local_dir.name}/{relative}"
        )
        client.upload_file(str(path), bucket_name, key)
        uploaded.append(key)

    manifest = {
        "uploaded_at": datetime.now(UTC).isoformat(),
        "bucket": bucket_name,
        "prefix": base_prefix,
        "files": uploaded,
    }
    manifest_key = f"{base_prefix}/{local_dir.name}/_sync_manifest.json"
    client.put_object(
        Bucket=bucket_name,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Uploaded %s files to s3://%s/%s", len(uploaded), bucket_name, base_prefix)
    return manifest


def download_directory(
    local_dir: Path,
    bucket: str | None = None,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Download ``s3://bucket/prefix/<local_dir.name>`` into ``local_dir``."""
    bucket_name = _require_bucket(bucket)
    base_prefix = (prefix or settings.s3_prefix).strip("/")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    client = _client()
    root_key = f"{base_prefix}/{local_dir.name}/" if base_prefix else f"{local_dir.name}/"
    paginator = client.get_paginator("list_objects_v2")
    downloaded: list[str] = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=root_key):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("_sync_manifest.json"):
                continue
            relative = key[len(root_key) :]
            if not relative:
                continue
            target = local_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket_name, key, str(target))
            downloaded.append(relative)

    logger.info("Downloaded %s files from s3://%s/%s", len(downloaded), bucket_name, root_key)
    return {
        "downloaded_at": datetime.now(UTC).isoformat(),
        "bucket": bucket_name,
        "prefix": root_key,
        "files": downloaded,
    }


def sync_models(direction: str = "upload", keys: tuple[str, ...] = MODEL_KEYS) -> dict[str, Any]:
    """Upload or download the promoted bundle for each model key."""
    results: dict[str, Any] = {}
    for key in keys:
        directory = current_dir(key)
        if direction == "upload":
            results[key] = upload_directory(directory, prefix=f"{settings.s3_prefix}/{key}")
        elif direction == "download":
            results[key] = download_directory(directory, prefix=f"{settings.s3_prefix}/{key}")
        else:
            raise ValueError("direction must be 'upload' or 'download'")
    return results


def list_remote(prefix: str | None = None) -> list[dict[str, Any]]:
    bucket_name = _require_bucket(None)
    base_prefix = (prefix or settings.s3_prefix).strip("/")
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    items: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=base_prefix):
        for item in page.get("Contents", []):
            items.append(
                {
                    "key": item["Key"],
                    "size": int(item["Size"]),
                    "last_modified": item["LastModified"].isoformat(),
                }
            )
    return items


def local_inventory() -> list[dict[str, Any]]:
    """Describe the local bundles without touching S3."""
    inventory: list[dict[str, Any]] = []
    for key in MODEL_KEYS:
        directory = current_dir(key)
        metadata_path = directory / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        inventory.append(
            {
                "model": key,
                "path": str(directory),
                "exists": (directory / "model.joblib").exists(),
                "size_bytes": sum(path.stat().st_size for path in _iter_files(directory))
                if directory.exists()
                else 0,
                "run_id": metadata.get("run_id"),
                "version": metadata.get("mlflow", {}).get("version"),
                "trained_at": metadata.get("trained_at"),
            }
        )
    return inventory

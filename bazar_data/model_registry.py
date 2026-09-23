"""Persistent active-model registry; artifacts remain filesystem/object-store paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from bazar_data.database import Database
from bazar_data.marketplace_schema import model_versions


def register_model(
    database: Database,
    *,
    version: str,
    model_type: str,
    artifact_uri: str | Path,
    metrics: dict[str, Any],
    data_hash: str,
    activate: bool = False,
) -> None:
    database.create_schema()
    if activate:
        database.execute("UPDATE model_versions SET is_active = false")
    database.upsert(model_versions, [{
        "model_version": version, "model_type": model_type, "artifact_uri": str(artifact_uri),
        "metrics": metrics, "data_hash": data_hash, "is_active": activate,
    }], key_columns=("model_version",))


def activate_model(database: Database, version: str) -> None:
    database.create_schema()
    row = database.fetch_one("SELECT model_version FROM model_versions WHERE model_version = :version", {"version": version})
    if row is None:
        raise ValueError(f"Unknown model version: {version}")
    database.execute("UPDATE model_versions SET is_active = false")
    database.execute("UPDATE model_versions SET is_active = true WHERE model_version = :version", {"version": version})


def active_model(database: Database) -> dict[str, Any] | None:
    database.create_schema()
    frame = database.read_dataframe("SELECT model_version, model_type, artifact_uri, metrics, data_hash, is_active, created_at FROM model_versions WHERE is_active = true ORDER BY created_at DESC LIMIT 1")
    return frame.iloc[0].to_dict() if not frame.empty else None


def list_models(database: Database) -> list[dict[str, Any]]:
    database.create_schema()
    return database.read_dataframe("SELECT model_version, model_type, artifact_uri, metrics, data_hash, is_active, created_at FROM model_versions ORDER BY created_at DESC").to_dict(orient="records")

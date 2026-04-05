from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def artifact_root() -> Path:
    configured = os.getenv("OCEANPULSE_ARTIFACT_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path
    return Path(__file__).resolve().parents[1] / "artifacts"


def manifests_dir() -> Path:
    path = artifact_root() / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def published_dir() -> Path:
    path = artifact_root() / "published"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def data_manifest_path() -> Path:
    return manifests_dir() / "latest_data_manifest.json"


def training_manifest_path() -> Path:
    return manifests_dir() / "latest_training_manifest.json"


def map_bundle_path(region: str, resolution: str) -> Path:
    safe_region = region.replace("/", "_")
    safe_resolution = resolution.replace("/", "_")
    return published_dir() / f"map_bundle__{safe_region}__{safe_resolution}.json"


def write_data_manifest(payload: dict[str, Any]) -> Path:
    document = {
        "kind": "training_data_manifest",
        "created_at": datetime.now(timezone.utc),
        **payload,
    }
    return write_json_atomic(data_manifest_path(), document)


def write_training_manifest(payload: dict[str, Any]) -> Path:
    document = {
        "kind": "training_manifest",
        "created_at": datetime.now(timezone.utc),
        **payload,
    }
    return write_json_atomic(training_manifest_path(), document)


def write_published_map_bundle(*, region: str, resolution: str, payload: dict[str, Any]) -> Path:
    document = {
        "kind": "published_map_bundle",
        "published_at": datetime.now(timezone.utc),
        "region": region,
        "resolution": resolution,
        **payload,
    }
    return write_json_atomic(map_bundle_path(region, resolution), document)


def load_published_map_bundle(*, region: str, resolution: str, date: str | None = None) -> dict[str, Any] | None:
    payload = read_json(map_bundle_path(region, resolution))
    if not payload:
        return None
    metadata = payload.get("metadata", {})
    if date and metadata.get("date") != date:
        return None
    return payload


def load_best_published_map_bundle(*, region: str, resolution: str, date: str | None = None) -> dict[str, Any] | None:
    candidates: list[tuple[str, str]] = []
    for candidate in [(region, resolution), (region, "2deg"), (region, "1deg")]:
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate_region, candidate_resolution in candidates:
        exact = load_published_map_bundle(region=candidate_region, resolution=candidate_resolution, date=date)
        if exact:
            exact.setdefault("metadata", {})
            exact["metadata"]["published_bundle_resolution"] = candidate_resolution
            exact["metadata"]["published_bundle_region"] = candidate_region
            exact["metadata"]["published_bundle_date_match"] = True
            return exact

    for candidate_region, candidate_resolution in candidates:
        payload = read_json(map_bundle_path(candidate_region, candidate_resolution))
        if payload:
            payload.setdefault("metadata", {})
            payload["metadata"]["published_bundle_resolution"] = candidate_resolution
            payload["metadata"]["published_bundle_region"] = candidate_region
            payload["metadata"]["published_bundle_date_match"] = payload["metadata"].get("date") == date if date else True
            return payload
    return None

from __future__ import annotations

import csv
import io
import json
from typing import Any

import requests


class TrashDataError(RuntimeError):
    pass


def _extract_float(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    lat = _extract_float(record, "lat", "latitude", "y")
    lon = _extract_float(record, "lon", "longitude", "lng", "x")
    if lat is None or lon is None:
        return None

    intensity = _extract_float(record, "intensity", "density", "score", "weight", "concentration") or 0.0
    return {
        "label": str(record.get("label") or record.get("name") or record.get("site") or "Trash observation"),
        "lat": lat,
        "lon": lon,
        "intensity": intensity,
        "source": str(record.get("source") or "trash_observation"),
        "observed": True,
        "metadata": {key: value for key, value in record.items() if key not in {"lat", "latitude", "lon", "longitude", "lng", "x", "y"}},
    }


def _normalize_geojson(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features = payload.get("features", [])
    items: list[dict[str, Any]] = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        coords = geometry.get("coordinates") or []
        if geometry.get("type") == "Point" and len(coords) >= 2:
            properties = {**properties, "lon": coords[0], "lat": coords[1]}
        normalized = _normalize_record(properties)
        if normalized:
            items.append(normalized)
    return items


def _normalize_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        return _normalize_geojson(payload)
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in (_normalize_record(record) for record in value if isinstance(record, dict)) if item]
    if isinstance(payload, list):
        return [item for item in (_normalize_record(record) for record in payload if isinstance(record, dict)) if item]
    return []


def _normalize_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    return [item for item in (_normalize_record(row) for row in reader) if item]


def load_trash_observations(data_url: str, *, timeout: int = 25, limit: int = 100) -> list[dict[str, Any]]:
    if not data_url.strip():
        raise TrashDataError("Trash data URL is empty.")

    response = requests.get(data_url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()

    items: list[dict[str, Any]]
    if "json" in content_type or data_url.endswith((".json", ".geojson")):
        items = _normalize_json(response.json())
    elif "csv" in content_type or data_url.endswith(".csv"):
        items = _normalize_csv(response.text)
    else:
        try:
            items = _normalize_json(response.json())
        except json.JSONDecodeError:
            items = _normalize_csv(response.text)

    if not items:
        raise TrashDataError(f"No usable trash observation rows were found at {data_url}.")
    return items[: max(1, min(limit, 500))]

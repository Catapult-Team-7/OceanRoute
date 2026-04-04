from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_data_directories() -> None:
    for subdir in ["raw", "interim", "features", "forecasts", "routes", "datasets", "models"]:
        (settings.data_root / subdir).mkdir(parents=True, exist_ok=True)


def write_debug_json(subdir: str, filename: str, payload: Any) -> None:
    if not settings.write_debug_artifacts:
        return
    ensure_data_directories()
    path = settings.data_root / subdir / filename
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_raw_payload(prefix: str, payload: Any) -> None:
    write_debug_json("raw", f"{prefix}-{_now_stamp()}.json", payload)


def write_latest_snapshot(snapshot: dict[str, Any]) -> None:
    write_debug_json("forecasts", "latest_forecast.json", snapshot)


def write_route_snapshot(mission_id: str, payload: dict[str, Any]) -> None:
    write_debug_json("routes", f"{mission_id}.json", payload)

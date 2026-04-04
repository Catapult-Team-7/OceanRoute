from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default


@dataclass(frozen=True)
class Settings:
    app_name: str
    api_prefix: str
    pilot_region: str
    database_url: str
    data_root: Path
    forecast_horizon_hours: int
    scheduler_enabled: bool
    scheduler_interval_minutes: int
    ingest_mode: str
    live_request_timeout_seconds: float
    forecast_stale_after_minutes: int
    default_depot_lat: float
    default_depot_lon: float
    vessel_speed_kmh: float
    write_debug_artifacts: bool
    cors_allowed_origins: tuple[str, ...]


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[3]
    _load_env_file(project_root / ".env")
    return Settings(
        app_name=os.getenv("OCEANROUTE_APP_NAME", "SeaSweep API"),
        api_prefix=os.getenv("OCEANROUTE_API_PREFIX", "/api"),
        pilot_region=os.getenv("OCEANROUTE_PILOT_REGION", "sf_bay_estuary"),
        database_url=os.getenv(
            "OCEANROUTE_DATABASE_URL",
            "postgresql+psycopg://oceanroute:oceanroute@localhost:5432/oceanroute",
        ),
        data_root=Path(os.getenv("OCEANROUTE_DATA_ROOT", str(project_root / "data"))),
        forecast_horizon_hours=int(os.getenv("OCEANROUTE_FORECAST_HOURS", "24")),
        scheduler_enabled=_env_bool("OCEANROUTE_SCHEDULER_ENABLED", True),
        scheduler_interval_minutes=int(os.getenv("OCEANROUTE_SCHEDULER_INTERVAL_MINUTES", "60")),
        ingest_mode=os.getenv("OCEANROUTE_INGEST_MODE", "auto"),
        live_request_timeout_seconds=float(os.getenv("OCEANROUTE_LIVE_TIMEOUT_SECONDS", "6")),
        forecast_stale_after_minutes=int(os.getenv("OCEANROUTE_FORECAST_STALE_AFTER_MINUTES", "180")),
        default_depot_lat=float(os.getenv("OCEANROUTE_DEFAULT_DEPOT_LAT", "37.8066")),
        default_depot_lon=float(os.getenv("OCEANROUTE_DEFAULT_DEPOT_LON", "-122.4659")),
        vessel_speed_kmh=float(os.getenv("OCEANROUTE_VESSEL_SPEED_KMH", "18.0")),
        write_debug_artifacts=_env_bool("OCEANROUTE_WRITE_DEBUG_ARTIFACTS", True),
        cors_allowed_origins=_env_csv("OCEANROUTE_CORS_ALLOWED_ORIGINS", ("*",)),
    )


settings = load_settings()

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


def _env_raw(*names: str) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw
    return None


def _env_str(*names: str, default: str) -> str:
    raw = _env_raw(*names)
    if raw is None:
        return default
    return raw


def _env_bool(*names: str, default: bool) -> bool:
    raw = _env_raw(*names)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(*names: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env_raw(*names)
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
        app_name=_env_str("SEASWEEP_APP_NAME", "OCEANROUTE_APP_NAME", default="SeaSweep API"),
        api_prefix=_env_str("SEASWEEP_API_PREFIX", "OCEANROUTE_API_PREFIX", default="/api"),
        pilot_region=_env_str("SEASWEEP_PILOT_REGION", "OCEANROUTE_PILOT_REGION", default="sf_bay_estuary"),
        database_url=_env_str(
            "SEASWEEP_DATABASE_URL",
            "OCEANROUTE_DATABASE_URL",
            default="postgresql+psycopg://seasweep:seasweep@localhost:5432/seasweep",
        ),
        data_root=Path(_env_str("SEASWEEP_DATA_ROOT", "OCEANROUTE_DATA_ROOT", default=str(project_root / "data"))),
        forecast_horizon_hours=int(_env_str("SEASWEEP_FORECAST_HOURS", "OCEANROUTE_FORECAST_HOURS", default="24")),
        scheduler_enabled=_env_bool("SEASWEEP_SCHEDULER_ENABLED", "OCEANROUTE_SCHEDULER_ENABLED", default=True),
        scheduler_interval_minutes=int(
            _env_str("SEASWEEP_SCHEDULER_INTERVAL_MINUTES", "OCEANROUTE_SCHEDULER_INTERVAL_MINUTES", default="60")
        ),
        ingest_mode=_env_str("SEASWEEP_INGEST_MODE", "OCEANROUTE_INGEST_MODE", default="auto"),
        live_request_timeout_seconds=float(
            _env_str("SEASWEEP_LIVE_TIMEOUT_SECONDS", "OCEANROUTE_LIVE_TIMEOUT_SECONDS", default="6")
        ),
        forecast_stale_after_minutes=int(
            _env_str(
                "SEASWEEP_FORECAST_STALE_AFTER_MINUTES",
                "OCEANROUTE_FORECAST_STALE_AFTER_MINUTES",
                default="180",
            )
        ),
        default_depot_lat=float(
            _env_str("SEASWEEP_DEFAULT_DEPOT_LAT", "OCEANROUTE_DEFAULT_DEPOT_LAT", default="37.8066")
        ),
        default_depot_lon=float(
            _env_str("SEASWEEP_DEFAULT_DEPOT_LON", "OCEANROUTE_DEFAULT_DEPOT_LON", default="-122.4659")
        ),
        vessel_speed_kmh=float(_env_str("SEASWEEP_VESSEL_SPEED_KMH", "OCEANROUTE_VESSEL_SPEED_KMH", default="18.0")),
        write_debug_artifacts=_env_bool(
            "SEASWEEP_WRITE_DEBUG_ARTIFACTS",
            "OCEANROUTE_WRITE_DEBUG_ARTIFACTS",
            default=True,
        ),
        cors_allowed_origins=_env_csv(
            "SEASWEEP_CORS_ALLOWED_ORIGINS",
            "OCEANROUTE_CORS_ALLOWED_ORIGINS",
            default=("*",),
        ),
    )


settings = load_settings()

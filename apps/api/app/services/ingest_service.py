from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import atan2, cos, radians, sin, sqrt
from typing import Any

import httpx

from app.config import settings
from app.schemas import GridPoint, OperationalContext, OperationalGridFrame, SourceMode
from app.services.artifact_service import write_raw_payload


SF_BAY_BOUNDS = {
    "lat_min": 37.45,
    "lat_max": 38.25,
    "lon_min": -123.05,
    "lon_max": -121.75,
}

SF_BAY_CELLS = [
    {"cell_id": "golden_gate", "lat": 37.805, "lon": -122.475, "shoreline_proximity": 0.55, "restricted": False},
    {"cell_id": "north_bay", "lat": 38.025, "lon": -122.395, "shoreline_proximity": 0.36, "restricted": False},
    {"cell_id": "angel_island", "lat": 37.86, "lon": -122.43, "shoreline_proximity": 0.52, "restricted": False},
    {"cell_id": "central_bay", "lat": 37.81, "lon": -122.375, "shoreline_proximity": 0.21, "restricted": False},
    {"cell_id": "oakland_approach", "lat": 37.79, "lon": -122.29, "shoreline_proximity": 0.33, "restricted": False},
    {"cell_id": "richmond_shoal", "lat": 37.925, "lon": -122.39, "shoreline_proximity": 0.48, "restricted": False},
    {"cell_id": "alameda_corridor", "lat": 37.765, "lon": -122.265, "shoreline_proximity": 0.41, "restricted": False},
    {"cell_id": "san_leandro", "lat": 37.71, "lon": -122.19, "shoreline_proximity": 0.58, "restricted": False},
    {"cell_id": "south_bay", "lat": 37.585, "lon": -122.16, "shoreline_proximity": 0.64, "restricted": False},
    {"cell_id": "mission_creek", "lat": 37.772, "lon": -122.387, "shoreline_proximity": 0.82, "restricted": True},
    {"cell_id": "port_of_oakland", "lat": 37.787, "lon": -122.312, "shoreline_proximity": 0.72, "restricted": True},
    {"cell_id": "berkeley_marina", "lat": 37.866, "lon": -122.313, "shoreline_proximity": 0.67, "restricted": False},
]

SAMPLE_CURRENT_STATIONS = [
    {"station_id": "gg01", "name": "Golden Gate current prediction", "lat": 37.8066, "lon": -122.4659, "speed": 1.28, "direction": 118.0, "phase": 0.2},
    {"station_id": "oak01", "name": "Oakland estuary current prediction", "lat": 37.7955, "lon": -122.2782, "speed": 0.84, "direction": 142.0, "phase": 1.1},
    {"station_id": "sb01", "name": "South Bay current prediction", "lat": 37.587, "lon": -122.06, "speed": 0.62, "direction": 171.0, "phase": 2.3},
]

SAMPLE_WIND_STATIONS = [
    {"station_id": "9414290", "name": "San Francisco", "lat": 37.8063, "lon": -122.4659, "speed": 6.2, "direction": 284.0, "water_temperature_c": 13.8},
    {"station_id": "9414750", "name": "Alameda", "lat": 37.771, "lon": -122.3, "speed": 4.9, "direction": 296.0, "water_temperature_c": 14.2},
    {"station_id": "9413450", "name": "Richmond", "lat": 37.9233, "lon": -122.415, "speed": 5.5, "direction": 273.0, "water_temperature_c": 13.5},
]


@dataclass
class VectorObservation:
    station_id: str
    name: str
    lat: float
    lon: float
    speed: float
    direction_deg: float
    water_temperature_c: float | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _forecast_hours(max_horizon: int) -> list[int]:
    hours = list(range(1, min(max_horizon, 24) + 1))
    rolling = 27
    while rolling <= max_horizon:
        hours.append(rolling)
        rolling += 3
    if max_horizon > 24 and hours[-1] != max_horizon:
        hours.append(max_horizon)
    return hours


def _in_bounds(lat: float, lon: float) -> bool:
    return (
        SF_BAY_BOUNDS["lat_min"] <= lat <= SF_BAY_BOUNDS["lat_max"]
        and SF_BAY_BOUNDS["lon_min"] <= lon <= SF_BAY_BOUNDS["lon_max"]
    )


def _to_float(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])
    raise KeyError(f"Missing numeric keys {keys} in payload: {payload}")


def _vector_components(speed: float, direction_deg: float) -> tuple[float, float]:
    direction_rad = radians(direction_deg)
    u = speed * sin(direction_rad)
    v = speed * cos(direction_rad)
    return (u, v)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2) + cos(radians(lat1)) * cos(radians(lat2)) * (sin(dlon / 2) ** 2)
    return 2 * radius * atan2(sqrt(a), sqrt(max(1 - a, 0)))


def _nearest_weight(lat: float, lon: float, station_lat: float, station_lon: float) -> float:
    return 1.0 / max(_haversine_km(lat, lon, station_lat, station_lon), 0.2)


def _interpolate_vector(lat: float, lon: float, vectors: list[VectorObservation]) -> tuple[float, float]:
    u_total = 0.0
    v_total = 0.0
    weight_total = 0.0
    for vector in vectors:
        weight = _nearest_weight(lat, lon, vector.lat, vector.lon)
        u_component, v_component = _vector_components(vector.speed, vector.direction_deg)
        u_total += u_component * weight
        v_total += v_component * weight
        weight_total += weight
    return (u_total / max(weight_total, 1e-6), v_total / max(weight_total, 1e-6))


def _interpolate_temperature(lat: float, lon: float, vectors: list[VectorObservation]) -> float | None:
    weighted_temp = 0.0
    weight_total = 0.0
    for vector in vectors:
        if vector.water_temperature_c is None:
            continue
        weight = _nearest_weight(lat, lon, vector.lat, vector.lon)
        weighted_temp += vector.water_temperature_c * weight
        weight_total += weight
    if weight_total == 0:
        return None
    return round(weighted_temp / weight_total, 2)


def _build_frame(
    valid_at: datetime,
    horizon_hour: int,
    current_vectors: list[VectorObservation],
    wind_vectors: list[VectorObservation],
) -> OperationalGridFrame:
    grid: list[GridPoint] = []
    for cell in SF_BAY_CELLS:
        current_u, current_v = _interpolate_vector(cell["lat"], cell["lon"], current_vectors)
        wind_u, wind_v = _interpolate_vector(cell["lat"], cell["lon"], wind_vectors)
        grid.append(
            GridPoint(
                cell_id=cell["cell_id"],
                lat=cell["lat"],
                lon=cell["lon"],
                current_u=round(current_u, 4),
                current_v=round(current_v, 4),
                wind_u=round(wind_u, 4),
                wind_v=round(wind_v, 4),
                shoreline_proximity=cell["shoreline_proximity"],
                restricted=cell["restricted"],
                water_temperature_c=_interpolate_temperature(cell["lat"], cell["lon"], wind_vectors),
            )
        )
    return OperationalGridFrame(valid_at=valid_at, horizon_hour=horizon_hour, grid=grid)


def _sample_current_vectors(horizon_hour: int, seed: int) -> list[VectorObservation]:
    vectors: list[VectorObservation] = []
    for index, station in enumerate(SAMPLE_CURRENT_STATIONS):
        wobble = sin((horizon_hour + seed + index) * 0.24 + station["phase"])
        speed = max(0.12, station["speed"] + (0.18 * wobble))
        direction = (station["direction"] + (17.0 * wobble)) % 360
        vectors.append(
            VectorObservation(
                station_id=station["station_id"],
                name=station["name"],
                lat=station["lat"],
                lon=station["lon"],
                speed=round(speed, 3),
                direction_deg=round(direction, 2),
            )
        )
    return vectors


def _sample_wind_vectors(horizon_hour: int, seed: int) -> list[VectorObservation]:
    vectors: list[VectorObservation] = []
    for index, station in enumerate(SAMPLE_WIND_STATIONS):
        wobble = sin((horizon_hour + seed + index) * 0.18)
        speed = max(0.5, station["speed"] + (0.8 * wobble))
        direction = (station["direction"] + (12.0 * wobble)) % 360
        vectors.append(
            VectorObservation(
                station_id=station["station_id"],
                name=station["name"],
                lat=station["lat"],
                lon=station["lon"],
                speed=round(speed, 3),
                direction_deg=round(direction, 2),
                water_temperature_c=station["water_temperature_c"] + (0.2 * wobble),
            )
        )
    return vectors


def _load_sample_context(horizon_hours: int, seed: int, requested_mode: SourceMode) -> OperationalContext:
    generated_at = _now()
    frames = []
    for horizon_hour in _forecast_hours(horizon_hours):
        frames.append(
            _build_frame(
                valid_at=generated_at + timedelta(hours=horizon_hour),
                horizon_hour=horizon_hour,
                current_vectors=_sample_current_vectors(horizon_hour, seed),
                wind_vectors=_sample_wind_vectors(horizon_hour, seed),
            )
        )
    return OperationalContext(
        generated_at=generated_at,
        pilot_region=settings.pilot_region,
        source_mode_requested=requested_mode,
        source_mode_used="sample",
        is_fallback=False,
        source_notes=["Using deterministic SF Bay NOAA-style fixture data."],
        frames=frames,
    )


def _coops_station_payload_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("stations", "stationList", "stationsList"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _series_payload_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("current_predictions", "predictions", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _fetch_json(client: httpx.Client, url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = client.get(url, params=params)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    write_raw_payload("ingest", {"url": str(response.request.url), "payload": payload})
    return payload


def _fetch_live_current_stations(client: httpx.Client) -> list[dict[str, Any]]:
    metadata_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    for station_type in ("currentpredictions", "currents"):
        payload = _fetch_json(client, metadata_url, {"type": station_type})
        items = [
            item
            for item in _coops_station_payload_items(payload)
            if _in_bounds(_to_float(item, "lat"), _to_float(item, "lng", "lon"))
        ]
        if items:
            return items[:4]
    return []


def _fetch_live_met_stations(client: httpx.Client) -> list[dict[str, Any]]:
    metadata_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    payload = _fetch_json(client, metadata_url, {"type": "waterlevels"})
    items = [
        item
        for item in _coops_station_payload_items(payload)
        if _in_bounds(_to_float(item, "lat"), _to_float(item, "lng", "lon"))
    ]
    return items[:5]


def _parse_timestamp(raw_time: str) -> datetime:
    normalized = raw_time.replace("Z", "+00:00").replace(" ", "T")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_current_prediction_series(
    station: dict[str, Any],
    payload: dict[str, Any],
    generated_at: datetime,
    horizon_hours: int,
) -> dict[int, VectorObservation]:
    observations_by_hour: dict[int, VectorObservation] = {}
    raw_records = _series_payload_items(payload)
    for horizon_hour in _forecast_hours(horizon_hours):
        target = generated_at + timedelta(hours=horizon_hour)
        chosen: dict[str, Any] | None = None
        chosen_delta: float | None = None
        for record in raw_records:
            raw_time = record.get("Time") or record.get("t")
            if raw_time is None:
                continue
            record_time = _parse_timestamp(str(raw_time))
            delta_seconds = abs((record_time - target).total_seconds())
            if chosen is None or delta_seconds < (chosen_delta or float("inf")):
                chosen = record
                chosen_delta = delta_seconds
        if chosen is None:
            continue
        speed = float(chosen.get("Speed") or chosen.get("s") or 0.0)
        direction = float(chosen.get("Direction") or chosen.get("d") or 0.0)
        observations_by_hour[horizon_hour] = VectorObservation(
            station_id=str(station["id"]),
            name=str(station["name"]),
            lat=_to_float(station, "lat"),
            lon=_to_float(station, "lng", "lon"),
            speed=speed,
            direction_deg=direction,
        )
    return observations_by_hour


def _fetch_live_current_series(
    client: httpx.Client,
    station: dict[str, Any],
    generated_at: datetime,
    horizon_hours: int,
) -> dict[int, VectorObservation]:
    payload = _fetch_json(
        client,
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        {
            "product": "currents_predictions",
            "application": "SeaSweep",
            "station": station["id"],
            "begin_date": generated_at.strftime("%Y%m%d"),
            "end_date": (generated_at + timedelta(hours=horizon_hours + 12)).strftime("%Y%m%d"),
            "time_zone": "gmt",
            "units": "metric",
            "interval": "h",
            "vel_type": "speed_dir",
            "format": "json",
        },
    )
    return _parse_current_prediction_series(station, payload, generated_at, horizon_hours)


def _fetch_latest_wind(client: httpx.Client, station_id: str) -> dict[str, Any]:
    return _fetch_json(
        client,
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        {
            "product": "wind",
            "application": "SeaSweep",
            "date": "latest",
            "station": station_id,
            "time_zone": "gmt",
            "units": "metric",
            "format": "json",
        },
    )


def _fetch_latest_water_temperature(client: httpx.Client, station_id: str) -> dict[str, Any]:
    return _fetch_json(
        client,
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        {
            "product": "water_temperature",
            "application": "SeaSweep",
            "date": "latest",
            "station": station_id,
            "time_zone": "gmt",
            "units": "metric",
            "format": "json",
        },
    )


def _parse_wind_observation(station: dict[str, Any], wind_payload: dict[str, Any], temp_payload: dict[str, Any] | None) -> VectorObservation | None:
    wind_items = _series_payload_items(wind_payload)
    if not wind_items:
        return None
    latest = wind_items[-1]
    speed = float(latest.get("s") or latest.get("Speed") or 0.0)
    direction = float(latest.get("d") or latest.get("Direction") or 0.0)
    water_temp = None
    if temp_payload is not None:
        temp_items = _series_payload_items(temp_payload)
        if temp_items:
            water_temp = float(temp_items[-1].get("v") or temp_items[-1].get("Value") or 0.0)
    return VectorObservation(
        station_id=str(station["id"]),
        name=str(station["name"]),
        lat=_to_float(station, "lat"),
        lon=_to_float(station, "lng", "lon"),
        speed=speed,
        direction_deg=direction,
        water_temperature_c=water_temp,
    )


def _load_live_context(horizon_hours: int, requested_mode: SourceMode) -> OperationalContext:
    generated_at = _now()
    timeout = httpx.Timeout(settings.live_request_timeout_seconds)
    with httpx.Client(timeout=timeout) as client:
        current_stations = _fetch_live_current_stations(client)
        if not current_stations:
            raise RuntimeError("No NOAA current-prediction stations were available for the SF Bay bounds.")

        current_vectors_by_hour: dict[int, list[VectorObservation]] = {hour: [] for hour in _forecast_hours(horizon_hours)}
        for station in current_stations:
            station_series = _fetch_live_current_series(client, station, generated_at, horizon_hours)
            for hour, observation in station_series.items():
                current_vectors_by_hour.setdefault(hour, []).append(observation)

        met_vectors: list[VectorObservation] = []
        for station in _fetch_live_met_stations(client):
            try:
                wind_payload = _fetch_latest_wind(client, str(station["id"]))
                try:
                    temp_payload = _fetch_latest_water_temperature(client, str(station["id"]))
                except Exception:
                    temp_payload = None
                parsed = _parse_wind_observation(station, wind_payload, temp_payload)
                if parsed is not None:
                    met_vectors.append(parsed)
            except Exception:
                continue

    if not met_vectors:
        raise RuntimeError("No NOAA met stations with wind observations were available for the SF Bay bounds.")

    frames: list[OperationalGridFrame] = []
    for horizon_hour in _forecast_hours(horizon_hours):
        current_vectors = current_vectors_by_hour.get(horizon_hour, [])
        if not current_vectors:
            raise RuntimeError(f"Missing live current predictions for horizon {horizon_hour}h.")
        frames.append(
            _build_frame(
                valid_at=generated_at + timedelta(hours=horizon_hour),
                horizon_hour=horizon_hour,
                current_vectors=current_vectors,
                wind_vectors=met_vectors,
            )
        )

    return OperationalContext(
        generated_at=generated_at,
        pilot_region=settings.pilot_region,
        source_mode_requested=requested_mode,
        source_mode_used="live",
        is_fallback=False,
        source_notes=[
            f"NOAA CO-OPS live current-prediction stations: {len(current_stations)}",
            f"NOAA CO-OPS live met stations: {len(met_vectors)}",
        ],
        frames=frames,
    )


def load_operational_context(horizon_hours: int, seed: int, source_mode: SourceMode) -> OperationalContext:
    if source_mode == "sample":
        return _load_sample_context(horizon_hours, seed, source_mode)
    if source_mode == "live":
        return _load_live_context(horizon_hours, source_mode)
    try:
        return _load_live_context(horizon_hours, source_mode)
    except Exception as exc:
        context = _load_sample_context(horizon_hours, seed, source_mode)
        context.is_fallback = True
        context.source_notes.append(f"Live ingest failed and fell back to sample data: {exc}")
        return context

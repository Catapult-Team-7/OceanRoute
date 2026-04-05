from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from db.database import get_repo
from db.demo_data import DemoOceanRepository, REGION_BOUNDS
from db.models import PortHint, TrashHotspot, TrashResponse
from ingest.fetch_trash import TrashDataError, load_trash_observations
from ingest.fetch_world_port_index import DEFAULT_WORLD_PORT_INDEX_URL, fetch_world_ports, nearest_port

router = APIRouter()


def _region_bounds(region: str) -> tuple[float, float, float, float]:
    min_lat, max_lat, min_lon, max_lon = REGION_BOUNDS.get(region, REGION_BOUNDS["global"])
    return min_lat, max_lat, min_lon, max_lon


def _trash_data_url(repo: DemoOceanRepository) -> tuple[str | None, str]:
    emodnet = repo.trainer._config_by_id("emodnet_litter") or {}
    oceanscan = repo.trainer._config_by_id("oceanscan") or {}
    emodnet_url = str(emodnet.get("notes", "")).strip() or str(emodnet.get("url", "")).strip()
    oceanscan_url = str(oceanscan.get("notes", "")).strip() or str(oceanscan.get("url", "")).strip()
    if emodnet.get("enabled") and emodnet_url and "emodnet.ec.europa.eu/en/chemistry" not in emodnet_url:
        return emodnet_url, "emodnet_litter"
    if oceanscan.get("enabled") and oceanscan_url and "oceanscan.org" not in oceanscan_url:
        return oceanscan_url, "oceanscan"
    return None, "unconfigured"


def _distance_score(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    return ((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2) ** 0.5


def _normalize_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _trash_operational_weight(lat: float) -> float:
    absolute_lat = abs(float(lat))
    if absolute_lat <= 45.0:
        return 1.0
    if absolute_lat >= 62.0:
        return 0.0
    return max(0.0, 1.0 - ((absolute_lat - 45.0) / 17.0))


def _nearest_axis_value(values: list[float], target: float, *, longitude: bool = False) -> float:
    if longitude:
        normalized_target = _normalize_lon(target)
        return min(values, key=lambda value: abs(_normalize_lon(value - normalized_target)))
    return min(values, key=lambda value: abs(value - target))


def _advect_with_currents(lat: float, lon: float, current_u: float | None, current_v: float | None) -> tuple[float, float]:
    if current_u is None and current_v is None:
        return lat, lon
    eastward = float(current_u or 0.0)
    northward = float(current_v or 0.0)
    lat_shift = max(-4.5, min(4.5, northward * 3.2))
    lon_scale = max(0.35, abs(math.cos(math.radians(lat))))
    lon_shift = max(-7.5, min(7.5, (eastward * 4.8) / lon_scale))
    advected_lat = max(-84.0, min(84.0, lat + lat_shift))
    advected_lon = _normalize_lon(lon + lon_shift)
    return advected_lat, advected_lon


def _advance_transport_step(row) -> tuple[float, float]:
    eastward = float(getattr(row, "current_u", 0.0) or 0.0)
    northward = float(getattr(row, "current_v", 0.0) or 0.0)
    lat_shift = max(-2.0, min(2.0, northward * 1.4))
    lon_scale = max(0.35, abs(math.cos(math.radians(row.lat))))
    lon_shift = max(-3.0, min(3.0, (eastward * 2.2) / lon_scale))
    next_lat = max(-84.0, min(84.0, row.lat + lat_shift))
    next_lon = _normalize_lon(row.lon + lon_shift)
    return next_lat, next_lon


def _select_spaced_rows(rows, *, limit: int, min_distance_deg: float):
    selected = []
    for row in rows:
        if all(_distance_score(row.lat, row.lon, item.lat, item.lon) >= min_distance_deg for item in selected):
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _nearest_ocean_row(rows, lat: float, lon: float):
    if not rows:
        return None
    return min(rows, key=lambda row: _distance_score(lat, lon, row.lat, row.lon))


def _build_row_lookup(rows):
    lat_values = sorted({round(float(row.lat), 6) for row in rows})
    lon_values = sorted({round(float(row.lon), 6) for row in rows})
    lookup = {(round(float(row.lat), 6), round(float(row.lon), 6)): row for row in rows}
    return lat_values, lon_values, lookup


def _nearest_supported_row(lat_values, lon_values, lookup, lat: float, lon: float):
    if not lookup:
        return None
    nearest_lat = _nearest_axis_value(lat_values, lat, longitude=False)
    nearest_lon = _nearest_axis_value(lon_values, lon, longitude=True)
    direct = lookup.get((round(float(nearest_lat), 6), round(float(nearest_lon), 6)))
    if direct is not None:
        return direct
    for lat_candidate in sorted(lat_values, key=lambda value: abs(value - lat))[:4]:
        for lon_candidate in sorted(lon_values, key=lambda value: abs(_normalize_lon(value - lon)))[:6]:
            candidate = lookup.get((round(float(lat_candidate), 6), round(float(lon_candidate), 6)))
            if candidate is not None:
                return candidate
    return None


def _trace_transport_path(seed_row, lat_values, lon_values, lookup, steps: int = 6):
    current_row = seed_row
    path = [(seed_row.lon, seed_row.lat)]
    for _ in range(steps):
        next_lat, next_lon = _advance_transport_step(current_row)
        next_row = _nearest_supported_row(lat_values, lon_values, lookup, next_lat, next_lon)
        if next_row is None:
            break
        if _distance_score(next_row.lat, next_row.lon, current_row.lat, current_row.lon) < 0.25:
            break
        path.append((next_row.lon, next_row.lat))
        current_row = next_row
    return current_row, path


def _predicted_trash_hotspots(
    repo: DemoOceanRepository,
    *,
    region: str,
    limit: int,
    ports: list,
    observed_items: list[dict] | None = None,
) -> list[TrashHotspot]:
    observed_items = observed_items or []
    rows = repo.get_flux_grid(date=None, resolution="2deg" if region == "global" else "1deg", region=region)
    lat_values, lon_values, row_lookup = _build_row_lookup(rows)
    ranked = sorted(
        rows,
        key=lambda row: (
            (
                (row.route_priority or 0.0) * 0.95
                + (row.weakening_score or 0.0) * 0.75
                + (row.anomaly_score or 0.0) * 0.45
                + min(
                    1.5,
                    math.sqrt(
                        (float(getattr(row, "current_u", 0.0) or 0.0) ** 2)
                        + (float(getattr(row, "current_v", 0.0) or 0.0) ** 2)
                    ),
                )
            )
            * _trash_operational_weight(row.lat)
        ),
        reverse=True,
    )
    candidates = [
        row
        for row in ranked
        if _trash_operational_weight(row.lat) > 0
        and (
            (row.route_priority or 0.0) >= 0.28
            or (row.weakening_score or 0.0) >= 0.18
            or (row.anomaly_score or 0.0) >= 0.22
        )
    ][: max(limit * 8, 80)]

    spaced_candidates = _select_spaced_rows(
        candidates,
        limit=limit,
        min_distance_deg=10.0 if region == "global" else 5.0,
    )

    endpoint_scores: dict[tuple[float, float], dict] = {}
    for row in spaced_candidates:
        endpoint_row, path = _trace_transport_path(row, lat_values, lon_values, row_lookup, steps=7 if region == "global" else 5)
        endpoint_key = (round(float(endpoint_row.lat), 6), round(float(endpoint_row.lon), 6))
        base_score = (
            (row.route_priority or 0.0) * 1.05
            + (row.weakening_score or 0.0) * 0.8
            + (row.anomaly_score or 0.0) * 0.55
        ) * _trash_operational_weight(endpoint_row.lat)
        bucket = endpoint_scores.get(endpoint_key) or {
            "row": endpoint_row,
            "score": 0.0,
            "source_count": 0,
            "best_path": path,
            "best_seed": row,
        }
        bucket["score"] += base_score
        bucket["source_count"] += 1
        if len(path) > len(bucket["best_path"]) or base_score > (
            (bucket["best_seed"].route_priority or 0.0) + (bucket["best_seed"].weakening_score or 0.0)
        ):
            bucket["best_path"] = path
            bucket["best_seed"] = row
        endpoint_scores[endpoint_key] = bucket

    ranked_endpoints = sorted(
        endpoint_scores.values(),
        key=lambda item: item["score"] * (1.0 + 0.55 * max(0, item["source_count"] - 1)),
        reverse=True,
    )
    selected_endpoints = []
    for candidate in ranked_endpoints:
        endpoint_row = candidate["row"]
        if _trash_operational_weight(endpoint_row.lat) <= 0:
            continue
        if all(
            _distance_score(endpoint_row.lat, endpoint_row.lon, selected["row"].lat, selected["row"].lon)
            >= (10.0 if region == "global" else 5.0)
            for selected in selected_endpoints
        ):
            selected_endpoints.append(candidate)
        if len(selected_endpoints) >= limit:
            break

    hotspots: list[TrashHotspot] = []
    for index, candidate in enumerate(selected_endpoints):
        endpoint_row = candidate["row"]
        hotspot_lat = endpoint_row.lat
        hotspot_lon = endpoint_row.lon
        port = nearest_port(hotspot_lat, hotspot_lon, ports=ports) if ports else None
        cross_check = None
        if observed_items:
            nearest_observed = min(
                observed_items,
                key=lambda item: _distance_score(hotspot_lat, hotspot_lon, item["lat"], item["lon"]),
            )
            if _distance_score(hotspot_lat, hotspot_lon, nearest_observed["lat"], nearest_observed["lon"]) <= 8:
                cross_check = {
                    "label": nearest_observed["label"],
                    "lat": nearest_observed["lat"],
                    "lon": nearest_observed["lon"],
                    "intensity": nearest_observed.get("intensity", 0.0),
                    "source": nearest_observed.get("source", "observed_trash"),
                }
        hotspots.append(
            TrashHotspot(
                id=f"model-trash-{index}",
                label="Predicted trash convergence zone",
                lat=hotspot_lat,
                lon=hotspot_lon,
                intensity=round(
                    candidate["score"],
                    4,
                ),
                observed=False,
                source="ml_predicted_trash_transport",
                nearest_port=(
                    PortHint(
                        name=port.name,
                        lat=port.lat,
                        lon=port.lon,
                        country=port.country,
                        source=port.source,
                    )
                    if port
                    else None
                ),
                metadata={
                    "source_lat": candidate["best_seed"].lat,
                    "source_lon": candidate["best_seed"].lon,
                    "transport_path": candidate["best_path"],
                    "source_count": candidate["source_count"],
                    "route_priority": candidate["best_seed"].route_priority or 0.0,
                    "weakening_score": candidate["best_seed"].weakening_score or 0.0,
                    "anomaly_score": candidate["best_seed"].anomaly_score or 0.0,
                    "current_u": getattr(endpoint_row, "current_u", None),
                    "current_v": getattr(endpoint_row, "current_v", None),
                    "cross_checked_observation": cross_check,
                },
            )
        )
    return hotspots


@router.get("/trash", response_model=TrashResponse)
async def trash_hotspots(
    region: str = Query(default="global", pattern="^(global|pacific|atlantic|indian)$"),
    limit: int = Query(default=25, ge=1, le=200),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    data_url, source_name = _trash_data_url(repo)
    min_lat, max_lat, min_lon, max_lon = _region_bounds(region)
    items: list[dict] = []
    observed_summary = ""
    if data_url:
        try:
            items = load_trash_observations(data_url, limit=limit * 3)
            observed_summary = "Measured trash observations were loaded and used to cross-check model-predicted trash clusters."
        except TrashDataError as exc:
            observed_summary = f"Trash observation feed was unavailable for cross-checking: {exc}"
    else:
        observed_summary = "No measured trash feed is configured yet, so trash clusters come purely from the ML transport prediction."
    filtered = [
        item
        for item in items
        if min_lat <= item["lat"] <= max_lat and (min_lon <= max_lon and min_lon <= item["lon"] <= max_lon or min_lon > max_lon)
    ]
    if min_lon > max_lon:
        filtered = [item for item in items if min_lat <= item["lat"] <= max_lat and (item["lon"] >= min_lon or item["lon"] <= max_lon)]

    filtered = sorted(filtered, key=lambda item: item.get("intensity", 0.0), reverse=True)[:limit]

    port_config = repo.trainer._config_by_id("world_port_index") or {}
    port_url = str(port_config.get("notes", "")).strip() or str(port_config.get("url", "")).strip() or DEFAULT_WORLD_PORT_INDEX_URL
    ports = []
    try:
        ports = fetch_world_ports(
            feature_service_url=port_url if "FeatureServer" in port_url else DEFAULT_WORLD_PORT_INDEX_URL,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon if min_lon <= max_lon else -180,
            max_lon=max_lon if min_lon <= max_lon else 180,
            limit=250,
        )
    except Exception:
        ports = []

    hotspots = _predicted_trash_hotspots(repo, region=region, limit=limit, ports=ports, observed_items=filtered)

    return TrashResponse(
        source="ml_predicted_trash_transport",
        observed=bool(filtered),
        source_summary=observed_summary,
        hotspots=hotspots,
    )

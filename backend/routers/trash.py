from __future__ import annotations

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
    ranked = sorted(
        rows,
        key=lambda row: (
            (row.route_priority or 0.0) * 1.3
            + (row.weakening_score or 0.0)
            + (row.anomaly_score or 0.0) * 0.8
        ),
        reverse=True,
    )
    candidates = [
        row
        for row in ranked
        if (row.route_priority or 0.0) >= 0.2 or (row.weakening_score or 0.0) >= 0.15 or (row.anomaly_score or 0.0) >= 0.2
    ][: max(limit * 3, 12)]

    hotspots: list[TrashHotspot] = []
    for index, row in enumerate(candidates[:limit]):
        port = nearest_port(row.lat, row.lon, ports=ports) if ports else None
        cross_check = None
        if observed_items:
            nearest_observed = min(
                observed_items,
                key=lambda item: _distance_score(row.lat, row.lon, item["lat"], item["lon"]),
            )
            if _distance_score(row.lat, row.lon, nearest_observed["lat"], nearest_observed["lon"]) <= 8:
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
                label="Predicted trash cluster",
                lat=row.lat,
                lon=row.lon,
                intensity=round(
                    (row.route_priority or 0.0) + (row.weakening_score or 0.0) + (row.anomaly_score or 0.0),
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
                    "route_priority": row.route_priority or 0.0,
                    "weakening_score": row.weakening_score or 0.0,
                    "anomaly_score": row.anomaly_score or 0.0,
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

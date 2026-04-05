from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

import requests


DEFAULT_WORLD_PORT_INDEX_URL = (
    "https://vcps.nga.mil/nauticalpubs-feature/rest/services/WPI/World_Port_Index_Viewer/FeatureServer/0"
)


@dataclass
class PortRecord:
    name: str
    country: str | None
    lat: float
    lon: float
    harbor_size: str | None = None
    harbor_type: str | None = None
    source: str = "nga_world_port_index"


def _extract_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_port_name(attributes: dict[str, Any]) -> str:
    for key in ("PORT_NAME", "MAIN_PORT_NAME", "PORT", "NAME", "PORT_NM"):
        value = attributes.get(key)
        if value:
            return str(value)
    return "Unknown port"


def _extract_country(attributes: dict[str, Any]) -> str | None:
    for key in ("COUNTRY", "COUNTRY_NAME", "CTRY_NAME", "NATION"):
        value = attributes.get(key)
        if value:
            return str(value)
    return None


def _extract_lat_lon(feature: dict[str, Any]) -> tuple[float, float] | None:
    geometry = feature.get("geometry") or {}
    x = _extract_number(geometry.get("x"))
    y = _extract_number(geometry.get("y"))
    if x is not None and y is not None:
        return y, x

    attributes = feature.get("attributes") or {}
    lat = None
    lon = None
    for key in ("LATITUDE", "LAT", "Y"):
        lat = _extract_number(attributes.get(key))
        if lat is not None:
            break
    for key in ("LONGITUDE", "LON", "LONG", "X"):
        lon = _extract_number(attributes.get(key))
        if lon is not None:
            break
    if lat is None or lon is None:
        return None
    return lat, lon


def fetch_world_ports(
    *,
    feature_service_url: str = DEFAULT_WORLD_PORT_INDEX_URL,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    limit: int = 100,
    timeout: int = 25,
) -> list[PortRecord]:
    query_url = feature_service_url.rstrip("/") + "/query"
    where = "1=1"
    if None not in {min_lat, max_lat, min_lon, max_lon}:
      where = (
            f"LATITUDE >= {min_lat} AND LATITUDE <= {max_lat} AND "
            f"LONGITUDE >= {min_lon} AND LONGITUDE <= {max_lon}"
        )

    response = requests.get(
        query_url,
        params={
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "f": "json",
            "resultRecordCount": max(1, min(limit, 500)),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features", [])

    ports: list[PortRecord] = []
    for feature in features:
        attributes = feature.get("attributes") or {}
        lat_lon = _extract_lat_lon(feature)
        if lat_lon is None:
            continue
        lat, lon = lat_lon
        ports.append(
            PortRecord(
                name=_extract_port_name(attributes),
                country=_extract_country(attributes),
                lat=lat,
                lon=lon,
                harbor_size=str(attributes.get("HARBOR_SIZE") or attributes.get("HARB_SIZE") or "") or None,
                harbor_type=str(attributes.get("HARBOR_TYPE") or attributes.get("HARB_TYPE") or "") or None,
            )
        )
    return ports


def nearest_port(
    lat: float,
    lon: float,
    *,
    ports: list[PortRecord],
) -> PortRecord | None:
    if not ports:
        return None
    return min(ports, key=lambda port: hypot((port.lat - lat) / 8, (port.lon - lon) / 10))

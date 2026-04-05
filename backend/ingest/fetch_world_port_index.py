from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import hypot
from typing import Any

import requests
import urllib3


DEFAULT_WORLD_PORT_INDEX_URL = (
    "https://vcps.nga.mil/nauticalpubs-feature/rest/services/WPI/World_Port_Index_Viewer/FeatureServer/0"
)
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    for key in ("main_port_name", "port_name", "port", "name", "port_nm", "MAIN_PORT_NAME", "PORT_NAME", "PORT", "NAME", "PORT_NM"):
        value = attributes.get(key)
        if value:
            return str(value)
    return "Unknown port"


def _extract_country(attributes: dict[str, Any]) -> str | None:
    for key in ("country", "country_name", "ctry_name", "nation", "wpi_cc", "COUNTRY", "COUNTRY_NAME", "CTRY_NAME", "NATION", "WPI_CC"):
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
    for key in ("latitude", "lat", "y", "LATITUDE", "LAT", "Y"):
        lat = _extract_number(attributes.get(key))
        if lat is not None:
            break
    for key in ("longitude", "lon", "long", "x", "LONGITUDE", "LON", "LONG", "X"):
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
    offset: int = 0,
    timeout: int = 25,
) -> list[PortRecord]:
    query_url = feature_service_url.rstrip("/") + "/query"
    def _query(payload_where: str, payload_offset: int, payload_limit: int) -> dict:
        response = requests.get(
            query_url,
            params={
                "where": payload_where,
                "outFields": "*",
                "returnGeometry": "true",
                "f": "json",
                "resultRecordCount": max(1, min(payload_limit, 250)),
                "resultOffset": max(0, int(payload_offset)),
            },
            timeout=timeout,
            verify=False,
        )
        response.raise_for_status()
        return response.json()

    ports: list[PortRecord] = []
    seen: set[tuple[str, float, float]] = set()
    page_size = 200
    current_offset = max(0, int(offset))
    max_pages = 30
    for _ in range(max_pages):
        payload = _query("1=1", current_offset, page_size)
        features = payload.get("features", [])
        if not features:
            break
        current_offset += len(features)
        for feature in features:
            attributes = feature.get("attributes") or {}
            lat_lon = _extract_lat_lon(feature)
            if lat_lon is None:
                continue
            lat, lon = lat_lon
            if min_lat is not None and lat < min_lat:
                continue
            if max_lat is not None and lat > max_lat:
                continue
            if min_lon is not None and max_lon is not None:
                if min_lon <= max_lon:
                    if lon < min_lon or lon > max_lon:
                        continue
                elif not (lon >= min_lon or lon <= max_lon):
                    continue
            name = _extract_port_name(attributes)
            if not name or name == "Unknown port":
                continue
            key = (name, round(lat, 4), round(lon, 4))
            if key in seen:
                continue
            seen.add(key)
            ports.append(
                PortRecord(
                    name=name,
                    country=_extract_country(attributes),
                    lat=lat,
                    lon=lon,
                    harbor_size=str(attributes.get("harbor_size_code") or attributes.get("harb_size") or attributes.get("HARBOR_SIZE") or attributes.get("HARB_SIZE") or "") or None,
                    harbor_type=str(attributes.get("harbor_type_code") or attributes.get("harb_type") or attributes.get("HARBOR_TYPE") or attributes.get("HARB_TYPE") or "") or None,
                )
            )
            if len(ports) >= limit:
                return ports
    return ports


@lru_cache(maxsize=256)
def _fetch_world_ports_nearby_cached(
    *,
    lat: float,
    lon: float,
    lat_radius: float,
    lon_radius: float,
    feature_service_url: str = DEFAULT_WORLD_PORT_INDEX_URL,
    limit: int = 40,
    timeout: int = 20,
) -> tuple[PortRecord, ...]:
    query_url = feature_service_url.rstrip("/") + "/query"
    min_lon = max(-180.0, lon - lon_radius)
    max_lon = min(180.0, lon + lon_radius)
    min_lat = max(-90.0, lat - lat_radius)
    max_lat = min(90.0, lat + lat_radius)
    response = requests.get(
        query_url,
        params={
            "where": "1=1",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "json",
            "resultRecordCount": max(1, min(limit, 100)),
        },
        timeout=timeout,
        verify=False,
    )
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features", [])
    ports: list[PortRecord] = []
    seen: set[tuple[str, float, float]] = set()
    for feature in features:
        attributes = feature.get("attributes") or {}
        lat_lon = _extract_lat_lon(feature)
        if lat_lon is None:
            continue
        port_lat, port_lon = lat_lon
        name = _extract_port_name(attributes)
        if not name or name == "Unknown port":
            continue
        key = (name, round(port_lat, 4), round(port_lon, 4))
        if key in seen:
            continue
        seen.add(key)
        ports.append(
            PortRecord(
                name=name,
                country=_extract_country(attributes),
                lat=port_lat,
                lon=port_lon,
                harbor_size=str(attributes.get("harbor_size_code") or attributes.get("harb_size") or attributes.get("HARBOR_SIZE") or attributes.get("HARB_SIZE") or "") or None,
                harbor_type=str(attributes.get("harbor_type_code") or attributes.get("harb_type") or attributes.get("HARBOR_TYPE") or attributes.get("HARB_TYPE") or "") or None,
            )
        )
    return tuple(ports)


def fetch_world_ports_nearby(
    *,
    lat: float,
    lon: float,
    lat_radius: float = 10.0,
    lon_radius: float = 14.0,
    feature_service_url: str = DEFAULT_WORLD_PORT_INDEX_URL,
    limit: int = 40,
    timeout: int = 20,
) -> list[PortRecord]:
    return list(
        _fetch_world_ports_nearby_cached(
            lat=round(float(lat), 2),
            lon=round(float(lon), 2),
            lat_radius=round(float(lat_radius), 2),
            lon_radius=round(float(lon_radius), 2),
            feature_service_url=feature_service_url,
            limit=limit,
            timeout=timeout,
        )
    )


@lru_cache(maxsize=128)
def _fetch_osm_nearby_ports_cached(
    *,
    lat: float,
    lon: float,
    radius_m: int = 900000,
    timeout: int = 30,
    overpass_url: str = DEFAULT_OVERPASS_URL,
) -> list[PortRecord]:
    query = f"""
    [out:json][timeout:{max(10, timeout)}];
    (
      node(around:{radius_m},{lat},{lon})["harbour"];
      node(around:{radius_m},{lat},{lon})["seamark:type"="harbour"];
      node(around:{radius_m},{lat},{lon})["landuse"="port"];
      node(around:{radius_m},{lat},{lon})["industrial"="port"];
      way(around:{radius_m},{lat},{lon})["harbour"];
      way(around:{radius_m},{lat},{lon})["seamark:type"="harbour"];
      way(around:{radius_m},{lat},{lon})["landuse"="port"];
      way(around:{radius_m},{lat},{lon})["industrial"="port"];
      relation(around:{radius_m},{lat},{lon})["harbour"];
      relation(around:{radius_m},{lat},{lon})["seamark:type"="harbour"];
      relation(around:{radius_m},{lat},{lon})["landuse"="port"];
      relation(around:{radius_m},{lat},{lon})["industrial"="port"];
    );
    out center tags;
    """
    response = requests.post(
        overpass_url,
        data=query,
        timeout=timeout,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    payload = response.json()
    elements = payload.get("elements", [])
    ports: list[PortRecord] = []
    for element in elements:
        tags = element.get("tags") or {}
        port_lat = _extract_number(element.get("lat"))
        port_lon = _extract_number(element.get("lon"))
        center = element.get("center") or {}
        if port_lat is None:
            port_lat = _extract_number(center.get("lat"))
        if port_lon is None:
            port_lon = _extract_number(center.get("lon"))
        if port_lat is None or port_lon is None:
            continue
        name = tags.get("name") or tags.get("seamark:name") or tags.get("name:en") or tags.get("operator")
        if not name:
            continue
        ports.append(
            PortRecord(
                name=str(name),
                country=tags.get("addr:country") or tags.get("country"),
                lat=float(port_lat),
                lon=float(port_lon),
                harbor_type=tags.get("harbour") or tags.get("seamark:type"),
                source="osm_overpass_port",
            )
        )
    unique: dict[tuple[str, float, float], PortRecord] = {}
    for port in ports:
        key = (port.name, round(port.lat, 4), round(port.lon, 4))
        unique[key] = port
    return tuple(unique.values())


def fetch_osm_nearby_ports(
    *,
    lat: float,
    lon: float,
    radius_m: int = 900000,
    timeout: int = 30,
    overpass_url: str = DEFAULT_OVERPASS_URL,
) -> list[PortRecord]:
    rounded_lat = round(float(lat), 2)
    rounded_lon = round(float(lon), 2)
    return list(
        _fetch_osm_nearby_ports_cached(
            lat=rounded_lat,
            lon=rounded_lon,
            radius_m=radius_m,
            timeout=timeout,
            overpass_url=overpass_url,
        )
    )


def nearest_port(
    lat: float,
    lon: float,
    *,
    ports: list[PortRecord],
) -> PortRecord | None:
    if not ports:
        return None
    def wrapped_lon_delta(port_lon: float, target_lon: float) -> float:
        delta = abs(port_lon - target_lon)
        return min(delta, abs(delta - 360.0))

    return min(ports, key=lambda port: hypot((port.lat - lat) / 8, wrapped_lon_delta(port.lon, lon) / 10))

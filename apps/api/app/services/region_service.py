from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.schemas import RegionInfo


@dataclass(frozen=True)
class RegionDefinition:
    id: str
    name: str
    description: str
    bbox: dict[str, float]
    default_depot_lat: float
    default_depot_lon: float
    grid_dx_km: float
    grid_dy_km: float
    tags: tuple[str, ...]
    cells: tuple[dict[str, float | bool | str], ...]
    sample_current_stations: tuple[dict[str, float | str], ...]
    sample_wind_stations: tuple[dict[str, float | str], ...]

    def to_info(self) -> RegionInfo:
        return RegionInfo(
            id=self.id,
            name=self.name,
            description=self.description,
            bbox=dict(self.bbox),
            default_depot_lat=self.default_depot_lat,
            default_depot_lon=self.default_depot_lon,
            grid_dx_km=self.grid_dx_km,
            grid_dy_km=self.grid_dy_km,
            tags=list(self.tags),
        )


REGIONS: dict[str, RegionDefinition] = {
    "sf_bay_estuary": RegionDefinition(
        id="sf_bay_estuary",
        name="San Francisco Bay Estuary",
        description="Harbor-estuary pilot for post-outflow and daily floating debris response.",
        bbox={"lat_min": 37.45, "lat_max": 38.25, "lon_min": -123.05, "lon_max": -121.75},
        default_depot_lat=37.8066,
        default_depot_lon=-122.4659,
        grid_dx_km=1.2,
        grid_dy_km=1.2,
        tags=("pilot", "harbor", "estuary"),
        cells=(
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
        ),
        sample_current_stations=(
            {"station_id": "gg01", "name": "Golden Gate current prediction", "lat": 37.8066, "lon": -122.4659, "speed": 1.28, "direction": 118.0, "phase": 0.2},
            {"station_id": "oak01", "name": "Oakland estuary current prediction", "lat": 37.7955, "lon": -122.2782, "speed": 0.84, "direction": 142.0, "phase": 1.1},
            {"station_id": "sb01", "name": "South Bay current prediction", "lat": 37.587, "lon": -122.06, "speed": 0.62, "direction": 171.0, "phase": 2.3},
        ),
        sample_wind_stations=(
            {"station_id": "9414290", "name": "San Francisco", "lat": 37.8063, "lon": -122.4659, "speed": 6.2, "direction": 284.0, "water_temperature_c": 13.8},
            {"station_id": "9414750", "name": "Alameda", "lat": 37.771, "lon": -122.3, "speed": 4.9, "direction": 296.0, "water_temperature_c": 14.2},
            {"station_id": "9413450", "name": "Richmond", "lat": 37.9233, "lon": -122.415, "speed": 5.5, "direction": 273.0, "water_temperature_c": 13.5},
        ),
    ),
    "puget_sound": RegionDefinition(
        id="puget_sound",
        name="Puget Sound",
        description="Large estuarine sound with port, ferry, and shoreline cleanup operations.",
        bbox={"lat_min": 47.0, "lat_max": 48.45, "lon_min": -123.35, "lon_max": -122.1},
        default_depot_lat=47.6026,
        default_depot_lon=-122.3398,
        grid_dx_km=1.8,
        grid_dy_km=1.8,
        tags=("estuary", "port", "northwest"),
        cells=(
            {"cell_id": "elliott_bay", "lat": 47.602, "lon": -122.353, "shoreline_proximity": 0.58, "restricted": False},
            {"cell_id": "duwamish_outflow", "lat": 47.565, "lon": -122.35, "shoreline_proximity": 0.74, "restricted": False},
            {"cell_id": "west_point", "lat": 47.661, "lon": -122.439, "shoreline_proximity": 0.49, "restricted": False},
            {"cell_id": "shilshole", "lat": 47.68, "lon": -122.41, "shoreline_proximity": 0.46, "restricted": False},
            {"cell_id": "bainbridge_east", "lat": 47.617, "lon": -122.48, "shoreline_proximity": 0.37, "restricted": False},
            {"cell_id": "tacoma_narrows", "lat": 47.269, "lon": -122.551, "shoreline_proximity": 0.61, "restricted": False},
            {"cell_id": "commencement_bay", "lat": 47.285, "lon": -122.432, "shoreline_proximity": 0.67, "restricted": False},
            {"cell_id": "everett_harbor", "lat": 47.985, "lon": -122.233, "shoreline_proximity": 0.71, "restricted": True},
        ),
        sample_current_stations=(
            {"station_id": "ps01", "name": "Elliott Bay current prediction", "lat": 47.6026, "lon": -122.3398, "speed": 0.92, "direction": 162.0, "phase": 0.3},
            {"station_id": "ps02", "name": "Tacoma Narrows current prediction", "lat": 47.269, "lon": -122.551, "speed": 1.15, "direction": 146.0, "phase": 1.2},
            {"station_id": "ps03", "name": "Everett current prediction", "lat": 47.985, "lon": -122.233, "speed": 0.74, "direction": 181.0, "phase": 2.1},
        ),
        sample_wind_stations=(
            {"station_id": "psw01", "name": "Seattle Harbor", "lat": 47.61, "lon": -122.339, "speed": 5.2, "direction": 211.0, "water_temperature_c": 11.4},
            {"station_id": "psw02", "name": "Tacoma", "lat": 47.273, "lon": -122.454, "speed": 4.6, "direction": 198.0, "water_temperature_c": 10.8},
            {"station_id": "psw03", "name": "Everett", "lat": 47.982, "lon": -122.227, "speed": 5.8, "direction": 221.0, "water_temperature_c": 10.9},
        ),
    ),
    "long_island_sound": RegionDefinition(
        id="long_island_sound",
        name="Long Island Sound",
        description="Tidal sound for interception and shoreline debris response between New York and Connecticut.",
        bbox={"lat_min": 40.8, "lat_max": 41.45, "lon_min": -73.95, "lon_max": -71.75},
        default_depot_lat=41.0518,
        default_depot_lon=-73.5401,
        grid_dx_km=2.0,
        grid_dy_km=2.0,
        tags=("tidal", "sound", "interception"),
        cells=(
            {"cell_id": "western_sound", "lat": 41.02, "lon": -73.62, "shoreline_proximity": 0.52, "restricted": False},
            {"cell_id": "stamford_reach", "lat": 41.01, "lon": -73.55, "shoreline_proximity": 0.63, "restricted": False},
            {"cell_id": "norwalk_islands", "lat": 41.065, "lon": -73.39, "shoreline_proximity": 0.59, "restricted": False},
            {"cell_id": "bridgeport_approach", "lat": 41.145, "lon": -73.175, "shoreline_proximity": 0.56, "restricted": False},
            {"cell_id": "new_haven_harbor", "lat": 41.265, "lon": -72.89, "shoreline_proximity": 0.69, "restricted": False},
            {"cell_id": "new_london_corridor", "lat": 41.322, "lon": -72.04, "shoreline_proximity": 0.48, "restricted": False},
            {"cell_id": "thames_mouth", "lat": 41.337, "lon": -72.087, "shoreline_proximity": 0.72, "restricted": True},
            {"cell_id": "central_sound", "lat": 41.15, "lon": -72.68, "shoreline_proximity": 0.31, "restricted": False},
        ),
        sample_current_stations=(
            {"station_id": "lis01", "name": "Stamford current prediction", "lat": 41.01, "lon": -73.546, "speed": 0.71, "direction": 88.0, "phase": 0.4},
            {"station_id": "lis02", "name": "Bridgeport current prediction", "lat": 41.153, "lon": -73.157, "speed": 0.84, "direction": 96.0, "phase": 1.6},
            {"station_id": "lis03", "name": "New London current prediction", "lat": 41.355, "lon": -72.093, "speed": 0.94, "direction": 104.0, "phase": 2.5},
        ),
        sample_wind_stations=(
            {"station_id": "lisw01", "name": "Stamford", "lat": 41.03, "lon": -73.542, "speed": 5.4, "direction": 245.0, "water_temperature_c": 12.7},
            {"station_id": "lisw02", "name": "Bridgeport", "lat": 41.161, "lon": -73.126, "speed": 4.8, "direction": 237.0, "water_temperature_c": 12.5},
            {"station_id": "lisw03", "name": "New London", "lat": 41.355, "lon": -72.087, "speed": 5.1, "direction": 228.0, "water_temperature_c": 12.1},
        ),
    ),
}


def list_regions() -> list[RegionInfo]:
    return [definition.to_info() for definition in REGIONS.values()]


def get_region_definition(region_id: str | None = None) -> RegionDefinition:
    resolved_region_id = region_id or settings.pilot_region
    if resolved_region_id not in REGIONS:
        available = ", ".join(sorted(REGIONS))
        raise KeyError(f"Unknown region '{resolved_region_id}'. Available regions: {available}")
    return REGIONS[resolved_region_id]


def get_region_info(region_id: str | None = None) -> RegionInfo:
    return get_region_definition(region_id).to_info()

from __future__ import annotations

from math import ceil

import numpy as np

from app.schemas import GridSpec
from app.services.region_service import RegionDefinition, get_region_definition


def build_region_grid_spec(region_id: str) -> GridSpec:
    region = get_region_definition(region_id)
    lat_span_km = (region.bbox["lat_max"] - region.bbox["lat_min"]) * 111.0
    mid_lat = (region.bbox["lat_min"] + region.bbox["lat_max"]) / 2
    lon_span_km = (region.bbox["lon_max"] - region.bbox["lon_min"]) * max(55.0, 111.0 * abs(np.cos(np.radians(mid_lat))))
    height = max(8, min(24, ceil(lat_span_km / max(region.grid_dy_km, 0.5))))
    width = max(8, min(24, ceil(lon_span_km / max(region.grid_dx_km, 0.5))))
    lat_res = (region.bbox["lat_max"] - region.bbox["lat_min"]) / max(height - 1, 1)
    lon_res = (region.bbox["lon_max"] - region.bbox["lon_min"]) / max(width - 1, 1)
    return GridSpec(
        crs="EPSG:4326",
        width=width,
        height=height,
        bbox=dict(region.bbox),
        resolution_km=(round(region.grid_dx_km, 3), round(region.grid_dy_km, 3)),
        transform=[
            region.bbox["lon_min"],
            lon_res,
            0.0,
            region.bbox["lat_max"],
            0.0,
            -lat_res,
        ],
    )


def cell_position(region: RegionDefinition, grid_spec: GridSpec, *, lat: float, lon: float) -> tuple[int, int]:
    lat_span = max(region.bbox["lat_max"] - region.bbox["lat_min"], 1e-6)
    lon_span = max(region.bbox["lon_max"] - region.bbox["lon_min"], 1e-6)
    row = int(round(((region.bbox["lat_max"] - lat) / lat_span) * (grid_spec.height - 1)))
    col = int(round(((lon - region.bbox["lon_min"]) / lon_span) * (grid_spec.width - 1)))
    return (
        min(max(row, 0), grid_spec.height - 1),
        min(max(col, 0), grid_spec.width - 1),
    )


def tensorize_cell_values(
    region_id: str,
    values_by_cell: dict[str, float],
) -> tuple[GridSpec, np.ndarray, dict[str, tuple[int, int]]]:
    region = get_region_definition(region_id)
    grid_spec = build_region_grid_spec(region_id)
    tensor = np.zeros((grid_spec.height, grid_spec.width), dtype=np.float32)
    cell_map: dict[str, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    for cell in region.cells:
        row, col = cell_position(region, grid_spec, lat=float(cell["lat"]), lon=float(cell["lon"]))
        while (row, col) in occupied and col + 1 < grid_spec.width:
            col += 1
        occupied.add((row, col))
        cell_id = str(cell["cell_id"])
        tensor[row, col] = float(values_by_cell.get(cell_id, 0.0))
        cell_map[cell_id] = (row, col)
    return grid_spec, tensor, cell_map


def static_region_masks(region_id: str) -> tuple[GridSpec, np.ndarray, np.ndarray, np.ndarray]:
    region = get_region_definition(region_id)
    grid_spec = build_region_grid_spec(region_id)
    shoreline = np.zeros((grid_spec.height, grid_spec.width), dtype=np.float32)
    restricted = np.zeros((grid_spec.height, grid_spec.width), dtype=np.float32)
    bathymetry = np.zeros((grid_spec.height, grid_spec.width), dtype=np.float32)
    occupied: set[tuple[int, int]] = set()
    for cell in region.cells:
        row, col = cell_position(region, grid_spec, lat=float(cell["lat"]), lon=float(cell["lon"]))
        while (row, col) in occupied and col + 1 < grid_spec.width:
            col += 1
        occupied.add((row, col))
        shoreline[row, col] = float(cell["shoreline_proximity"])
        restricted[row, col] = 1.0 if bool(cell["restricted"]) else 0.0
        bathymetry[row, col] = max(0.0, 1.0 - float(cell["shoreline_proximity"]))
    return grid_spec, shoreline, restricted, bathymetry

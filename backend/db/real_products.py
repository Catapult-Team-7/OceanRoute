from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from ingest.compute_flux import compute_co2_flux
from ingest.fetch_copernicus import default_copernicus_directory
from ingest.real_training_data import (
    RealDataLoadError,
    _normalize_longitude,
    default_era5_directory,
    load_copernicus_monthly,
    load_era5_monthly,
    load_noaa_gml_monthly,
    load_socat_observations,
)
from ml.preprocess import _grid_values, _resolution_step

from .models import AnomalyRecord, FluxPoint, ForecastPoint, ForecastResponse

if TYPE_CHECKING:
    from ml.trainer import MLTrainerService


REGION_BOUNDS = {
    "global": (-90, 90, -180, 180),
    "pacific": (-60, 65, 110, -70),
    "atlantic": (-60, 70, -80, 20),
    "indian": (-60, 30, 20, 120),
}


def _normalize_directory_key(path_value: str | Path | None, fallback: Path) -> str:
    raw = str(path_value or "").strip()
    path = Path(raw).expanduser() if raw else fallback
    if not path.is_absolute():
        path = fallback.parents[1] / path
    return str(path.resolve())


@dataclass
class RealGridBundle:
    rows: list[FluxPoint]
    metadata: dict
    anomalies: list[AnomalyRecord]


@dataclass
class RealPointBundle:
    forecast: ForecastResponse
    history: list[dict]


def _in_region(lat: float, lon: float, region: str) -> bool:
    min_lat, max_lat, min_lon, max_lon = REGION_BOUNDS.get(region, REGION_BOUNDS["global"])
    if min_lon <= max_lon:
        lon_ok = min_lon <= lon <= max_lon
    else:
        lon_ok = lon >= min_lon or lon <= max_lon
    return min_lat <= lat <= max_lat and lon_ok


def _operational_latitude_weight(lat: float) -> float:
    absolute_lat = abs(float(lat))
    if absolute_lat <= 55:
        return 1.0
    if absolute_lat >= 80:
        return 0.35
    return max(0.35, 1.0 - ((absolute_lat - 55.0) / 25.0) * 0.65)


def _edge_longitude_weight(lon: float, resolution: str) -> float:
    edge_band = _resolution_step(resolution) * 1.5
    distance_to_edge = abs(180.0 - abs(float(lon)))
    if distance_to_edge >= edge_band:
        return 1.0
    return max(0.12, distance_to_edge / max(edge_band, 1e-6))


def _smooth_spatial_grid(values: np.ndarray, support_mask: np.ndarray | None = None, passes: int = 2) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float32).copy()
    support = np.asarray(support_mask, dtype=np.float32) if support_mask is not None else np.ones_like(smoothed, dtype=np.float32)
    kernel = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    kernel /= float(kernel.sum())
    for _ in range(max(1, passes)):
        padded_values = np.pad(smoothed * support, ((1, 1), (1, 1)), mode="edge")
        padded_support = np.pad(support, ((1, 1), (1, 1)), mode="edge")
        numerator = np.zeros_like(smoothed, dtype=np.float32)
        denominator = np.zeros_like(smoothed, dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                weight = kernel[di, dj]
                numerator += padded_values[di : di + smoothed.shape[0], dj : dj + smoothed.shape[1]] * weight
                denominator += padded_support[di : di + smoothed.shape[0], dj : dj + smoothed.shape[1]] * weight
        smoothed = np.divide(numerator, np.maximum(denominator, 1e-6), out=smoothed, where=denominator > 0)
        smoothed = np.where(support > 0, smoothed, values)
    return smoothed


def _normalized_score_grid(values: np.ndarray, support_mask: np.ndarray, floor: float) -> np.ndarray:
    supported = np.asarray(values, dtype=np.float32)[support_mask]
    if supported.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = max(float(np.nanstd(supported)), floor)
    return np.asarray(values, dtype=np.float32) / scale


def _current_convergence_grid(current_u: np.ndarray, current_v: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    du_dy, du_dx = np.gradient(np.asarray(current_u, dtype=np.float32))
    dv_dy, dv_dx = np.gradient(np.asarray(current_v, dtype=np.float32))
    divergence = du_dx + dv_dy
    convergence = np.maximum(0.0, -divergence)
    convergence = _smooth_spatial_grid(convergence, support_mask, passes=2)
    return _normalized_score_grid(convergence, support_mask, floor=0.04)


def _support_density_grid(support_mask: np.ndarray) -> np.ndarray:
    support = np.asarray(support_mask, dtype=np.float32)
    if not support.any():
        return np.zeros_like(support, dtype=np.float32)
    padded = np.pad(support, ((1, 1), (1, 1)), mode="edge")
    density = np.zeros_like(support, dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            density += padded[di : di + support.shape[0], dj : dj + support.shape[1]]
    density /= 9.0
    return np.where(support_mask, density, 0.0)


def _local_peak_mask(
    values: np.ndarray,
    support_mask: np.ndarray,
    *,
    min_value: float,
    min_prominence: float,
    neighborhood: int = 1,
) -> np.ndarray:
    rows, cols = values.shape
    peaks = np.zeros_like(values, dtype=bool)
    for i in range(rows):
        for j in range(cols):
            if not support_mask[i, j]:
                continue
            center = float(values[i, j])
            if center < min_value:
                continue
            i0 = max(0, i - neighborhood)
            i1 = min(rows, i + neighborhood + 1)
            j0 = max(0, j - neighborhood)
            j1 = min(cols, j + neighborhood + 1)
            window_values = values[i0:i1, j0:j1]
            window_support = support_mask[i0:i1, j0:j1]
            supported_window = window_values[window_support]
            if supported_window.size == 0:
                continue
            local_max = float(np.nanmax(supported_window))
            local_mean = float(np.nanmean(supported_window))
            if center >= local_max - 1e-6 and center >= local_mean + min_prominence:
                peaks[i, j] = True
    return peaks


def _select_spaced_anomalies(
    candidates: list[AnomalyRecord],
    *,
    limit: int = 24,
    min_distance_deg: float = 8.0,
) -> list[AnomalyRecord]:
    selected: list[AnomalyRecord] = []
    for candidate in sorted(candidates, key=lambda item: item.anomaly_score, reverse=True):
        if all(math.hypot(candidate.lat - item.lat, candidate.lon - item.lon) >= min_distance_deg for item in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _residual_structure_grids(
    predicted_grid: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed_prediction = _smooth_spatial_grid(predicted_grid, support_mask, passes=2)
    background_grid = _smooth_spatial_grid(smoothed_prediction, support_mask, passes=7)
    residual_grid = np.where(support_mask, smoothed_prediction - background_grid, 0.0)
    gy, gx = np.gradient(residual_grid)
    gradient_grid = np.sqrt((gy**2) + (gx**2))
    return smoothed_prediction, residual_grid, gradient_grid


def _month_from_string(date_str: str | None) -> tuple[int, int] | None:
    if not date_str:
        return None
    year, month = map(int, date_str.split("-"))
    return year, month


def _common_months(atmospheric: pd.DataFrame, era_monthly: pd.DataFrame, copernicus_monthly: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted(
        {(int(row.year), int(row.month)) for row in atmospheric.itertuples(index=False)}
        & {(int(row.year), int(row.month)) for row in era_monthly.itertuples(index=False)}
        & {(int(row.year), int(row.month)) for row in copernicus_monthly.itertuples(index=False)}
    )


def _resolve_target_month(months: list[tuple[int, int]], requested: tuple[int, int] | None) -> tuple[int, int]:
    if not months:
      raise RealDataLoadError("No overlapping Copernicus, ERA5, and NOAA monthly records are available.")
    if requested is None:
        return months[-1]
    eligible = [month for month in months if month <= requested]
    if not eligible:
        raise RealDataLoadError(
            f"Requested month {requested[0]}-{requested[1]:02d} is earlier than the available overlapping driver data."
        )
    return eligible[-1]


def _resolve_latest_publishable_target(
    months: list[tuple[int, int]],
    requested: tuple[int, int] | None,
    trained_month_window: int,
) -> tuple[tuple[int, int], int, int]:
    if not months:
        raise RealDataLoadError("No overlapping Copernicus, ERA5, and NOAA monthly records are available.")

    if requested is None:
        candidates = list(reversed(months))
    else:
        candidates = [month for month in months if month <= requested]
        if not candidates:
            raise RealDataLoadError(
                f"Requested month {requested[0]}-{requested[1]:02d} is earlier than the available overlapping driver data."
            )
        candidates = list(reversed(candidates))

    for candidate in candidates:
        target_index = months.index(candidate)
        try:
            month_window = _resolve_inference_month_window(months, target_index, trained_month_window)
            return candidate, target_index, month_window
        except RealDataLoadError:
            continue

    latest = candidates[0]
    raise RealDataLoadError(
        f"Not enough gridded monthly history for inference at {latest[0]}-{latest[1]:02d}. Need at least 1 prior month."
    )


def _resolve_inference_month_window(
    available_months: list[tuple[int, int]],
    target_index: int,
    trained_month_window: int,
) -> int:
    available_prior = target_index
    if available_prior >= trained_month_window:
        return trained_month_window

    # OceanPulseLSTM accepts variable sequence length, so allow the latest
    # real map to run with whatever prior real history exists, down to a
    # single prior month, instead of blocking the verified map entirely.
    fallback_window = min(trained_month_window, available_prior)
    if fallback_window >= 1:
        return fallback_window

    raise RealDataLoadError(
        f"Not enough gridded monthly history for inference at {available_months[target_index][0]}-"
        f"{available_months[target_index][1]:02d}. Need at least 1 prior month."
    )


def _scalar_month_value(frame: pd.DataFrame, year: int, month: int, column: str, default: float = 0.0) -> float:
    subset = frame[(frame["year"] == year) & (frame["month"] == month)]
    if subset.empty or column not in subset.columns:
        return default
    values = subset[column].dropna()
    if values.empty:
        return default
    return float(values.iloc[0])


def _latest_scalar_value(frame: pd.DataFrame, year: int, month: int, column: str, default: float = 0.0) -> float:
    subset = frame[frame["year"].notna() & frame["month"].notna()].copy()
    if subset.empty or column not in subset.columns:
        return default
    subset["year"] = subset["year"].astype(int)
    subset["month"] = subset["month"].astype(int)
    eligible = subset[(subset["year"] < year) | ((subset["year"] == year) & (subset["month"] <= month))]
    if eligible.empty:
        eligible = subset
    eligible = eligible.sort_values(["year", "month"])
    values = eligible[column].dropna()
    if values.empty:
        return default
    return float(values.iloc[-1])


def _aggregate_copernicus_to_resolution(copernicus_monthly: pd.DataFrame, resolution: str) -> pd.DataFrame:
    step = _resolution_step(resolution)
    aggregated = copernicus_monthly.copy()
    aggregated["lat_bin"] = (pd.to_numeric(aggregated["lat_bin"], errors="coerce") / step).round() * step
    aggregated["lon_bin"] = (_normalize_longitude(aggregated["lon_bin"]) / step).round() * step
    return aggregated.groupby(["year", "month", "lat_bin", "lon_bin"], as_index=False).mean(numeric_only=True)


def _build_feature_grids(
    atmospheric: pd.DataFrame,
    era_monthly: pd.DataFrame,
    copernicus_monthly: pd.DataFrame,
    resolution: str,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float], np.ndarray, np.ndarray]:
    lat_values, lon_values = _grid_values(resolution)
    lat_index = {round(float(lat), 6): idx for idx, lat in enumerate(lat_values)}
    lon_index = {round(float(lon), 6): idx for idx, lon in enumerate(lon_values)}
    months = _common_months(atmospheric, era_monthly, copernicus_monthly)
    monthly_features: dict[tuple[int, int], np.ndarray] = {}
    monthly_atm: dict[tuple[int, int], float] = {}

    for year, month in months:
        feature_grid = np.zeros((13, len(lat_values), len(lon_values)), dtype=np.float32)
        scalar_subset = copernicus_monthly[(copernicus_monthly["year"] == year) & (copernicus_monthly["month"] == month)]
        era_subset = era_monthly[(era_monthly["year"] == year) & (era_monthly["month"] == month)]

        thetao_default = _scalar_month_value(scalar_subset, year, month, "copernicus_thetao", default=0.0)
        salinity_default = _scalar_month_value(scalar_subset, year, month, "copernicus_salinity", default=0.0)
        u_default = _scalar_month_value(scalar_subset, year, month, "copernicus_u", default=0.0)
        v_default = _scalar_month_value(scalar_subset, year, month, "copernicus_v", default=0.0)
        sea_level_default = _scalar_month_value(scalar_subset, year, month, "copernicus_zos", default=0.0)
        wind_default = _scalar_month_value(era_subset, year, month, "era5_wind_speed", default=0.0)
        atm_default = _scalar_month_value(atmospheric, year, month, "pco2_atm", default=0.0)
        angle = 2 * np.pi * month / 12.0

        feature_grid[0, :, :] = thetao_default / 30.0
        feature_grid[1, :, :] = salinity_default / 40.0
        feature_grid[2, :, :] = u_default / 3.0
        feature_grid[3, :, :] = v_default / 3.0
        feature_grid[4, :, :] = np.sqrt((u_default**2) + (v_default**2)) / 4.0
        feature_grid[5, :, :] = wind_default / 20.0
        feature_grid[6, :, :] = sea_level_default / 2.0
        feature_grid[7, :, :] = atm_default / 500.0
        feature_grid[8, :, :] = np.sin(angle)
        feature_grid[9, :, :] = np.cos(angle)
        feature_grid[10, :, :] = lat_values[:, None] / 80.0
        feature_grid[11, :, :] = lon_values[None, :] / 180.0
        feature_grid[12, :, :] = 0.0

        month_rows = scalar_subset
        for row in month_rows.itertuples(index=False):
            lat_key = round(float(row.lat_bin), 6)
            lon_key = round(float(row.lon_bin), 6)
            if lat_key not in lat_index or lon_key not in lon_index:
                continue
            i = lat_index[lat_key]
            j = lon_index[lon_key]
            if hasattr(row, "copernicus_thetao") and not np.isnan(getattr(row, "copernicus_thetao", np.nan)):
                feature_grid[0, i, j] = float(row.copernicus_thetao) / 30.0
            if hasattr(row, "copernicus_salinity") and not np.isnan(getattr(row, "copernicus_salinity", np.nan)):
                feature_grid[1, i, j] = float(row.copernicus_salinity) / 40.0
            if hasattr(row, "copernicus_u") and not np.isnan(getattr(row, "copernicus_u", np.nan)):
                feature_grid[2, i, j] = float(row.copernicus_u) / 3.0
            if hasattr(row, "copernicus_v") and not np.isnan(getattr(row, "copernicus_v", np.nan)):
                feature_grid[3, i, j] = float(row.copernicus_v) / 3.0
            feature_grid[4, i, j] = float(np.sqrt((feature_grid[2, i, j] * 3.0) ** 2 + (feature_grid[3, i, j] * 3.0) ** 2) / 4.0)
            if hasattr(row, "copernicus_zos") and not np.isnan(getattr(row, "copernicus_zos", np.nan)):
                feature_grid[6, i, j] = float(row.copernicus_zos) / 2.0
            feature_grid[12, i, j] = 1.0

        monthly_features[(year, month)] = feature_grid
        monthly_atm[(year, month)] = atm_default

    return months, monthly_features, monthly_atm, lat_values, lon_values


def _build_observed_flux_grid(
    socat_url: str,
    atmospheric: pd.DataFrame,
    target_month: tuple[int, int],
    resolution: str,
    wind_speed: float,
) -> dict[tuple[float, float], float]:
    year, month = target_month
    observations = load_socat_observations(socat_url, sample_size=24000)
    observations = observations[(observations["year"] == year) & (observations["month"] == month)].copy()
    if observations.empty:
        return {}
    step = _resolution_step(resolution)
    observations["lat_bin"] = (pd.to_numeric(observations["lat"], errors="coerce") / step).round() * step
    observations["lon_bin"] = (_normalize_longitude(observations["lon"]) / step).round() * step
    observations = observations.groupby(["lat_bin", "lon_bin"], as_index=False).agg(
        pco2_ocean=("pco2_ocean", "mean"),
        sst=("sst", "mean"),
        salinity=("salinity", "mean"),
    )
    pco2_atm = _scalar_month_value(atmospheric, year, month, "pco2_atm", default=0.0)
    if not pco2_atm:
        return {}
    observations["observed_flux"] = compute_co2_flux(
        observations["pco2_ocean"].to_numpy(dtype=float),
        np.full(len(observations), pco2_atm, dtype=float),
        observations["sst"].fillna(0.0).to_numpy(dtype=float),
        np.full(len(observations), wind_speed, dtype=float),
        observations["salinity"].fillna(35.0).to_numpy(dtype=float),
    )
    return {
        (round(float(row.lat_bin), 6), round(float(row.lon_bin), 6)): float(row.observed_flux)
        for row in observations.itertuples(index=False)
    }


def _nearest_grid_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _support_cells_for_month(copernicus_monthly: pd.DataFrame, target_month: tuple[int, int]) -> set[tuple[float, float]]:
    target_rows = copernicus_monthly[
        (copernicus_monthly["year"] == target_month[0]) & (copernicus_monthly["month"] == target_month[1])
    ]
    return {
        (round(float(row.lat_bin), 6), round(float(row.lon_bin), 6))
        for row in target_rows.itertuples(index=False)
    }


@lru_cache(maxsize=8)
def _load_monthly_drivers_cached(
    noaa_gml_url: str,
    era_directory_key: str,
    copernicus_directory_key: str,
    resolution: str,
):
    atmospheric = load_noaa_gml_monthly(noaa_gml_url)
    era_monthly = load_era5_monthly(era_directory_key)
    copernicus_monthly = load_copernicus_monthly(copernicus_directory_key)
    copernicus_monthly = _aggregate_copernicus_to_resolution(copernicus_monthly, resolution)
    return atmospheric, era_monthly, copernicus_monthly


def _load_monthly_drivers(noaa_gml_url: str, era_directory: str | Path | None, copernicus_directory: str | Path | None, resolution: str):
    era_key = _normalize_directory_key(era_directory, default_era5_directory())
    copernicus_key = _normalize_directory_key(copernicus_directory, default_copernicus_directory())
    return _load_monthly_drivers_cached(noaa_gml_url, era_key, copernicus_key, resolution)


@lru_cache(maxsize=24)
def _build_observed_flux_grid_cached(
    socat_url: str,
    noaa_gml_url: str,
    year: int,
    month: int,
    resolution: str,
    wind_speed: float,
) -> dict[tuple[float, float], float]:
    atmospheric = load_noaa_gml_monthly(noaa_gml_url)
    target_month = (year, month)
    return _build_observed_flux_grid(
        socat_url=socat_url,
        atmospheric=atmospheric,
        target_month=target_month,
        resolution=resolution,
        wind_speed=wind_speed,
    )


def build_provisional_real_grid_bundle(
    *,
    date: str | None,
    resolution: str,
    region: str,
    trainer: "MLTrainerService",
    noaa_gml_url: str,
    era_directory: str | Path | None,
    copernicus_directory: str | Path | None,
    reference_now: datetime,
) -> RealGridBundle:
    model, _ = trainer._load_spatial_model()
    atmospheric, era_monthly, copernicus_monthly = _load_monthly_drivers(
        noaa_gml_url=noaa_gml_url,
        era_directory=era_directory,
        copernicus_directory=copernicus_directory,
        resolution=resolution,
    )
    if copernicus_monthly.empty:
        raise RealDataLoadError("No Copernicus monthly grids are available for provisional checkpoint-backed inference.")

    available_months = sorted({(int(row.year), int(row.month)) for row in copernicus_monthly.itertuples(index=False)})
    requested = _month_from_string(date)
    target_month = _resolve_target_month(available_months, requested)
    lat_values, lon_values = _grid_values(resolution)
    lat_index = {round(float(lat), 6): idx for idx, lat in enumerate(lat_values)}
    lon_index = {round(float(lon), 6): idx for idx, lon in enumerate(lon_values)}
    target_rows = copernicus_monthly[
        (copernicus_monthly["year"] == target_month[0]) & (copernicus_monthly["month"] == target_month[1])
    ]
    if target_rows.empty:
        raise RealDataLoadError(
            f"No Copernicus monthly grid rows are available for provisional inference at {target_month[0]}-{target_month[1]:02d}."
        )

    thetao_default = _latest_scalar_value(copernicus_monthly, target_month[0], target_month[1], "copernicus_thetao", default=0.0)
    salinity_default = _latest_scalar_value(copernicus_monthly, target_month[0], target_month[1], "copernicus_salinity", default=0.0)
    u_default = _latest_scalar_value(copernicus_monthly, target_month[0], target_month[1], "copernicus_u", default=0.0)
    v_default = _latest_scalar_value(copernicus_monthly, target_month[0], target_month[1], "copernicus_v", default=0.0)
    sea_level_default = _latest_scalar_value(copernicus_monthly, target_month[0], target_month[1], "copernicus_zos", default=0.0)
    wind_default = _latest_scalar_value(era_monthly, target_month[0], target_month[1], "era5_wind_speed", default=0.0)
    atm_default = _latest_scalar_value(atmospheric, target_month[0], target_month[1], "pco2_atm", default=0.0)
    angle = 2 * np.pi * target_month[1] / 12.0

    feature_grid = np.zeros((13, len(lat_values), len(lon_values)), dtype=np.float32)
    feature_grid[0, :, :] = thetao_default / 30.0
    feature_grid[1, :, :] = salinity_default / 40.0
    feature_grid[2, :, :] = u_default / 3.0
    feature_grid[3, :, :] = v_default / 3.0
    feature_grid[4, :, :] = np.sqrt((u_default**2) + (v_default**2)) / 4.0
    feature_grid[5, :, :] = wind_default / 20.0
    feature_grid[6, :, :] = sea_level_default / 2.0
    feature_grid[7, :, :] = atm_default / 500.0
    feature_grid[8, :, :] = np.sin(angle)
    feature_grid[9, :, :] = np.cos(angle)
    feature_grid[10, :, :] = lat_values[:, None] / 80.0
    feature_grid[11, :, :] = lon_values[None, :] / 180.0
    feature_grid[12, :, :] = 0.0

    for row in target_rows.itertuples(index=False):
        lat_key = round(float(row.lat_bin), 6)
        lon_key = round(float(row.lon_bin), 6)
        if lat_key not in lat_index or lon_key not in lon_index:
            continue
        i = lat_index[lat_key]
        j = lon_index[lon_key]
        if hasattr(row, "copernicus_thetao") and not np.isnan(getattr(row, "copernicus_thetao", np.nan)):
            feature_grid[0, i, j] = float(row.copernicus_thetao) / 30.0
        if hasattr(row, "copernicus_salinity") and not np.isnan(getattr(row, "copernicus_salinity", np.nan)):
            feature_grid[1, i, j] = float(row.copernicus_salinity) / 40.0
        if hasattr(row, "copernicus_u") and not np.isnan(getattr(row, "copernicus_u", np.nan)):
            feature_grid[2, i, j] = float(row.copernicus_u) / 3.0
        if hasattr(row, "copernicus_v") and not np.isnan(getattr(row, "copernicus_v", np.nan)):
            feature_grid[3, i, j] = float(row.copernicus_v) / 3.0
        feature_grid[4, i, j] = float(np.sqrt((feature_grid[2, i, j] * 3.0) ** 2 + (feature_grid[3, i, j] * 3.0) ** 2) / 4.0)
        if hasattr(row, "copernicus_zos") and not np.isnan(getattr(row, "copernicus_zos", np.nan)):
            feature_grid[6, i, j] = float(row.copernicus_zos) / 2.0
        feature_grid[12, i, j] = 1.0

    with torch.no_grad():
        x_tensor = torch.tensor(feature_grid[None, ...], dtype=torch.float32)
        atm_tensor = torch.tensor([[atm_default]], dtype=torch.float32)
        predicted_grid = model(x_tensor, atm_tensor)[0, 0].detach().cpu().numpy()

    support_cells = _support_cells_for_month(copernicus_monthly, target_month)
    support_mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)
    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            support_mask[i, j] = (round(float(lat), 6), round(float(lon), 6)) in support_cells
    if not support_mask.any():
        raise RealDataLoadError("No supported Copernicus ocean cells were available for provisional inference.")

    predicted_grid, residual_grid, gradient_grid = _residual_structure_grids(predicted_grid, support_mask)
    supported_residuals = residual_grid[support_mask]
    supported_gradients = gradient_grid[support_mask]
    residual_scale = max(float(np.nanstd(supported_residuals)), 0.12)
    gradient_scale = max(float(np.nanstd(supported_gradients)), 0.08)
    current_u_grid = feature_grid[2] * 3.0
    current_v_grid = feature_grid[3] * 3.0
    current_speed_grid = np.sqrt((current_u_grid**2) + (current_v_grid**2))
    convergence_grid = _current_convergence_grid(current_u_grid, current_v_grid, support_mask)
    support_density_grid = _support_density_grid(support_mask)
    deviation_grid = np.abs(residual_grid) / residual_scale
    gradient_norm_grid = gradient_grid / gradient_scale
    route_signal_grid = (
        deviation_grid * 0.35
        + gradient_norm_grid * 0.28
        + convergence_grid * 0.9
        + np.clip(current_speed_grid / 2.5, 0.0, 2.0) * 0.18
    )
    anomaly_signal_grid = (
        deviation_grid * 0.22
        + gradient_norm_grid * 0.26
        + convergence_grid * 0.42
        + np.clip(current_speed_grid / 3.0, 0.0, 1.5) * 0.08
    )
    route_peak_mask = _local_peak_mask(
        route_signal_grid,
        support_mask,
        min_value=0.95 if region == "global" else 0.7,
        min_prominence=0.16 if region == "global" else 0.12,
        neighborhood=1 if region == "global" else 2,
    )
    anomaly_peak_mask = _local_peak_mask(
        anomaly_signal_grid,
        support_mask,
        min_value=0.85 if region == "global" else 0.65,
        min_prominence=0.14 if region == "global" else 0.1,
        neighborhood=1 if region == "global" else 2,
    )
    rows: list[FluxPoint] = []
    anomaly_candidates: list[AnomalyRecord] = []
    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            if not _in_region(float(lat), float(lon), region):
                continue
            if not support_mask[i, j]:
                continue
            predicted_flux = float(predicted_grid[i, j])
            model_deviation = float(deviation_grid[i, j])
            gradient_strength = float(gradient_norm_grid[i, j])
            current_u = float(current_u_grid[i, j])
            current_v = float(current_v_grid[i, j])
            current_speed = float(current_speed_grid[i, j])
            convergence_score = float(convergence_grid[i, j])
            latitude_weight = _operational_latitude_weight(float(lat))
            edge_weight = _edge_longitude_weight(float(lon), resolution)
            coastal_weight = float(np.clip((support_density_grid[i, j] - 0.42) / 0.52, 0.0, 1.0))
            operational_weight = latitude_weight * edge_weight * max(0.15, coastal_weight)
            display_flux = float(np.clip(residual_grid[i, j], -2.5, 2.5))
            display_signal = float(
                max(
                    0.0,
                    (
                        model_deviation * 0.65
                        + gradient_strength * 0.55
                        + convergence_score * 0.45
                    )
                    * operational_weight,
                )
            )
            weakening = max(
                0.0,
                ((model_deviation * 0.16) + (gradient_strength * 0.18) + (convergence_score * 0.42))
                * operational_weight,
            )
            route_priority = max(
                0.0,
                (
                    model_deviation * 0.24
                    + gradient_strength * 0.18
                    + convergence_score * 0.85
                    + current_speed * 0.14
                    + (wind_default / 36.0)
                )
                * operational_weight,
            )
            if not route_peak_mask[i, j]:
                route_priority *= 0.16
            anomaly_score = min(
                1.0,
                (
                    model_deviation * 0.12
                    + gradient_strength * 0.15
                    + convergence_score * 0.62
                    + current_speed * 0.02
                )
                * operational_weight,
            )
            if not anomaly_peak_mask[i, j]:
                anomaly_score *= 0.2
            deviation_pct = round((model_deviation * 35.0 + gradient_strength * 45.0) * operational_weight, 1)
            rows.append(
                FluxPoint(
                    lat=float(lat),
                    lon=float(lon),
                    co2_flux=round(predicted_flux, 4),
                    sst=round(float(feature_grid[0, i, j] * 30.0), 2),
                    salinity=round(float(feature_grid[1, i, j] * 40.0), 2),
                    wind_speed=round(float(feature_grid[5, i, j] * 20.0), 2),
                    current_u=round(current_u, 4),
                    current_v=round(current_v, 4),
                    sea_level=round(float(feature_grid[6, i, j] * 2.0), 4),
                    chl_a=0.0,
                    anomaly_score=round(anomaly_score, 4),
                    observed_flux=round(predicted_flux, 4),
                    predicted_flux=round(predicted_flux, 4),
                    display_flux=round(display_flux, 4),
                    display_signal=round(display_signal, 4),
                    weakening_score=round(weakening, 4),
                    route_priority=round(min(route_priority, 5.0), 4),
                    timestamp=datetime(target_month[0], target_month[1], 1, tzinfo=reference_now.tzinfo),
                    source="MODEL",
                )
            )
            if abs(float(lat)) <= 52 and anomaly_peak_mask[i, j] and anomaly_score >= 0.24 and deviation_pct >= 7.0:
                anomaly_candidates.append(
                    AnomalyRecord(
                        id=f"provisional-{target_month[0]}-{target_month[1]}-{i}-{j}",
                        lat=float(lat),
                        lon=float(lon),
                        region_name=f"Cell {float(lat):.1f}, {float(lon):.1f}",
                        anomaly_score=round(anomaly_score, 4),
                        deviation_pct=deviation_pct,
                        detected_at=datetime(target_month[0], target_month[1], 1, tzinfo=reference_now.tzinfo),
                        severity="critical" if anomaly_score >= 0.8 else "high" if anomaly_score >= 0.55 else "medium",
                    )
                )

    anomalies = _select_spaced_anomalies(anomaly_candidates, limit=24, min_distance_deg=8.0 if region == "global" else 4.0)

    mean_flux = sum(row.co2_flux for row in rows) / max(len(rows), 1)
    sink_area_pct = sum(1 for row in rows if row.co2_flux < 0) / max(len(rows), 1) * 100
    metadata = {
        "date": f"{target_month[0]}-{target_month[1]:02d}",
        "units": "mol CO2/m²/yr",
        "mean_flux": round(mean_flux, 3),
        "sink_area_pct": round(sink_area_pct, 1),
        "inference_mode": "checkpoint_backed_provisional_inference",
        "trained_model_ready": True,
        "verified_map": False,
        "map_source": "checkpoint_provisional_grid",
        "source_summary": (
            "Checkpoint-backed provisional grid built from the latest available Copernicus monthly fields with NOAA and ERA5 scalar forcing fallback. "
            "These outputs are real model predictions, but not yet a fully verified published grid."
        ),
        "trained_month_window": int(trainer.expected_month_window()),
        "effective_inference_month_window": 1,
        "observed_support_cells": 0,
        "supported_ocean_cells": int(support_mask.sum()),
    }
    return RealGridBundle(rows=rows, metadata=metadata, anomalies=anomalies)


def build_real_grid_bundle(
    *,
    date: str | None,
    resolution: str,
    region: str,
    trainer: "MLTrainerService",
    socat_url: str,
    noaa_gml_url: str,
    era_directory: str | Path | None,
    copernicus_directory: str | Path | None,
    reference_now: datetime,
) -> RealGridBundle:
    try:
        trainer._load_spatial_model()
    except RealDataLoadError:
        raise
    except Exception as exc:
        raise RealDataLoadError(f"A trained checkpoint is required before the real gridded map can be served: {exc}") from exc

    atmospheric, era_monthly, copernicus_monthly = _load_monthly_drivers(
        noaa_gml_url=noaa_gml_url,
        era_directory=era_directory,
        copernicus_directory=copernicus_directory,
        resolution=resolution,
    )
    months, monthly_features, monthly_atm, lat_values, lon_values = _build_feature_grids(
        atmospheric, era_monthly, copernicus_monthly, resolution
    )
    trained_month_window = int(trainer.expected_month_window())
    target_month, target_index, month_window = _resolve_latest_publishable_target(
        months,
        _month_from_string(date),
        trained_month_window,
    )

    sequence_months = months[target_index - month_window : target_index]
    sequence = np.stack([monthly_features[item] for item in sequence_months], axis=0)
    predicted_grid = trainer.predict_spatial_sequence(sequence, monthly_atm[target_month])
    support_cells = _support_cells_for_month(copernicus_monthly, target_month)
    support_mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)
    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            support_mask[i, j] = (round(float(lat), 6), round(float(lon), 6)) in support_cells
    if not support_mask.any():
        raise RealDataLoadError(f"No supported Copernicus ocean cells were available for {target_month[0]}-{target_month[1]:02d}.")
    era_wind = _scalar_month_value(era_monthly, target_month[0], target_month[1], "era5_wind_speed", default=0.0)
    observed_flux = _build_observed_flux_grid_cached(
        socat_url,
        noaa_gml_url,
        target_month[0],
        target_month[1],
        resolution,
        round(float(era_wind), 4),
    )
    if observed_flux:
        observed_pairs = [
            (predicted_grid[i, j], observed_flux[(round(float(lat_values[i]), 6), round(float(lon_values[j]), 6))])
            for i in range(len(lat_values))
            for j in range(len(lon_values))
            if support_mask[i, j]
            and (round(float(lat_values[i]), 6), round(float(lon_values[j]), 6)) in observed_flux
        ]
        if len(observed_pairs) >= 8:
            predicted_obs = np.array([item[0] for item in observed_pairs], dtype=np.float32)
            observed_obs = np.array([item[1] for item in observed_pairs], dtype=np.float32)
            pred_center = float(np.median(predicted_obs))
            obs_center = float(np.median(observed_obs))
            pred_spread = max(float(np.std(predicted_obs)), 1e-3)
            obs_spread = max(float(np.std(observed_obs)), 1e-3)
            spread_scale = float(np.clip(obs_spread / pred_spread, 0.7, 1.4))
            predicted_grid = ((predicted_grid - pred_center) * spread_scale) + obs_center
    predicted_grid, residual_grid, gradient_grid = _residual_structure_grids(predicted_grid, support_mask)
    supported_residuals = residual_grid[support_mask]
    supported_gradients = gradient_grid[support_mask]
    residual_scale = max(float(np.nanstd(supported_residuals)), 0.12)
    gradient_scale = max(float(np.nanstd(supported_gradients)), 0.08)

    target_features = monthly_features[target_month]
    current_u_grid = target_features[2] * 3.0
    current_v_grid = target_features[3] * 3.0
    current_speed_grid = np.sqrt((current_u_grid**2) + (current_v_grid**2))
    convergence_grid = _current_convergence_grid(current_u_grid, current_v_grid, support_mask)
    support_density_grid = _support_density_grid(support_mask)
    deviation_grid = np.abs(residual_grid) / residual_scale
    gradient_norm_grid = gradient_grid / gradient_scale
    observed_gap_grid = np.zeros_like(predicted_grid, dtype=np.float32)
    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            if not support_mask[i, j]:
                continue
            cell_key = (round(float(lat), 6), round(float(lon), 6))
            observed_value = observed_flux.get(cell_key)
            if observed_value is not None:
                observed_gap_grid[i, j] = abs(float(observed_value) - float(predicted_grid[i, j]))
    route_signal_grid = (
        observed_gap_grid * 0.65
        + deviation_grid * 0.18
        + gradient_norm_grid * 0.16
        + convergence_grid * 0.72
        + np.clip(current_speed_grid / 2.5, 0.0, 2.0) * 0.12
    )
    anomaly_signal_grid = (
        observed_gap_grid * 0.45
        + deviation_grid * 0.12
        + gradient_norm_grid * 0.14
        + convergence_grid * 0.58
        + np.clip(current_speed_grid / 3.0, 0.0, 1.5) * 0.05
    )
    route_peak_mask = _local_peak_mask(
        route_signal_grid,
        support_mask,
        min_value=0.8 if region == "global" else 0.65,
        min_prominence=0.12 if region == "global" else 0.1,
        neighborhood=1 if region == "global" else 2,
    )
    anomaly_peak_mask = _local_peak_mask(
        anomaly_signal_grid,
        support_mask,
        min_value=0.55 if region == "global" else 0.45,
        min_prominence=0.1 if region == "global" else 0.08,
        neighborhood=1 if region == "global" else 2,
    )

    rows: list[FluxPoint] = []
    anomaly_candidates: list[AnomalyRecord] = []
    observed_support_cells = 0
    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            if not _in_region(float(lat), float(lon), region):
                continue
            if not support_mask[i, j]:
                continue
            predicted_flux = float(predicted_grid[i, j])
            cell_key = (round(float(lat), 6), round(float(lon), 6))
            observed_value = observed_flux.get(cell_key, predicted_flux)
            if cell_key in observed_flux:
                observed_support_cells += 1
            observed_gap = float(observed_gap_grid[i, j])
            model_deviation = float(deviation_grid[i, j])
            gradient_strength = float(gradient_norm_grid[i, j])
            latitude_weight = _operational_latitude_weight(float(lat))
            edge_weight = _edge_longitude_weight(float(lon), resolution)
            coastal_weight = float(np.clip((support_density_grid[i, j] - 0.42) / 0.52, 0.0, 1.0))
            operational_weight = latitude_weight * edge_weight * max(0.15, coastal_weight)
            weakening = max(observed_gap, model_deviation * 0.35) * operational_weight
            current_u = float(current_u_grid[i, j])
            current_v = float(current_v_grid[i, j])
            current_speed = float(current_speed_grid[i, j])
            convergence_score = float(convergence_grid[i, j])
            route_priority = max(
                0.0,
                (
                    observed_gap * 0.5
                    + model_deviation * 0.12
                    + gradient_strength * 0.1
                    + convergence_score * 0.7
                    + current_speed * 0.1
                    + (era_wind / 40.0)
                )
                * operational_weight,
            )
            if not route_peak_mask[i, j]:
                route_priority *= 0.14
            anomaly_score = min(
                1.0,
                (
                    observed_gap * 0.28
                    + model_deviation * 0.08
                    + gradient_strength * 0.1
                    + convergence_score * 0.46
                    + current_speed * 0.02
                )
                * operational_weight,
            )
            if not anomaly_peak_mask[i, j]:
                anomaly_score *= 0.2
            display_flux = float(np.clip(residual_grid[i, j], -2.5, 2.5))
            display_signal = float(
                max(
                    0.0,
                    (
                        observed_gap * 0.85
                        + model_deviation * 0.45
                        + gradient_strength * 0.42
                        + convergence_score * 0.35
                    )
                    * operational_weight,
                )
            )
            deviation_pct = round(
                (
                    (observed_gap / max(abs(predicted_flux), 0.25)) * 100.0
                    + model_deviation * 28.0
                    + gradient_strength * 30.0
                )
                * operational_weight,
                1,
            )
            rows.append(
                FluxPoint(
                    lat=float(lat),
                    lon=float(lon),
                    co2_flux=round(predicted_flux, 4),
                    sst=round(float(target_features[0, i, j] * 30.0), 2),
                    salinity=round(float(target_features[1, i, j] * 40.0), 2),
                    wind_speed=round(float(target_features[5, i, j] * 20.0), 2),
                    current_u=round(current_u, 4),
                    current_v=round(current_v, 4),
                    sea_level=round(float(target_features[6, i, j] * 2.0), 4),
                    chl_a=0.0,
                    anomaly_score=round(anomaly_score, 4),
                    observed_flux=round(observed_value, 4),
                    predicted_flux=round(predicted_flux, 4),
                    display_flux=round(display_flux, 4),
                    display_signal=round(display_signal, 4),
                    weakening_score=round(weakening, 4),
                    route_priority=round(min(route_priority, 5.0), 4),
                    timestamp=datetime(target_month[0], target_month[1], 1, tzinfo=reference_now.tzinfo),
                    source="MODEL",
                )
            )
            if abs(float(lat)) <= 52 and anomaly_peak_mask[i, j] and anomaly_score >= 0.24 and deviation_pct >= 7.0:
                anomaly_candidates.append(
                    AnomalyRecord(
                        id=f"real-{target_month[0]}-{target_month[1]}-{i}-{j}",
                        lat=float(lat),
                        lon=float(lon),
                        region_name=f"Cell {float(lat):.1f}, {float(lon):.1f}",
                        anomaly_score=round(anomaly_score, 4),
                        deviation_pct=deviation_pct,
                        detected_at=datetime(target_month[0], target_month[1], 1, tzinfo=reference_now.tzinfo),
                        severity="critical" if anomaly_score >= 0.8 else "high" if anomaly_score >= 0.55 else "medium",
                    )
                )

    anomalies = _select_spaced_anomalies(anomaly_candidates, limit=24, min_distance_deg=8.0 if region == "global" else 4.0)

    mean_flux = sum(row.co2_flux for row in rows) / max(len(rows), 1)
    sink_area_pct = sum(1 for row in rows if row.co2_flux < 0) / max(len(rows), 1) * 100
    metadata = {
        "date": f"{target_month[0]}-{target_month[1]:02d}",
        "units": "mol CO2/m²/yr",
        "mean_flux": round(mean_flux, 3),
        "sink_area_pct": round(sink_area_pct, 1),
        "inference_mode": "checkpoint_backed_gridded_inference",
        "trained_model_ready": True,
        "verified_map": True,
        "map_source": "copernicus_era5_checkpoint_grid",
        "source_summary": (
            "Map grid built from Copernicus monthly physics, ERA5 monthly wind forcing, NOAA atmospheric CO2, "
            "and a trained checkpoint. Sparse SOCAT observations are used only for observed support and anomaly anchoring."
        ),
        "trained_month_window": trained_month_window,
        "effective_inference_month_window": month_window,
        "observed_support_cells": observed_support_cells,
        "supported_ocean_cells": int(support_mask.sum()),
    }
    return RealGridBundle(rows=rows, metadata=metadata, anomalies=anomalies)


def build_real_point_bundle(
    *,
    lat: float,
    lon: float,
    horizon_hours: int,
    trainer: "MLTrainerService",
    noaa_gml_url: str,
    era_directory: str | Path | None,
    copernicus_directory: str | Path | None,
) -> RealPointBundle:
    try:
        trainer._load_spatial_model()
    except RealDataLoadError:
        raise
    except Exception as exc:
        raise RealDataLoadError(
            f"A trained checkpoint is required before point inference can use real gridded drivers: {exc}"
        ) from exc

    resolution = "2deg"
    atmospheric, era_monthly, copernicus_monthly = _load_monthly_drivers(
        noaa_gml_url=noaa_gml_url,
        era_directory=era_directory,
        copernicus_directory=copernicus_directory,
        resolution=resolution,
    )
    months, monthly_features, monthly_atm, lat_values, lon_values = _build_feature_grids(
        atmospheric, era_monthly, copernicus_monthly, resolution
    )
    trained_month_window = int(trainer.expected_month_window())
    if len(months) <= 1:
        raise RealDataLoadError("Not enough monthly driver history for checkpoint-backed point inference.")

    month_window = min(trained_month_window, len(months) - 1)
    if len(months) <= month_window:
        raise RealDataLoadError("Not enough monthly driver history for checkpoint-backed point inference.")

    lat_idx = _nearest_grid_index(lat_values, lat)
    lon_idx = _nearest_grid_index(lon_values, lon)
    monthly_point_rows: list[dict] = []
    for target_index in range(month_window, len(months)):
        target_month = months[target_index]
        sequence_months = months[target_index - month_window : target_index]
        sequence = np.stack([monthly_features[item] for item in sequence_months], axis=0)
        predicted_grid = trainer.predict_spatial_sequence(sequence, monthly_atm[target_month])
        feature_grid = monthly_features[target_month]
        monthly_point_rows.append(
            {
                "date": f"{target_month[0]}-{target_month[1]:02d}",
                "flux": float(predicted_grid[lat_idx, lon_idx]),
                "current_u": float(feature_grid[2, lat_idx, lon_idx] * 3.0),
                "current_v": float(feature_grid[3, lat_idx, lon_idx] * 3.0),
                "sea_level": float(feature_grid[6, lat_idx, lon_idx] * 2.0),
                "wind_speed": float(feature_grid[5, lat_idx, lon_idx] * 20.0),
            }
        )

    if not monthly_point_rows:
        raise RealDataLoadError("No checkpoint-backed point series could be generated from the available monthly grids.")

    recent_history = monthly_point_rows[-12:]
    current = recent_history[-1]
    local_drift = (
        0.04 * current["current_u"] + 0.04 * current["current_v"] + 0.02 * current["sea_level"] + 0.01 * current["wind_speed"]
    )
    validation_loss = float(trainer.state.metrics.get("val_loss", 0.25) or 0.25)

    forecast_items: list[ForecastPoint] = []
    for hours in [24, 48, 72]:
        if hours > max(horizon_hours, 24):
            continue
        drift_scale = hours / 72.0
        forecast_flux = current["flux"] + local_drift * drift_scale
        spread = max(0.18, 0.25 + validation_loss * 0.35 + drift_scale * 0.12)
        forecast_items.append(
            ForecastPoint(
                hours_ahead=hours,
                flux=round(forecast_flux, 3),
                confidence_low=round(forecast_flux - spread, 3),
                confidence_high=round(forecast_flux + spread, 3),
            )
        )

    history = [{"date": item["date"], "flux": round(item["flux"], 3)} for item in recent_history]
    return RealPointBundle(
        forecast=ForecastResponse(lat=lat, lon=lon, current_flux=round(current["flux"], 3), forecast=forecast_items),
        history=history,
    )

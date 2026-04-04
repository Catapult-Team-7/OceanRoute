from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


PROCESSED_TENSOR_DIR = Path(__file__).resolve().parent / "processed"


@dataclass
class TensorBuildResult:
    tensor_dir: Path
    x_path: Path
    y_path: Path
    mask_path: Path
    atm_path: Path
    metadata_path: Path
    summary: dict[str, Any]
    feature_names: list[str]


def _resolution_step(resolution: str) -> float:
    return {
        "2deg": 2.0,
        "1deg": 1.0,
        "0.5deg": 0.5,
    }.get(resolution, 2.0)


def _grid_values(resolution: str) -> tuple[np.ndarray, np.ndarray]:
    step = _resolution_step(resolution)
    lat_values = np.arange(-80.0, 80.0 + step, step, dtype=np.float32)
    lon_values = np.arange(-180.0, 180.0 + step, step, dtype=np.float32)
    return lat_values, lon_values


def _bin_series(series: pd.Series, step: float, *, longitude: bool = False) -> pd.Series:
    values = _normalize_longitude(series) if longitude else pd.to_numeric(series, errors="coerce")
    return (values / step).round() * step


def _month_key(frame: pd.DataFrame) -> pd.Series:
    return frame["year"].astype(int).astype(str) + "-" + frame["month"].astype(int).astype(str).str.zfill(2)


def _scalar_value(frame: pd.DataFrame, year: int, month: int, column: str, default: float = 0.0) -> float:
    subset = frame[(frame["year"] == year) & (frame["month"] == month)]
    if subset.empty or column not in subset.columns:
        return default
    value = subset[column].dropna()
    if value.empty:
        return default
    return float(value.iloc[0])


def _prepare_target_frame(
    socat_url: str,
    noaa_gml_url: str,
    era_directory: str | Path | None,
    copernicus_directory: str | Path | None,
    resolution: str,
    sample_size: int = 20000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atmospheric = load_noaa_gml_monthly(noaa_gml_url)
    socat = load_socat_observations(socat_url, sample_size=sample_size)
    merged = socat.merge(atmospheric[["year", "month", "pco2_atm"]], on=["year", "month"], how="inner")
    if merged.empty:
        raise RealDataLoadError("SOCAT and NOAA GML have no overlapping month records for tensor preprocessing.")

    era_monthly = load_era5_monthly(era_directory or default_era5_directory())
    merged = merged.merge(era_monthly, on=["year", "month"], how="left")
    if "era5_wind_speed" not in merged.columns or merged["era5_wind_speed"].notna().sum() == 0:
        raise RealDataLoadError("ERA5 monthly wind data is required to construct real flux targets.")

    copernicus_monthly = load_copernicus_monthly(copernicus_directory or default_copernicus_directory())
    step = _resolution_step(resolution)
    merged["lat_bin"] = _bin_series(merged["lat"], step)
    merged["lon_bin"] = _bin_series(merged["lon"], step, longitude=True)
    cop = copernicus_monthly.copy()
    cop["lat_bin"] = _bin_series(cop["lat_bin"], step)
    cop["lon_bin"] = _bin_series(cop["lon_bin"], step, longitude=True)
    cop = cop.groupby(["year", "month", "lat_bin", "lon_bin"], as_index=False).mean(numeric_only=True)
    merged = merged.merge(cop, on=["year", "month", "lat_bin", "lon_bin"], how="left")

    merged["sst"] = merged["sst"].fillna(merged.get("copernicus_thetao", pd.Series(index=merged.index, dtype=float)))
    merged["sst"] = merged["sst"].fillna(merged.get("era5_sst_c", pd.Series(index=merged.index, dtype=float)))
    merged["salinity"] = merged["salinity"].fillna(
        merged.get("copernicus_salinity", pd.Series(index=merged.index, dtype=float))
    )
    merged["wind_speed"] = merged["era5_wind_speed"]
    merged = merged.dropna(subset=["sst", "salinity", "wind_speed", "pco2_ocean", "pco2_atm"])
    merged["target_flux"] = compute_co2_flux(
        merged["pco2_ocean"].to_numpy(dtype=float),
        merged["pco2_atm"].to_numpy(dtype=float),
        merged["sst"].to_numpy(dtype=float),
        merged["wind_speed"].to_numpy(dtype=float),
        merged["salinity"].to_numpy(dtype=float),
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_flux"])
    if merged.empty:
        raise RealDataLoadError("No usable target flux rows were produced for monthly tensor building.")

    target_grid = (
        merged.groupby(["year", "month", "lat_bin", "lon_bin"], as_index=False)
        .agg(
            target_flux=("target_flux", "mean"),
            observed_count=("target_flux", "size"),
            sst_obs=("sst", "mean"),
            salinity_obs=("salinity", "mean"),
        )
        .sort_values(["year", "month", "lat_bin", "lon_bin"])
    )
    return target_grid, atmospheric, era_monthly.merge(
        cop.groupby(["year", "month"], as_index=False).mean(numeric_only=True),
        on=["year", "month"],
        how="left",
    )


def build_monthly_training_tensors(
    *,
    socat_url: str,
    noaa_gml_url: str,
    era_directory: str | Path | None,
    copernicus_directory: str | Path | None,
    month_window: int,
    resolution: str,
    reference_now: datetime,
) -> TensorBuildResult:
    target_grid, atmospheric, monthly_scalar_features = _prepare_target_frame(
        socat_url=socat_url,
        noaa_gml_url=noaa_gml_url,
        era_directory=era_directory,
        copernicus_directory=copernicus_directory,
        resolution=resolution,
    )
    lat_values, lon_values = _grid_values(resolution)
    lat_index = {round(float(lat), 6): idx for idx, lat in enumerate(lat_values)}
    lon_index = {round(float(lon), 6): idx for idx, lon in enumerate(lon_values)}

    months = sorted(
        {(int(row.year), int(row.month)) for row in target_grid.itertuples(index=False)}
        & {(int(row.year), int(row.month)) for row in atmospheric.itertuples(index=False)}
    )
    if len(months) <= month_window:
        raise RealDataLoadError(
            f"Not enough monthly records for sequence training. Need > {month_window} overlapping months, found {len(months)}."
        )

    feature_names = [
        "thetao",
        "salinity",
        "current_u",
        "current_v",
        "wind_speed",
        "sea_level",
        "pco2_atm",
        "month_sin",
        "month_cos",
    ]
    monthly_features: dict[tuple[int, int], np.ndarray] = {}
    monthly_targets: dict[tuple[int, int], np.ndarray] = {}
    monthly_masks: dict[tuple[int, int], np.ndarray] = {}
    monthly_atm: dict[tuple[int, int], float] = {}

    for year, month in months:
        feature_grid = np.zeros((len(feature_names), len(lat_values), len(lon_values)), dtype=np.float32)
        target_values = np.full((1, len(lat_values), len(lon_values)), np.nan, dtype=np.float32)
        target_mask = np.zeros((1, len(lat_values), len(lon_values)), dtype=np.float32)

        scalar_subset = monthly_scalar_features[
            (monthly_scalar_features["year"] == year) & (monthly_scalar_features["month"] == month)
        ]
        thetao_default = _scalar_value(scalar_subset, year, month, "copernicus_thetao", default=0.0)
        salinity_default = _scalar_value(scalar_subset, year, month, "copernicus_salinity", default=0.0)
        u_default = _scalar_value(scalar_subset, year, month, "copernicus_u", default=0.0)
        v_default = _scalar_value(scalar_subset, year, month, "copernicus_v", default=0.0)
        sea_level_default = _scalar_value(scalar_subset, year, month, "copernicus_zos", default=0.0)
        wind_default = _scalar_value(scalar_subset, year, month, "era5_wind_speed", default=0.0)
        atm_default = _scalar_value(atmospheric, year, month, "pco2_atm", default=0.0)
        angle = 2 * np.pi * month / 12.0

        feature_grid[0, :, :] = thetao_default / 30.0
        feature_grid[1, :, :] = salinity_default / 40.0
        feature_grid[2, :, :] = u_default / 3.0
        feature_grid[3, :, :] = v_default / 3.0
        feature_grid[4, :, :] = wind_default / 20.0
        feature_grid[5, :, :] = sea_level_default / 2.0
        feature_grid[6, :, :] = atm_default / 500.0
        feature_grid[7, :, :] = np.sin(angle)
        feature_grid[8, :, :] = np.cos(angle)

        month_rows = target_grid[(target_grid["year"] == year) & (target_grid["month"] == month)]
        for row in month_rows.itertuples(index=False):
            lat_key = round(float(row.lat_bin), 6)
            lon_key = round(float(row.lon_bin), 6)
            if lat_key not in lat_index or lon_key not in lon_index:
                continue
            i = lat_index[lat_key]
            j = lon_index[lon_key]
            target_values[0, i, j] = float(row.target_flux)
            target_mask[0, i, j] = 1.0
            if not np.isnan(getattr(row, "sst_obs", np.nan)):
                feature_grid[0, i, j] = float(row.sst_obs) / 30.0
            if not np.isnan(getattr(row, "salinity_obs", np.nan)):
                feature_grid[1, i, j] = float(row.salinity_obs) / 40.0

        monthly_features[(year, month)] = feature_grid
        monthly_targets[(year, month)] = np.nan_to_num(target_values, nan=0.0)
        monthly_masks[(year, month)] = target_mask
        monthly_atm[(year, month)] = atm_default

    x_samples = []
    y_samples = []
    mask_samples = []
    atm_samples = []
    sample_months = []
    for end_index in range(month_window, len(months)):
        input_months = months[end_index - month_window : end_index]
        target_month = months[end_index]
        if monthly_masks[target_month].sum() <= 0:
            continue
        x_samples.append(np.stack([monthly_features[item] for item in input_months], axis=0))
        y_samples.append(monthly_targets[target_month])
        mask_samples.append(monthly_masks[target_month])
        atm_samples.append([monthly_atm[target_month]])
        sample_months.append(f"{target_month[0]}-{target_month[1]:02d}")

    if not x_samples:
        raise RealDataLoadError("No sequence samples were produced for the monthly ConvLSTM training set.")

    x_tensor = torch.tensor(np.stack(x_samples), dtype=torch.float32)
    y_tensor = torch.tensor(np.stack(y_samples), dtype=torch.float32)
    mask_tensor = torch.tensor(np.stack(mask_samples), dtype=torch.float32)
    atm_tensor = torch.tensor(np.stack(atm_samples), dtype=torch.float32)

    tensor_dir = PROCESSED_TENSOR_DIR / f"{resolution}_window_{month_window}"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    x_path = tensor_dir / "X.pt"
    y_path = tensor_dir / "y.pt"
    mask_path = tensor_dir / "mask.pt"
    atm_path = tensor_dir / "atm.pt"
    metadata_path = tensor_dir / "metadata.json"
    torch.save(x_tensor, x_path)
    torch.save(y_tensor, y_path)
    torch.save(mask_tensor, mask_path)
    torch.save(atm_tensor, atm_path)
    metadata = {
        "feature_names": feature_names,
        "sample_months": sample_months,
        "reference_now": reference_now.astimezone(timezone.utc).isoformat(),
        "resolution": resolution,
        "month_window": month_window,
        "grid_shape": [int(len(lat_values)), int(len(lon_values))],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    summary = {
        "source": "real_monthly_tensor_pipeline",
        "tensor_dir": str(tensor_dir),
        "sample_count": int(x_tensor.shape[0]),
        "grid_shape": [int(x_tensor.shape[-2]), int(x_tensor.shape[-1])],
        "feature_count": int(x_tensor.shape[2]),
        "feature_names": feature_names,
        "target_months": sample_months,
    }
    return TensorBuildResult(
        tensor_dir=tensor_dir,
        x_path=x_path,
        y_path=y_path,
        mask_path=mask_path,
        atm_path=atm_path,
        metadata_path=metadata_path,
        summary=summary,
        feature_names=feature_names,
    )

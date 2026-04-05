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


def _winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    lower = float(valid.quantile(lower_q))
    upper = float(valid.quantile(upper_q))
    if lower > upper:
        lower, upper = upper, lower
    return numeric.clip(lower=lower, upper=upper)


def _smooth_spatial_grid(values: np.ndarray, mask: np.ndarray | None = None, passes: int = 1) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float32).copy()
    support = np.asarray(mask, dtype=np.float32) if mask is not None else np.ones_like(smoothed, dtype=np.float32)
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
                value_window = padded_values[di : di + smoothed.shape[0], dj : dj + smoothed.shape[1]]
                support_window = padded_support[di : di + smoothed.shape[0], dj : dj + smoothed.shape[1]]
                numerator += value_window * weight
                denominator += support_window * weight
        smoothed = np.divide(numerator, np.maximum(denominator, 1e-6), out=smoothed, where=denominator > 0)
        smoothed = np.where(support > 0, smoothed, values)
    return smoothed


def _flux_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    current_u = pd.to_numeric(frame.get("copernicus_u", pd.Series(index=frame.index, dtype=float)), errors="coerce").fillna(0.0)
    current_v = pd.to_numeric(frame.get("copernicus_v", pd.Series(index=frame.index, dtype=float)), errors="coerce").fillna(0.0)
    sea_level = pd.to_numeric(frame.get("copernicus_zos", pd.Series(index=frame.index, dtype=float)), errors="coerce").fillna(0.0)
    lat = pd.to_numeric(frame.get("lat_bin", frame.get("lat", pd.Series(index=frame.index, dtype=float))), errors="coerce").fillna(0.0)
    lon = _normalize_longitude(frame.get("lon_bin", frame.get("lon", pd.Series(index=frame.index, dtype=float)))).fillna(0.0)
    month = pd.to_numeric(frame["month"], errors="coerce").fillna(1.0)
    month_angle = 2 * np.pi * month / 12.0
    current_speed = np.sqrt((current_u.to_numpy(dtype=float) ** 2) + (current_v.to_numpy(dtype=float) ** 2))
    return np.column_stack(
        [
            pd.to_numeric(frame["sst"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 30.0,
            pd.to_numeric(frame["salinity"], errors="coerce").fillna(35.0).to_numpy(dtype=float) / 40.0,
            pd.to_numeric(frame["wind_speed"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 20.0,
            current_u.to_numpy(dtype=float) / 3.0,
            current_v.to_numpy(dtype=float) / 3.0,
            current_speed / 4.0,
            sea_level.to_numpy(dtype=float) / 2.0,
            pd.to_numeric(frame["pco2_atm"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 500.0,
            lat.to_numpy(dtype=float) / 80.0,
            lon.to_numpy(dtype=float) / 180.0,
            np.sin(month_angle.to_numpy(dtype=float)),
            np.cos(month_angle.to_numpy(dtype=float)),
        ]
    ).astype(np.float64)


def _fit_flux_surrogate(observed_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    design = _flux_feature_matrix(observed_frame)
    target = pd.to_numeric(observed_frame["target_flux"], errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(design).all(axis=1) & np.isfinite(target)
    if valid_mask.sum() < 32:
        raise RealDataLoadError("Need at least 32 valid observed flux rows to fit dense global training targets.")

    design = design[valid_mask]
    target = target[valid_mask]
    feature_mean = design.mean(axis=0, keepdims=True)
    feature_std = design.std(axis=0, keepdims=True) + 1e-6
    normalized = (design - feature_mean) / feature_std
    ridge = 5e-2 * np.eye(normalized.shape[1], dtype=np.float64)
    weights = np.linalg.solve(normalized.T @ normalized + ridge, normalized.T @ target)
    bias = float(np.mean(target - (normalized @ weights)))
    lower_q, upper_q = np.quantile(target, [0.02, 0.98])
    return feature_mean, feature_std, weights, bias, float(lower_q), float(upper_q)


def _predict_flux_from_surrogate(
    frame: pd.DataFrame,
    surrogate: tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float],
) -> np.ndarray:
    feature_mean, feature_std, weights, bias, lower_q, upper_q = surrogate
    design = _flux_feature_matrix(frame)
    normalized = (design - feature_mean) / feature_std
    predicted = (normalized @ weights) + bias
    return np.clip(predicted, lower_q, upper_q).astype(np.float32)


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

    merged["sst"] = merged.get("sst", pd.Series(index=merged.index, dtype=float))
    merged["sst"] = merged["sst"].fillna(merged.get("copernicus_thetao", pd.Series(index=merged.index, dtype=float)))
    merged["sst"] = merged["sst"].fillna(merged.get("era5_sst_c", pd.Series(index=merged.index, dtype=float)))
    merged["salinity"] = merged.get("salinity", pd.Series(index=merged.index, dtype=float))
    merged["salinity"] = merged["salinity"].fillna(
        merged.get("copernicus_salinity", pd.Series(index=merged.index, dtype=float))
    )
    merged["wind_speed"] = merged["era5_wind_speed"]
    merged = merged.dropna(subset=["sst", "salinity", "wind_speed", "pco2_ocean", "pco2_atm"])
    for column in ["sst", "salinity", "wind_speed", "pco2_ocean", "pco2_atm"]:
        merged[column] = _winsorize_series(merged[column])
    merged["target_flux"] = compute_co2_flux(
        merged["pco2_ocean"].to_numpy(dtype=float),
        merged["pco2_atm"].to_numpy(dtype=float),
        merged["sst"].to_numpy(dtype=float),
        merged["wind_speed"].to_numpy(dtype=float),
        merged["salinity"].to_numpy(dtype=float),
    )
    merged["target_flux"] = _winsorize_series(merged["target_flux"], lower_q=0.02, upper_q=0.98)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_flux"])
    if merged.empty:
        raise RealDataLoadError("No usable target flux rows were produced for monthly tensor building.")

    observed_target_grid = (
        merged.groupby(["year", "month", "lat_bin", "lon_bin"], as_index=False)
        .agg(
            target_flux=("target_flux", "mean"),
            observed_count=("target_flux", "size"),
            sst_obs=("sst", "mean"),
            salinity_obs=("salinity", "mean"),
        )
        .sort_values(["year", "month", "lat_bin", "lon_bin"])
    )

    surrogate = _fit_flux_surrogate(merged)
    monthly_bias = observed_target_grid.merge(
        merged.groupby(["year", "month"], as_index=False)
        .agg(
            observed_flux_mean=("target_flux", "mean"),
        ),
        on=["year", "month"],
        how="left",
    )[["year", "month"]].drop_duplicates()
    observed_proxy = merged[["year", "month", "lat_bin", "lon_bin", "target_flux", "sst", "salinity", "wind_speed", "pco2_atm"]].copy()
    observed_proxy["copernicus_u"] = merged.get("copernicus_u", pd.Series(index=merged.index, dtype=float)).fillna(0.0)
    observed_proxy["copernicus_v"] = merged.get("copernicus_v", pd.Series(index=merged.index, dtype=float)).fillna(0.0)
    observed_proxy["copernicus_zos"] = merged.get("copernicus_zos", pd.Series(index=merged.index, dtype=float)).fillna(0.0)
    observed_proxy["predicted_flux_proxy"] = _predict_flux_from_surrogate(observed_proxy, surrogate)
    month_bias = (
        observed_proxy.groupby(["year", "month"], as_index=False)
        .apply(lambda frame: pd.Series({"month_bias": float(np.mean(frame["target_flux"] - frame["predicted_flux_proxy"]))}))
        .reset_index(drop=True)
    )

    dense_grid = cop.copy()
    dense_grid = dense_grid.merge(atmospheric[["year", "month", "pco2_atm"]], on=["year", "month"], how="inner")
    dense_grid = dense_grid.merge(era_monthly, on=["year", "month"], how="left")
    dense_grid["sst"] = dense_grid.get("copernicus_thetao", pd.Series(index=dense_grid.index, dtype=float))
    dense_grid["sst"] = dense_grid["sst"].fillna(dense_grid.get("era5_sst_c", pd.Series(index=dense_grid.index, dtype=float)))
    dense_grid["salinity"] = dense_grid.get("copernicus_salinity", pd.Series(index=dense_grid.index, dtype=float))
    dense_grid["wind_speed"] = dense_grid.get("era5_wind_speed", pd.Series(index=dense_grid.index, dtype=float))
    dense_grid["copernicus_u"] = dense_grid.get("copernicus_u", pd.Series(index=dense_grid.index, dtype=float)).fillna(0.0)
    dense_grid["copernicus_v"] = dense_grid.get("copernicus_v", pd.Series(index=dense_grid.index, dtype=float)).fillna(0.0)
    dense_grid["copernicus_zos"] = dense_grid.get("copernicus_zos", pd.Series(index=dense_grid.index, dtype=float)).fillna(0.0)
    dense_grid = dense_grid.dropna(subset=["sst", "salinity", "wind_speed", "pco2_atm"])
    if dense_grid.empty:
        raise RealDataLoadError("No dense gridded Copernicus/ERA5/NOAA rows were available to build global training targets.")

    dense_grid["target_flux"] = _predict_flux_from_surrogate(dense_grid, surrogate)
    dense_grid = dense_grid.merge(month_bias, on=["year", "month"], how="left")
    dense_grid["target_flux"] = dense_grid["target_flux"] + dense_grid["month_bias"].fillna(0.0)
    dense_grid["target_flux"] = _winsorize_series(dense_grid["target_flux"], lower_q=0.02, upper_q=0.98)

    target_grid = dense_grid.merge(
        observed_target_grid,
        on=["year", "month", "lat_bin", "lon_bin"],
        how="left",
        suffixes=("", "_observed"),
    )
    target_grid["target_flux"] = target_grid["target_flux_observed"].fillna(target_grid["target_flux"])
    target_grid["observed_count"] = target_grid["observed_count"].fillna(0).astype(int)
    target_grid["sst_obs"] = target_grid["sst_obs"].fillna(target_grid["sst"])
    target_grid["salinity_obs"] = target_grid["salinity_obs"].fillna(target_grid["salinity"])
    target_grid = target_grid[["year", "month", "lat_bin", "lon_bin", "target_flux", "observed_count", "sst_obs", "salinity_obs"]]
    target_grid = target_grid.sort_values(["year", "month", "lat_bin", "lon_bin"]).reset_index(drop=True)

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
    if len(months) < 3:
        raise RealDataLoadError(
            f"Not enough monthly records for sequence training. Need at least 3 overlapping months, found {len(months)}."
        )

    requested_month_window = int(month_window)
    minimum_sequence_samples = 8
    effective_month_window = min(requested_month_window, max(1, len(months) - minimum_sequence_samples))
    if effective_month_window < requested_month_window:
        month_window = effective_month_window
    if len(months) <= month_window:
        raise RealDataLoadError(
            f"Not enough monthly records for sequence training. Need > {month_window} overlapping months, found {len(months)}."
        )

    feature_names = [
        "thetao",
        "salinity",
        "current_u",
        "current_v",
        "current_speed",
        "wind_speed",
        "sea_level",
        "pco2_atm",
        "month_sin",
        "month_cos",
        "lat_norm",
        "lon_norm",
        "ocean_mask",
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
        feature_grid[4, :, :] = np.sqrt((u_default**2) + (v_default**2)) / 4.0
        feature_grid[5, :, :] = wind_default / 20.0
        feature_grid[6, :, :] = sea_level_default / 2.0
        feature_grid[7, :, :] = atm_default / 500.0
        feature_grid[8, :, :] = np.sin(angle)
        feature_grid[9, :, :] = np.cos(angle)
        feature_grid[10, :, :] = lat_values[:, None] / 80.0
        feature_grid[11, :, :] = lon_values[None, :] / 180.0
        feature_grid[12, :, :] = 0.0

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
            feature_grid[12, i, j] = 1.0
            if not np.isnan(getattr(row, "sst_obs", np.nan)):
                feature_grid[0, i, j] = float(row.sst_obs) / 30.0
            if not np.isnan(getattr(row, "salinity_obs", np.nan)):
                feature_grid[1, i, j] = float(row.salinity_obs) / 40.0

        if target_mask.sum() > 0:
            smoothed_target = _smooth_spatial_grid(target_values[0], target_mask[0], passes=2)
            target_values[0] = np.where(target_mask[0] > 0, smoothed_target, target_values[0])

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
        "requested_month_window": requested_month_window,
        "effective_month_window": month_window,
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
        "requested_month_window": requested_month_window,
        "effective_month_window": int(month_window),
        "available_overlap_months": int(len(months)),
        "minimum_sequence_samples_target": int(minimum_sequence_samples),
        "denoising": {
            "winsorized_columns": ["sst", "salinity", "wind_speed", "pco2_ocean", "pco2_atm", "target_flux"],
            "target_flux_clip_quantiles": [0.02, 0.98],
            "target_grid_smoothing": {"kernel": "3x3 weighted mean", "passes": 2},
        },
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

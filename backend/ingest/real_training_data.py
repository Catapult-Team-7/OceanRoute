from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compute_flux import compute_co2_flux


logger = logging.getLogger("oceanpulse.ingest.real_data")

DEFAULT_NOAA_GML_CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
DEFAULT_ERA5_DIRECTORY = Path(__file__).resolve().parents[2] / "Training_Data" / "ERA"


class RealDataLoadError(RuntimeError):
    pass


@dataclass
class RealTrainingDataset:
    features: np.ndarray
    targets: np.ndarray
    summary: dict[str, Any]
    atmospheric_lookup: dict[tuple[int, int], float]
    feature_names: list[str]


def default_era5_directory() -> Path:
    return DEFAULT_ERA5_DIRECTORY


def _normalize_longitude(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return ((values + 180) % 360) - 180


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_noaa_gml_monthly(data_url: str | None = None) -> pd.DataFrame:
    url = (data_url or DEFAULT_NOAA_GML_CO2_URL).strip() or DEFAULT_NOAA_GML_CO2_URL
    logger.info("loading_noaa_gml_monthly url=%s", url)

    try:
        if url.endswith(".txt"):
            frame = pd.read_csv(
                url,
                sep=r"\s+",
                comment="#",
                header=None,
                names=["year", "month", "decimal", "average", "de_season", "days", "stdev", "uncertainty"],
                engine="python",
            )
        else:
            frame = pd.read_csv(url, comment="#")
    except Exception as exc:
        raise RealDataLoadError(f"Failed to load NOAA GML CO2 data from {url}: {exc}") from exc

    normalized = _normalize_noaa_gml_frame(frame)
    if normalized.empty:
        raise RealDataLoadError("NOAA GML CO2 data loaded, but no valid monthly values were found.")
    return normalized


def _normalize_noaa_gml_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]

    average_column = None
    for candidate in ("average", "trend", "interpolated", "co2", "monthly_average"):
        if candidate in normalized.columns:
            average_column = candidate
            break
    if average_column is None:
        raise RealDataLoadError(
            f"NOAA GML CO2 data is missing a recognized concentration column. Columns: {list(normalized.columns)}"
        )

    if "year" not in normalized.columns or "month" not in normalized.columns:
        raise RealDataLoadError("NOAA GML CO2 data must include year and month columns.")

    normalized = _safe_numeric(normalized, ["year", "month", average_column])
    normalized = normalized.rename(columns={average_column: "pco2_atm"})
    normalized = normalized[(normalized["pco2_atm"] > 0) & (normalized["month"].between(1, 12))]
    normalized["year"] = normalized["year"].astype(int)
    normalized["month"] = normalized["month"].astype(int)
    normalized["date"] = pd.to_datetime(
        {"year": normalized["year"], "month": normalized["month"], "day": 1},
        utc=True,
        errors="coerce",
    )
    normalized = normalized.dropna(subset=["date"])
    normalized = normalized[["date", "year", "month", "pco2_atm"]].sort_values("date").reset_index(drop=True)
    return normalized


def load_socat_observations(data_url: str, sample_size: int = 15000) -> pd.DataFrame:
    url = data_url.strip()
    if not url:
        raise RealDataLoadError("SOCAT URL is empty.")

    logger.info("loading_socat_observations url=%s sample_size=%s", url, sample_size)

    delimiter = "\t" if url.endswith((".tsv", ".txt", ".zip")) else ","
    read_kwargs = {
        "sep": delimiter,
        "comment": "%",
        "compression": "infer",
        "usecols": lambda column: str(column).strip()
        in {"yr", "mon", "day", "latitude", "longitude", "fCO2rec", "SST_C", "salinity"},
        "nrows": max(sample_size * 3, 6000),
        "low_memory": False,
    }

    try:
        frame = pd.read_csv(url, **read_kwargs)
    except Exception as exc:
        raise RealDataLoadError(f"Failed to load SOCAT data from {url}: {exc}") from exc

    normalized = _normalize_socat_frame(frame)
    if normalized.empty:
        raise RealDataLoadError("SOCAT data loaded, but no valid rows were found after cleaning.")

    if len(normalized) > sample_size:
        normalized = normalized.sample(n=sample_size, random_state=42).sort_values("datetime")

    return normalized.reset_index(drop=True)


def _normalize_socat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    required = {"yr", "mon", "day", "latitude", "longitude", "fCO2rec", "SST_C", "salinity"}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise RealDataLoadError(f"SOCAT data is missing required columns: {', '.join(missing)}")

    normalized = _safe_numeric(
        normalized,
        ["yr", "mon", "day", "latitude", "longitude", "fCO2rec", "SST_C", "salinity"],
    )
    normalized = normalized.dropna(subset=["yr", "mon", "day", "latitude", "longitude", "fCO2rec"])
    normalized["datetime"] = pd.to_datetime(
        {
            "year": normalized["yr"].astype(int),
            "month": normalized["mon"].astype(int),
            "day": normalized["day"].astype(int),
        },
        utc=True,
        errors="coerce",
    )
    normalized["lon"] = _normalize_longitude(normalized["longitude"])
    normalized["lat"] = pd.to_numeric(normalized["latitude"], errors="coerce")
    normalized["sst"] = pd.to_numeric(normalized["SST_C"], errors="coerce")
    normalized["salinity"] = pd.to_numeric(normalized["salinity"], errors="coerce")
    normalized["pco2_ocean"] = pd.to_numeric(normalized["fCO2rec"], errors="coerce")
    normalized = normalized.dropna(subset=["datetime", "lat", "lon", "pco2_ocean"])
    normalized = normalized[(normalized["lat"].between(-89.5, 89.5)) & (normalized["lon"].between(-180, 180))]
    normalized["year"] = normalized["datetime"].dt.year
    normalized["month"] = normalized["datetime"].dt.month
    normalized = normalized[["datetime", "year", "month", "lat", "lon", "sst", "salinity", "pco2_ocean"]]
    normalized = normalized.sort_values("datetime").reset_index(drop=True)
    return normalized


def _estimate_climatology(lat: pd.Series, lon: pd.Series, month: pd.Series) -> tuple[pd.Series, pd.Series]:
    month_angle = 2 * np.pi * month / 12
    wind_speed = 4.5 + np.abs(np.sin(np.radians(lat * 2))) * 6.0 + np.cos(month_angle) * 0.35
    chl_a = np.maximum(0.02, 0.8 + np.cos(np.radians(lat * 3)) * 0.4 + np.sin(month_angle) * 0.08)
    return wind_speed.astype(float), pd.Series(chl_a, index=lat.index, dtype=float)


def load_era5_monthly(era_directory: str | Path | None = None) -> pd.DataFrame:
    base_path = Path(era_directory) if era_directory else default_era5_directory()
    if not base_path.exists():
        raise RealDataLoadError(f"ERA5 directory not found: {base_path}")

    files = sorted(base_path.glob("*.csv"))
    if not files:
        raise RealDataLoadError(f"No ERA5 CSV files found in {base_path}")

    monthly_frames = []
    for file_path in files:
        logger.info("loading_era5_file path=%s", file_path)
        try:
            frame = pd.read_csv(file_path, low_memory=False)
        except Exception as exc:
            raise RealDataLoadError(f"Failed to load ERA5 file {file_path}: {exc}") from exc

        if "valid_time" not in frame.columns:
            raise RealDataLoadError(f"ERA5 file {file_path} is missing the valid_time column.")

        frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["valid_time"])
        for column in frame.columns:
            if column not in {"valid_time"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["year"] = frame["valid_time"].dt.year.astype(int)
        frame["month"] = frame["valid_time"].dt.month.astype(int)

        numeric_columns = [column for column in frame.columns if column not in {"valid_time", "year", "month"}]
        monthly = frame.groupby(["year", "month"], as_index=False)[numeric_columns].mean(numeric_only=True)
        monthly_frames.append(monthly)

    merged = monthly_frames[0]
    for monthly in monthly_frames[1:]:
        overlapping = [column for column in monthly.columns if column in merged.columns and column not in {"year", "month"}]
        monthly = monthly.drop(columns=overlapping)
        merged = merged.merge(monthly, on=["year", "month"], how="outer")

    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)
    if {"u10", "v10"}.issubset(merged.columns):
        merged["era5_wind_speed"] = np.sqrt((merged["u10"] ** 2) + (merged["v10"] ** 2))
    if "sst" in merged.columns:
        merged["era5_sst_c"] = merged["sst"] - 273.15
    return merged


def build_real_training_dataset(
    *,
    socat_url: str,
    noaa_gml_url: str | None,
    month_window: int,
    reference_now: datetime,
    era_directory: str | Path | None = None,
    sample_size: int = 12000,
) -> RealTrainingDataset:
    atmospheric = load_noaa_gml_monthly(noaa_gml_url)
    socat = load_socat_observations(socat_url, sample_size=sample_size)

    merged = socat.merge(atmospheric[["year", "month", "pco2_atm"]], on=["year", "month"], how="inner")
    if merged.empty:
        raise RealDataLoadError("SOCAT and NOAA GML data loaded, but no overlapping year/month records were found.")

    latest_data_time = merged["datetime"].max().to_pydatetime()
    if latest_data_time.tzinfo is None:
        latest_data_time = latest_data_time.replace(tzinfo=timezone.utc)
    lookback_months = max(month_window * 6, 36)
    cutoff = (pd.Timestamp(latest_data_time) - pd.DateOffset(months=lookback_months)).to_pydatetime()
    recent = merged[merged["datetime"] >= cutoff]
    if len(recent) >= 1000:
        merged = recent.copy()

    era_summary: dict[str, Any] = {
        "available": False,
        "path": str(Path(era_directory) if era_directory else default_era5_directory()),
    }
    try:
        era_monthly = load_era5_monthly(era_directory)
        merged = merged.merge(era_monthly, on=["year", "month"], how="left")
        era_summary = {
            "available": True,
            "path": str(Path(era_directory) if era_directory else default_era5_directory()),
            "months_loaded": int(len(era_monthly)),
            "columns": [column for column in era_monthly.columns if column not in {"year", "month"}],
        }
    except RealDataLoadError as exc:
        logger.warning("era5_enrichment_unavailable reason=%s", exc)
        era_summary["error"] = str(exc)

    merged["sst"] = merged["sst"].fillna(merged.get("era5_sst_c", pd.Series(index=merged.index, dtype=float)))
    merged["sst"] = merged["sst"].fillna(18.0)
    merged["salinity"] = merged["salinity"].fillna(35.0)
    climatology_wind, climatology_chl = _estimate_climatology(merged["lat"], merged["lon"], merged["month"])
    merged["wind_speed"] = merged.get("era5_wind_speed", pd.Series(index=merged.index, dtype=float))
    merged["wind_speed"] = merged["wind_speed"].fillna(climatology_wind)
    merged["chl_a"] = climatology_chl
    merged["target_flux"] = compute_co2_flux(
        merged["pco2_ocean"].to_numpy(dtype=float),
        merged["pco2_atm"].to_numpy(dtype=float),
        merged["sst"].to_numpy(dtype=float),
        merged["wind_speed"].to_numpy(dtype=float),
        merged["salinity"].to_numpy(dtype=float),
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_flux", "sst", "salinity", "wind_speed", "chl_a"]
    )

    if merged.empty:
        raise RealDataLoadError("Real observations were loaded, but target flux construction produced no usable rows.")

    feature_names = [
        "lat_norm",
        "lon_norm",
        "sst",
        "salinity",
        "wind_speed",
        "chl_a",
        "month_sin",
        "month_cos",
    ]

    month_angle = 2 * np.pi * merged["month"] / 12
    features = np.column_stack(
        [
            merged["lat"].to_numpy(dtype=float) / 90,
            merged["lon"].to_numpy(dtype=float) / 180,
            merged["sst"].to_numpy(dtype=float) / 30,
            merged["salinity"].to_numpy(dtype=float) / 40,
            merged["wind_speed"].to_numpy(dtype=float) / 20,
            merged["chl_a"].to_numpy(dtype=float) / 5,
            np.sin(month_angle.to_numpy(dtype=float)),
            np.cos(month_angle.to_numpy(dtype=float)),
        ]
    ).astype(np.float64)
    targets = merged["target_flux"].to_numpy(dtype=np.float64).reshape(-1, 1)

    atmospheric_lookup = {
        (int(row.year), int(row.month)): float(row.pco2_atm)
        for row in atmospheric.itertuples(index=False)
    }
    summary = {
        "source": "real_observation_sample",
        "target_mode": "flux_from_socat_plus_noaa_gml",
        "sample_count": int(len(merged)),
        "socat_rows_loaded": int(len(socat)),
        "noaa_months_loaded": int(len(atmospheric)),
        "lookback_months_applied": int(lookback_months),
        "latest_observation": latest_data_time.isoformat(),
        "feature_names": feature_names,
        "era5": era_summary,
        "coverage": {
            "lat_min": round(float(merged["lat"].min()), 3),
            "lat_max": round(float(merged["lat"].max()), 3),
            "lon_min": round(float(merged["lon"].min()), 3),
            "lon_max": round(float(merged["lon"].max()), 3),
        },
        "reference_now": reference_now.astimezone(timezone.utc).isoformat(),
    }

    return RealTrainingDataset(
        features=features,
        targets=targets,
        summary=summary,
        atmospheric_lookup=atmospheric_lookup,
        feature_names=feature_names,
    )

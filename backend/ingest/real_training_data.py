from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compute_flux import compute_co2_flux
from .fetch_copernicus import default_copernicus_directory


logger = logging.getLogger("oceanpulse.ingest.real_data")

DEFAULT_NOAA_GML_CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
DEFAULT_SOCAT_ERDDAP_URL = "https://data.pmel.noaa.gov/socat/erddap/tabledap/socat_v2025_decimated.csv"
DEFAULT_ERA5_DIRECTORY = Path(__file__).resolve().parents[2] / "Training_Data" / "ERA"
DEFAULT_TRAINING_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "Training_Data"
DEFAULT_REPO_ROOT = DEFAULT_TRAINING_DATA_DIRECTORY.parent


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


def _is_remote_source(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _discover_local_file(
    *,
    explicit_path: str | None,
    env_var: str,
    glob_patterns: list[str],
) -> Path | None:
    candidates: list[Path] = []
    if explicit_path and not _is_remote_source(explicit_path):
        raw = Path(explicit_path).expanduser()
        candidates.append(raw if raw.is_absolute() else (DEFAULT_REPO_ROOT / raw))
    env_value = os.getenv(env_var, "").strip()
    if env_value:
        raw = Path(env_value).expanduser()
        candidates.append(raw if raw.is_absolute() else (DEFAULT_REPO_ROOT / raw))
    for pattern in glob_patterns:
        candidates.extend(sorted(DEFAULT_TRAINING_DATA_DIRECTORY.glob(pattern)))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_directory_path(path_like: str | Path | None, default_path: Path) -> Path:
    if path_like is None or str(path_like).strip() == "":
        return default_path
    raw = Path(path_like).expanduser()
    return raw if raw.is_absolute() else (DEFAULT_REPO_ROOT / raw)


def _normalize_longitude(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return ((values + 180) % 360) - 180


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _decode_copernicus_time_coord(time_var) -> pd.DatetimeIndex | None:
    raw_values = np.asarray(time_var.values)
    if raw_values.size == 0:
        return None

    if np.issubdtype(raw_values.dtype, np.datetime64):
        decoded = pd.to_datetime(raw_values, utc=True, errors="coerce")
        return decoded.tz_localize(None) if getattr(decoded, "tz", None) is not None else decoded

    units = str(time_var.attrs.get("units", "")).strip().lower()
    origin = None
    unit = None
    if "since" in units:
        unit_label, origin_label = units.split("since", 1)
        origin = pd.to_datetime(origin_label.strip(), utc=True, errors="coerce")
        unit_label = unit_label.strip()
        if unit_label.startswith("hour"):
            unit = "h"
        elif unit_label.startswith("day"):
            unit = "D"
        elif unit_label.startswith("minute"):
            unit = "m"
        elif unit_label.startswith("second"):
            unit = "s"

    numeric_values = pd.to_numeric(raw_values.reshape(-1), errors="coerce")
    if unit and origin is not None and not pd.isna(origin):
        valid_mask = np.isfinite(numeric_values) & (np.abs(numeric_values) < 1e12)
        decoded_flat = np.full(numeric_values.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        if valid_mask.any():
            decoded_valid = origin + pd.to_timedelta(numeric_values[valid_mask], unit=unit)
            decoded_flat[valid_mask] = decoded_valid.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
        return pd.DatetimeIndex(decoded_flat.reshape(raw_values.shape))

    decoded = pd.to_datetime(raw_values.reshape(-1), utc=True, errors="coerce")
    decoded = decoded.tz_localize(None) if getattr(decoded, "tz", None) is not None else decoded
    return pd.DatetimeIndex(decoded.to_numpy().reshape(raw_values.shape))


def _build_socat_erddap_query(base_url: str, sample_size: int = 15000) -> str:
    root = base_url.split("?", 1)[0]
    columns = [
        "time",
        "longitude",
        "latitude",
        "sal",
        "temp",
        "fCO2_insitu_from_xCO2_water_sst_dry_ppm_ncep",
    ]
    # Keep the live query simple and compatible across ERDDAP deployments:
    # request only the columns we need and constrain to recent years.
    # pandas' nrows then bounds the streamed response size client-side.
    constraints = [
        "&time>=2018-01-01T00:00:00Z",
    ]
    return f"{root}?{','.join(columns)}{''.join(constraints)}"


def load_noaa_gml_monthly(data_url: str | None = None) -> pd.DataFrame:
    url = (data_url or DEFAULT_NOAA_GML_CO2_URL).strip() or DEFAULT_NOAA_GML_CO2_URL
    local_fallback = _discover_local_file(
        explicit_path=url,
        env_var="NOAA_GML_LOCAL_PATH",
        glob_patterns=[
            "NOAA_GML/*.txt",
            "NOAA_GML/*.csv",
            "**/co2_mm_mlo.txt",
            "**/*mlo*co2*.txt",
            "**/*monthly*co2*.csv",
        ],
    )
    logger.info("loading_noaa_gml_monthly url=%s", url)

    try:
        source = local_fallback if local_fallback and not _is_remote_source(url) else url
        if str(source).endswith(".txt"):
            frame = pd.read_csv(
                source,
                sep=r"\s+",
                comment="#",
                header=None,
                names=["year", "month", "decimal", "average", "de_season", "days", "stdev", "uncertainty"],
                engine="python",
            )
        else:
            frame = pd.read_csv(source, comment="#")
    except Exception as exc:
        if local_fallback and _is_remote_source(url):
            logger.warning("noaa_gml_remote_failed_falling_back_to_local url=%s path=%s", url, local_fallback)
            return load_noaa_gml_monthly(str(local_fallback))
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
    local_fallback = _discover_local_file(
        explicit_path=url,
        env_var="SOCAT_LOCAL_PATH",
        glob_patterns=[
            "SOCAT/*.tsv",
            "SOCAT/*.zip",
            "SOCAT/*.csv",
            "**/SOCAT*.tsv",
            "**/SOCAT*.zip",
            "**/SOCAT*.csv",
        ],
    )

    logger.info("loading_socat_observations url=%s sample_size=%s", url, sample_size)

    source = local_fallback if local_fallback and not _is_remote_source(url) else url
    if isinstance(source, str) and "/erddap/tabledap/" in source and "?" not in source:
        source = _build_socat_erddap_query(source, sample_size=sample_size)
    erddap_source = isinstance(source, str) and "/erddap/tabledap/" in source
    delimiter = "\t" if str(source).endswith((".tsv", ".txt", ".zip")) else ","
    read_kwargs = {
        "sep": delimiter,
        "comment": "%",
        "compression": "infer" if not str(source).startswith("http") else None,
        "usecols": lambda column: str(column).strip()
        in {
            "yr",
            "mon",
            "day",
            "latitude",
            "longitude",
            "fCO2rec",
            "SST_C",
            "salinity",
            "time",
            "sal",
            "temp",
            "fCO2_insitu_from_xCO2_water_sst_dry_ppm_ncep",
        },
        "nrows": max(sample_size * 10, 50000) if erddap_source else max(sample_size * 3, 6000),
        "low_memory": False,
    }

    try:
        frame = pd.read_csv(source, **read_kwargs)
    except Exception as exc:
        if local_fallback and _is_remote_source(url):
            logger.warning("socat_remote_failed_falling_back_to_local url=%s path=%s", url, local_fallback)
            return load_socat_observations(str(local_fallback), sample_size=sample_size)
        raise RealDataLoadError(f"Failed to load SOCAT data from {url}: {exc}") from exc

    normalized = _normalize_socat_frame(frame)
    if normalized.empty and erddap_source:
        logger.warning(
            "socat_erddap_empty_after_cleaning_retrying url=%s columns=%s",
            source,
            list(frame.columns),
        )
        retry_kwargs = {**read_kwargs, "nrows": max(read_kwargs["nrows"], 100000)}
        frame = pd.read_csv(source, **retry_kwargs)
        normalized = _normalize_socat_frame(frame)
    if normalized.empty:
        raise RealDataLoadError(
            f"SOCAT data loaded, but no valid rows were found after cleaning. Raw columns: {list(frame.columns)}"
        )

    if len(normalized) > sample_size:
        normalized = normalized.sample(n=sample_size, random_state=42).sort_values("datetime")

    return normalized.reset_index(drop=True)


def _normalize_socat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().strip('"').lower() for column in normalized.columns]

    if {"time", "latitude", "longitude"}.issubset(normalized.columns):
        salinity_column = next((column for column in ("sal", "salinity") if column in normalized.columns), None)
        temperature_column = next((column for column in ("temp", "sst_c", "sst") if column in normalized.columns), None)
        pco2_column = next(
            (
                column
                for column in normalized.columns
                if "fco2" in column.lower() or "pco2_ocean" in column.lower() or column.lower() == "fco2rec"
            ),
            None,
        )
        if pco2_column is None:
            raise RealDataLoadError(
                f"SOCAT ERDDAP data did not include a recognizable fCO2 column. Columns: {list(normalized.columns)}"
            )

        numeric_columns = ["latitude", "longitude", pco2_column]
        if salinity_column:
            numeric_columns.append(salinity_column)
        if temperature_column:
            numeric_columns.append(temperature_column)
        normalized = _safe_numeric(normalized, numeric_columns)

        normalized["datetime"] = pd.to_datetime(normalized["time"], utc=True, errors="coerce", format="ISO8601")
        if normalized["datetime"].isna().all():
            normalized["datetime"] = pd.to_datetime(normalized["time"], utc=True, errors="coerce")
        normalized["lon"] = _normalize_longitude(normalized["longitude"])
        normalized["lat"] = pd.to_numeric(normalized["latitude"], errors="coerce")
        normalized["sst"] = pd.to_numeric(normalized[temperature_column], errors="coerce") if temperature_column else np.nan
        normalized["salinity"] = pd.to_numeric(normalized[salinity_column], errors="coerce") if salinity_column else np.nan
        normalized["pco2_ocean"] = pd.to_numeric(normalized[pco2_column], errors="coerce")
        normalized = normalized.dropna(subset=["datetime", "lat", "lon", "pco2_ocean"])
        normalized = normalized[(normalized["lat"].between(-89.5, 89.5)) & (normalized["lon"].between(-180, 180))]
        normalized["year"] = normalized["datetime"].dt.year
        normalized["month"] = normalized["datetime"].dt.month
        normalized = normalized[["datetime", "year", "month", "lat", "lon", "sst", "salinity", "pco2_ocean"]]
        normalized = normalized.sort_values("datetime").reset_index(drop=True)
        return normalized

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


def load_era5_monthly(era_directory: str | Path | None = None) -> pd.DataFrame:
    base_path = _resolve_directory_path(era_directory, default_era5_directory())
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


def load_copernicus_monthly(copernicus_directory: str | Path | None = None) -> pd.DataFrame:
    base_path = _resolve_directory_path(copernicus_directory, default_copernicus_directory())
    if not base_path.exists():
        raise RealDataLoadError(f"Copernicus directory not found: {base_path}")

    files = sorted(base_path.glob("*.nc"))
    if not files:
        raise RealDataLoadError(f"No Copernicus NetCDF files found in {base_path}")

    try:
        import xarray as xr
    except ImportError as exc:
        raise RealDataLoadError(
            "Copernicus local files require `xarray` and `netCDF4`. Install backend/requirements-ml-ingest.txt."
        ) from exc

    monthly_frames: list[pd.DataFrame] = []
    variable_map = {
        "copernicus_salinity": ("so", "sos", "salinity"),
        "copernicus_u": ("uo", "eastward_sea_water_velocity", "water_u"),
        "copernicus_v": ("vo", "northward_sea_water_velocity", "water_v"),
        "copernicus_thetao": ("thetao", "bottomT", "temperature"),
        "copernicus_zos": ("zos", "sla", "adt"),
    }
    for file_path in files:
        logger.info("loading_copernicus_file path=%s", file_path)
        try:
            dataset = xr.open_dataset(file_path, decode_times=False)
        except Exception as exc:
            raise RealDataLoadError(f"Failed to open Copernicus file {file_path}: {exc}") from exc

        lat_name = next((name for name in ("latitude", "lat") if name in dataset.coords or name in dataset.dims), None)
        lon_name = next((name for name in ("longitude", "lon") if name in dataset.coords or name in dataset.dims), None)
        time_name = next((name for name in ("time", "valid_time") if name in dataset.coords or name in dataset.dims), None)
        if time_name and time_name in dataset.coords:
            decoded_time = _decode_copernicus_time_coord(dataset[time_name])
            if decoded_time is not None:
                valid_mask = ~pd.isna(decoded_time)
                if np.any(valid_mask):
                    valid_index = np.where(valid_mask)[0]
                    dataset = dataset.isel({time_name: valid_index})
                    dataset = dataset.assign_coords({time_name: decoded_time[valid_index]})
                    dataset = dataset.sortby(time_name)
        if not lat_name or not lon_name:
            continue

        extracted: list[pd.DataFrame] = []
        for alias, candidates in variable_map.items():
            variable_name = next((name for name in candidates if name in dataset.data_vars), None)
            if not variable_name:
                continue
            data_array = dataset[variable_name]
            depth_name = next((name for name in ("depth", "deptho", "depthu", "depthv") if name in data_array.dims), None)
            if depth_name:
                data_array = data_array.isel({depth_name: 0})
            if time_name and time_name in data_array.dims:
                data_array = data_array.resample({time_name: "1MS"}).mean()
            frame = data_array.to_dataframe(name=alias).reset_index()
            frame = frame.rename(columns={lat_name: "lat", lon_name: "lon", time_name or "time": "time"})
            if "time" not in frame.columns:
                file_time = pd.Timestamp(file_path.stat().st_mtime, unit="s", tz="UTC")
                frame["time"] = file_time
            frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["time", "lat", "lon", alias])
            frame["year"] = frame["time"].dt.year.astype(int)
            frame["month"] = frame["time"].dt.month.astype(int)
            frame["lat_bin"] = (pd.to_numeric(frame["lat"], errors="coerce") / 0.25).round() * 0.25
            frame["lon_bin"] = (_normalize_longitude(frame["lon"]) / 0.25).round() * 0.25
            frame = frame[["year", "month", "lat_bin", "lon_bin", alias]]
            frame = frame.groupby(["year", "month", "lat_bin", "lon_bin"], as_index=False).mean(numeric_only=True)
            extracted.append(frame)
        dataset.close()

        if not extracted:
            continue

        merged = extracted[0]
        for frame in extracted[1:]:
            merged = merged.merge(frame, on=["year", "month", "lat_bin", "lon_bin"], how="outer")
        monthly_frames.append(merged)

    if not monthly_frames:
        raise RealDataLoadError(f"Copernicus files in {base_path} did not expose salinity/current variables.")

    merged = pd.concat(monthly_frames, ignore_index=True)
    merged = merged.groupby(["year", "month", "lat_bin", "lon_bin"], as_index=False).mean(numeric_only=True)
    return merged.sort_values(["year", "month", "lat_bin", "lon_bin"]).reset_index(drop=True)


def build_real_training_dataset(
    *,
    socat_url: str,
    noaa_gml_url: str | None,
    month_window: int,
    reference_now: datetime,
    era_directory: str | Path | None = None,
    copernicus_directory: str | Path | None = None,
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
    copernicus_monthly: pd.DataFrame | None = None
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

    copernicus_summary: dict[str, Any] = {
        "available": False,
        "path": str(Path(copernicus_directory) if copernicus_directory else default_copernicus_directory()),
    }
    try:
        copernicus_monthly = load_copernicus_monthly(copernicus_directory)
        merged["lat_bin"] = (merged["lat"] / 0.25).round() * 0.25
        merged["lon_bin"] = (_normalize_longitude(merged["lon"]) / 0.25).round() * 0.25
        merged = merged.merge(copernicus_monthly, on=["year", "month", "lat_bin", "lon_bin"], how="left")
        copernicus_summary = {
            "available": True,
            "path": str(Path(copernicus_directory) if copernicus_directory else default_copernicus_directory()),
            "grid_rows_loaded": int(len(copernicus_monthly)),
            "columns": [
                column for column in copernicus_monthly.columns if column not in {"year", "month", "lat_bin", "lon_bin"}
            ],
        }
    except RealDataLoadError as exc:
        logger.warning("copernicus_enrichment_unavailable reason=%s", exc)
        copernicus_summary["error"] = str(exc)

    merged["sst"] = merged.get("sst", pd.Series(index=merged.index, dtype=float))
    merged["sst"] = merged["sst"].fillna(merged.get("copernicus_thetao", pd.Series(index=merged.index, dtype=float)))
    merged["sst"] = merged["sst"].fillna(merged.get("era5_sst_c", pd.Series(index=merged.index, dtype=float)))
    merged["salinity"] = merged.get("salinity", pd.Series(index=merged.index, dtype=float))
    merged["salinity"] = merged["salinity"].fillna(
        merged.get("copernicus_salinity", pd.Series(index=merged.index, dtype=float))
    )
    merged["wind_speed"] = merged.get("era5_wind_speed", pd.Series(index=merged.index, dtype=float))
    if merged["wind_speed"].notna().sum() == 0:
        raise RealDataLoadError(
            "ERA5 wind enrichment is required for real flux targets. Put real ERA5 CSVs in ~/OceanPulseData/ERA or set ERA5_LOCAL_PATH."
        )
    merged["current_u"] = merged.get("copernicus_u", pd.Series(index=merged.index, dtype=float)).fillna(0.0)
    merged["current_v"] = merged.get("copernicus_v", pd.Series(index=merged.index, dtype=float)).fillna(0.0)
    merged["sea_level"] = merged.get("copernicus_zos", pd.Series(index=merged.index, dtype=float)).fillna(0.0)
    merged["target_flux"] = compute_co2_flux(
        merged["pco2_ocean"].to_numpy(dtype=float),
        merged["pco2_atm"].to_numpy(dtype=float),
        merged["sst"].to_numpy(dtype=float),
        merged["wind_speed"].to_numpy(dtype=float),
        merged["salinity"].to_numpy(dtype=float),
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_flux", "sst", "salinity", "wind_speed"])

    if merged.empty:
        raise RealDataLoadError("Real observations were loaded, but target flux construction produced no usable rows.")

    feature_names = [
        "lat_norm",
        "lon_norm",
        "sst",
        "salinity",
        "wind_speed",
        "current_u",
        "current_v",
        "sea_level",
        "pco2_atm",
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
            merged["current_u"].to_numpy(dtype=float) / 3,
            merged["current_v"].to_numpy(dtype=float) / 3,
            merged["sea_level"].to_numpy(dtype=float) / 2,
            merged["pco2_atm"].to_numpy(dtype=float) / 500,
            np.sin(month_angle.to_numpy(dtype=float)),
            np.cos(month_angle.to_numpy(dtype=float)),
        ]
    ).astype(np.float64)
    targets = merged["target_flux"].to_numpy(dtype=np.float64).reshape(-1, 1)

    atmospheric_lookup = {
        (int(row.year), int(row.month)): float(row.pco2_atm)
        for row in atmospheric.itertuples(index=False)
    }
    observed_coverage = {
        "lat_min": round(float(merged["lat"].min()), 3),
        "lat_max": round(float(merged["lat"].max()), 3),
        "lon_min": round(float(merged["lon"].min()), 3),
        "lon_max": round(float(merged["lon"].max()), 3),
    }
    gridded_coverage = observed_coverage
    if copernicus_monthly is not None and not copernicus_monthly.empty:
        gridded_coverage = {
            "lat_min": round(float(copernicus_monthly["lat_bin"].min()), 3),
            "lat_max": round(float(copernicus_monthly["lat_bin"].max()), 3),
            "lon_min": round(float(copernicus_monthly["lon_bin"].min()), 3),
            "lon_max": round(float(copernicus_monthly["lon_bin"].max()), 3),
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
        "copernicus": copernicus_summary,
        "coverage": gridded_coverage,
        "observed_coverage": observed_coverage,
        "gridded_coverage": gridded_coverage,
        "real_sources_only": True,
        "reference_now": reference_now.astimezone(timezone.utc).isoformat(),
    }

    return RealTrainingDataset(
        features=features,
        targets=targets,
        summary=summary,
        atmospheric_lookup=atmospheric_lookup,
        feature_names=feature_names,
    )

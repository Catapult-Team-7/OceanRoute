from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_COPERNICUS_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "Training_Data" / "Copernicus"
DEFAULT_REPO_ROOT = DEFAULT_COPERNICUS_OUTPUT_DIR.parents[1]
SURFACE_DEPTH_MIN = 0.0
SURFACE_DEPTH_MAX = 0.0


class CopernicusSyncError(RuntimeError):
    pass


@dataclass
class CopernicusSubsetRequest:
    dataset_id: str
    variables: list[str]
    minimum_longitude: float
    maximum_longitude: float
    minimum_latitude: float
    maximum_latitude: float
    start_datetime: str
    end_datetime: str
    output_directory: Path
    output_filename: str
    username: str | None = None
    password: str | None = None
    minimum_depth: float | None = None
    maximum_depth: float | None = None
    overwrite: bool = True
    disable_progress_bar: bool = False
    netcdf3_compatible: bool = True
    coordinates_selection_method: str = "nearest"


def default_copernicus_directory() -> Path:
    return DEFAULT_COPERNICUS_OUTPUT_DIR


def resolve_copernicus_output_directory(path_like: str | Path | None) -> Path:
    if path_like is None or str(path_like).strip() == "":
        return default_copernicus_directory()
    raw = Path(path_like).expanduser()
    return raw if raw.is_absolute() else (DEFAULT_REPO_ROOT / raw)


def parse_copernicus_notes(notes: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    raw = (notes or "").strip()
    if not raw:
        return parsed
    if "=" not in raw:
        parsed["path"] = raw
        return parsed
    for chunk in raw.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def build_default_copernicus_time_window(months: int = 18) -> tuple[str, str]:
    end = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start = (end - timedelta(days=31 * max(months, 1))).replace(day=1)
    return start.strftime("%Y-%m-%dT00:00:00"), end.strftime("%Y-%m-%dT00:00:00")


def resolve_copernicus_credentials() -> tuple[str, str]:
    username = os.getenv("COPERNICUS_USERNAME", "").strip()
    password = os.getenv("COPERNICUS_PASSWORD", "").strip()
    if not username or not password:
        raise CopernicusSyncError(
            "Missing Copernicus credentials. Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in backend/.env."
        )
    return username, password


def _copernicus_client():
    try:
        import copernicusmarine
        import copernicusmarine.download_functions.download_zarr as download_zarr
    except ImportError as exc:
        raise CopernicusSyncError(
            "Missing `copernicusmarine`. Install backend/requirements-ml-ingest.txt to enable live Copernicus downloads."
        ) from exc

    if not getattr(download_zarr, "_oceanpulse_netcdf_patch", False):
        def _oceanpulse_download_dataset_as_netcdf(
            dataset,
            output_path,
            netcdf_compression_level,
            netcdf3_compatible,
        ):
            for coord in dataset.coords:
                dataset[coord].encoding["_FillValue"] = None

            encoding = None
            if netcdf_compression_level and netcdf_compression_level > 0:
                comp = {
                    "zlib": True,
                    "complevel": netcdf_compression_level,
                    "contiguous": False,
                    "shuffle": True,
                }
                keys_to_keep = {"scale_factor", "add_offset", "dtype", "_FillValue", "units"}
                encoding = {
                    name: {
                        **{
                            key: value
                            for key, value in var.encoding.items()
                            if key in keys_to_keep
                        },
                        **comp,
                    }
                    for name, var in dataset.data_vars.items()
                }

            # CopernicusMarine defaults to NETCDF3_CLASSIC when netcdf3_compatible=True,
            # which trips size limits for these subsets. We intentionally write standard
            # NetCDF4 with the netCDF4 engine, which is available in this project env.
            return dataset.to_netcdf(
                output_path,
                mode="w",
                encoding=encoding,
                engine="netcdf4",
            )

        download_zarr._download_dataset_as_netcdf = _oceanpulse_download_dataset_as_netcdf
        download_zarr._oceanpulse_netcdf_patch = True

    return copernicusmarine


def login_copernicus() -> dict[str, Any]:
    copernicusmarine = _copernicus_client()
    username, password = resolve_copernicus_credentials()
    copernicusmarine.login(username=username, password=password, force_overwrite=True)
    return {"username": username}


def subset_copernicus(request: CopernicusSubsetRequest) -> dict[str, Any]:
    copernicusmarine = _copernicus_client()
    output_directory = Path(request.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    username = request.username or resolve_copernicus_credentials()[0]
    password = request.password or resolve_copernicus_credentials()[1]
    subset_kwargs = {
        "dataset_id": request.dataset_id,
        "username": username,
        "password": password,
        "variables": request.variables,
        "minimum_longitude": request.minimum_longitude,
        "maximum_longitude": request.maximum_longitude,
        "minimum_latitude": request.minimum_latitude,
        "maximum_latitude": request.maximum_latitude,
        "start_datetime": request.start_datetime,
        "end_datetime": request.end_datetime,
        "output_directory": str(output_directory),
        "output_filename": request.output_filename,
        "overwrite": request.overwrite,
        "disable_progress_bar": request.disable_progress_bar,
        "netcdf3_compatible": request.netcdf3_compatible,
        "coordinates_selection_method": request.coordinates_selection_method,
    }
    if request.minimum_depth is not None:
        subset_kwargs["minimum_depth"] = request.minimum_depth
    if request.maximum_depth is not None:
        subset_kwargs["maximum_depth"] = request.maximum_depth

    try:
        result = copernicusmarine.subset(
            **subset_kwargs,
        )
    except ImportError as exc:
        raise CopernicusSyncError(
            "Copernicus download started, but local NetCDF export failed because required NetCDF dependencies are "
            "missing. Reinstall backend/requirements-ml-ingest.txt and retry."
        ) from exc
    except Exception as exc:
        raise CopernicusSyncError(str(exc)) from exc
    return {
        "result": str(result),
        "output_file": str(output_directory / request.output_filename),
        "dataset_id": request.dataset_id,
        "variables": request.variables,
    }


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _build_request(
    *,
    dataset_id_env: str,
    default_filename: str,
    variables: list[str],
    months: int = 18,
    overrides: dict[str, str] | None = None,
) -> CopernicusSubsetRequest:
    overrides = overrides or {}
    dataset_override_key_map = {
        "COPERNICUS_CURRENTS_DATASET_ID": "currents_dataset_id",
        "COPERNICUS_SALINITY_DATASET_ID": "salinity_dataset_id",
        "COPERNICUS_TEMPERATURE_DATASET_ID": "temperature_dataset_id",
        "COPERNICUS_SEA_LEVEL_DATASET_ID": "sea_level_dataset_id",
        "COPERNICUS_MONTHLY_PHYSICS_DATASET_ID": "monthly_physics_dataset_id",
        "COPERNICUS_ROUTING_DATASET_ID": "routing_dataset_id",
    }
    dataset_override_key = dataset_override_key_map.get(dataset_id_env, "")
    dataset_id = overrides.get(dataset_override_key, "").strip() or os.getenv(dataset_id_env, "").strip()
    if not dataset_id:
        raise CopernicusSyncError(
            f"Missing {dataset_id_env}. Set the exact Copernicus dataset ID in backend/.env before syncing."
        )
    start_datetime, end_datetime = build_default_copernicus_time_window(months=months)
    username, password = resolve_copernicus_credentials()
    output_directory = resolve_copernicus_output_directory(
        overrides.get("path", "").strip()
        or overrides.get("output_directory", "").strip()
        or os.getenv("COPERNICUS_OUTPUT_DIR", "").strip()
        or default_copernicus_directory()
    )
    use_surface_depth = any(variable in {"uo", "vo", "so", "thetao"} for variable in variables)
    minimum_depth = None
    maximum_depth = None
    if use_surface_depth:
        minimum_depth = float(overrides.get("min_depth", _env_float("COPERNICUS_MIN_DEPTH", SURFACE_DEPTH_MIN)))
        maximum_depth = float(overrides.get("max_depth", _env_float("COPERNICUS_MAX_DEPTH", SURFACE_DEPTH_MAX)))
    return CopernicusSubsetRequest(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=float(overrides.get("min_longitude", _env_float("COPERNICUS_MIN_LONGITUDE", -180.0))),
        maximum_longitude=float(overrides.get("max_longitude", _env_float("COPERNICUS_MAX_LONGITUDE", 180.0))),
        minimum_latitude=float(overrides.get("min_latitude", _env_float("COPERNICUS_MIN_LATITUDE", -80.0))),
        maximum_latitude=float(overrides.get("max_latitude", _env_float("COPERNICUS_MAX_LATITUDE", 80.0))),
        start_datetime=overrides.get("start_datetime", "").strip() or os.getenv("COPERNICUS_START_DATETIME", "").strip() or start_datetime,
        end_datetime=overrides.get("end_datetime", "").strip() or os.getenv("COPERNICUS_END_DATETIME", "").strip() or end_datetime,
        output_directory=output_directory,
        output_filename=overrides.get("output_filename", "").strip() or os.getenv("COPERNICUS_OUTPUT_FILENAME", "").strip() or default_filename,
        username=username,
        password=password,
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        overwrite=overrides.get("overwrite", "").strip().lower() not in {"false", "0", "no"},
        disable_progress_bar=overrides.get("disable_progress_bar", "").strip().lower() in {"true", "1", "yes"},
        netcdf3_compatible=overrides.get("netcdf3_compatible", "").strip().lower() not in {"false", "0", "no"},
        coordinates_selection_method=overrides.get("coordinates_selection_method", "").strip() or "nearest",
    )


def sync_copernicus_currents(months: int = 18, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    request = _build_request(
        dataset_id_env="COPERNICUS_CURRENTS_DATASET_ID",
        default_filename="copernicus_currents.nc",
        variables=["uo", "vo"],
        months=months,
        overrides=overrides,
    )
    return subset_copernicus(request)


def sync_copernicus_salinity(months: int = 18, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    request = _build_request(
        dataset_id_env="COPERNICUS_SALINITY_DATASET_ID",
        default_filename="copernicus_salinity.nc",
        variables=["so"],
        months=months,
        overrides=overrides,
    )
    return subset_copernicus(request)


def sync_copernicus_temperature(months: int = 18, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    request = _build_request(
        dataset_id_env="COPERNICUS_TEMPERATURE_DATASET_ID",
        default_filename="copernicus_temperature.nc",
        variables=["thetao"],
        months=months,
        overrides=overrides,
    )
    return subset_copernicus(request)


def sync_copernicus_sea_level(months: int = 18, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    request = _build_request(
        dataset_id_env="COPERNICUS_SEA_LEVEL_DATASET_ID",
        default_filename="copernicus_sea_level.nc",
        variables=["zos"],
        months=months,
        overrides=overrides,
    )
    return subset_copernicus(request)


def sync_copernicus_monthly_training(months: int = 18, overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    overrides = overrides or {}
    downloads: list[dict[str, Any]] = []
    monthly_dataset_id = overrides.get("monthly_physics_dataset_id", "").strip() or os.getenv(
        "COPERNICUS_MONTHLY_PHYSICS_DATASET_ID", ""
    ).strip()
    if monthly_dataset_id:
        request = _build_request(
            dataset_id_env="COPERNICUS_MONTHLY_PHYSICS_DATASET_ID",
            default_filename="copernicus_monthly_training.nc",
            variables=["uo", "vo", "so", "thetao"],
            months=months,
            overrides={**overrides, "currents_dataset_id": monthly_dataset_id},
        )
        request.dataset_id = monthly_dataset_id
        downloads.append(subset_copernicus(request))
        return downloads

    downloads.append(sync_copernicus_currents(months=months, overrides=overrides))
    downloads.append(sync_copernicus_salinity(months=months, overrides=overrides))
    downloads.append(sync_copernicus_temperature(months=months, overrides=overrides))
    return downloads


def sync_copernicus_routing(months: int = 3, overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    overrides = overrides or {}
    downloads: list[dict[str, Any]] = []
    routing_dataset_id = overrides.get("routing_dataset_id", "").strip() or os.getenv(
        "COPERNICUS_ROUTING_DATASET_ID", ""
    ).strip()
    if routing_dataset_id:
        request = _build_request(
            dataset_id_env="COPERNICUS_ROUTING_DATASET_ID",
            default_filename="copernicus_routing.nc",
            variables=["uo", "vo", "zos"],
            months=months,
            overrides={**overrides, "currents_dataset_id": routing_dataset_id},
        )
        request.dataset_id = routing_dataset_id
        downloads.append(subset_copernicus(request))
        return downloads

    downloads.append(
        sync_copernicus_currents(
            months=months,
            overrides={**overrides, "output_filename": "copernicus_routing_currents.nc"},
        )
    )
    downloads.append(
        sync_copernicus_sea_level(
            months=months,
            overrides={**overrides, "output_filename": "copernicus_routing_sea_level.nc"},
        )
    )
    return downloads


def sync_copernicus_defaults(months: int = 18, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    downloads = []
    downloads.extend(sync_copernicus_monthly_training(months=months, overrides=overrides))
    downloads.extend(sync_copernicus_routing(months=min(months, 3), overrides=overrides))
    return {
        "output_directory": str(
            Path(
                (overrides or {}).get("path", "").strip()
                or (overrides or {}).get("output_directory", "").strip()
                or os.getenv("COPERNICUS_OUTPUT_DIR", "").strip()
                or default_copernicus_directory()
            )
        ),
        "downloads": downloads,
    }

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from ingest.fetch_global_fishing_watch import GlobalFishingWatchError, search_vessels
from ingest.fetch_trash import TrashDataError, load_trash_observations
from ingest.fetch_copernicus import (
    CopernicusSyncError,
    default_copernicus_directory,
    parse_copernicus_notes,
    resolve_copernicus_output_directory,
    resolve_copernicus_credentials,
    sync_copernicus_currents,
    sync_copernicus_salinity,
    sync_copernicus_temperature,
)
from ingest.fetch_world_port_index import DEFAULT_WORLD_PORT_INDEX_URL, fetch_world_ports
from ingest.real_training_data import (
    DEFAULT_NOAA_GML_CO2_URL,
    DEFAULT_SOCAT_ERDDAP_URL,
    RealDataLoadError,
    default_era5_directory,
    load_era5_monthly,
    load_noaa_gml_monthly,
    load_socat_observations,
)

if TYPE_CHECKING:
    from ml.trainer import MLTrainerService


class DataConnectorService:
    def __init__(self, trainer: "MLTrainerService"):
        self.trainer = trainer

    def _set_progress_state(self, *, stage: str, detail: str, progress: float | None = None):
        lock = getattr(self.trainer, "_lock", None)
        state = getattr(self.trainer, "state", None)
        if lock is None or state is None:
            return
        with lock:
            state.metrics = {
                **getattr(state, "metrics", {}),
                "stage": stage,
                "detail": detail,
            }
            if progress is not None:
                state.progress = progress

    def _set_download_progress(self, *, completed: int, total: int, current: str):
        lock = getattr(self.trainer, "_lock", None)
        state = getattr(self.trainer, "state", None)
        if lock is None or state is None:
            return
        with lock:
            state.metrics = {
                **getattr(state, "metrics", {}),
                "download_progress": {
                    "completed": completed,
                    "total": total,
                    "current": current,
                },
            }

    def list_connectors(self):
        return self.trainer.list_required_apis()

    def preview(self, connector_id: str):
        if connector_id == "copernicus_marine":
            return self._preview_copernicus()
        if connector_id == "era5":
            return self._preview_era5()
        if connector_id == "noaa_hycom":
            return self._preview_hycom()
        if connector_id == "noaa_coraltemp":
            return self._preview_erddap_like(
                connector_id=connector_id,
                default_search="coral temp",
                default_url="https://coastwatch.pfeg.noaa.gov/erddap",
            )
        if connector_id == "socat":
            return self._preview_socat()
        if connector_id == "noaa_gml_co2":
            return self._preview_noaa_gml()
        if connector_id == "global_fishing_watch":
            return self._preview_global_fishing_watch()
        if connector_id == "world_port_index":
            return self._preview_world_port_index()
        if connector_id == "emodnet_litter":
            return self._preview_trash_feed(connector_id, default_portal="https://emodnet.ec.europa.eu/en/chemistry")
        if connector_id == "oceanscan":
            return self._preview_trash_feed(connector_id, default_portal="https://www.oceanscan.org")
        return {"status": "error", "message": f"Unknown connector: {connector_id}"}

    def sync(self, connector_id: str):
        if connector_id != "copernicus_marine":
            return {
                "connector_id": connector_id,
                "status": "error",
                "message": f"Sync is not implemented for {connector_id}.",
        }
        config = self._config_for("copernicus_marine")
        note_settings = parse_copernicus_notes(config.get("notes", ""))
        output_directory = resolve_copernicus_output_directory(
            note_settings.get("path", "").strip()
            or os.getenv("COPERNICUS_OUTPUT_DIR", "").strip()
            or default_copernicus_directory()
        )
        before_files = sorted(output_directory.glob("*.nc")) if output_directory.exists() else []
        months_to_sync = max(self.trainer.expected_month_window() + 1, 6)
        self._set_progress_state(
            stage="syncing_copernicus",
            detail=(
                f"Syncing {months_to_sync} months of monthly surface-only Copernicus training files into "
                f"{output_directory} for the configured window."
            ),
            progress=0.05,
        )
        sync_steps = [
            ("currents", sync_copernicus_currents),
            ("salinity", sync_copernicus_salinity),
            ("temperature", sync_copernicus_temperature),
        ]
        downloads = []
        try:
            for index, (label, sync_fn) in enumerate(sync_steps, start=1):
                self._set_progress_state(
                    stage="syncing_copernicus",
                    detail=f"Downloading Copernicus {label} surface grid ({index} of {len(sync_steps)}).",
                    progress=0.05 + index * 0.02,
                )
                self._set_download_progress(
                    completed=index - 1,
                    total=len(sync_steps),
                    current=label,
                )
                downloads.append(sync_fn(months=months_to_sync, overrides=note_settings))
        except CopernicusSyncError as exc:
            self._set_progress_state(
                stage="syncing_copernicus_failed",
                detail=str(exc),
            )
            self._set_download_progress(
                completed=max(0, len(downloads)),
                total=len(sync_steps),
                current="failed",
            )
            return {
                "connector_id": "copernicus_marine",
                "connector_name": config["name"],
                "status": "error",
                "message": str(exc),
                "output_directory": str(output_directory),
                "local_netcdf_count": len(before_files),
                "local_netcdf_files": [str(path) for path in before_files[:12]],
                "config": config,
            }
        after_files = sorted(output_directory.glob("*.nc")) if output_directory.exists() else []
        self._set_progress_state(
            stage="syncing_copernicus_complete",
            detail=f"Copernicus sync complete. {max(0, len(after_files) - len(before_files))} new NetCDF files detected.",
            progress=0.12,
        )
        self._set_download_progress(
            completed=len(sync_steps),
            total=len(sync_steps),
            current="complete",
        )
        return {
            "connector_id": "copernicus_marine",
            "connector_name": config["name"],
            "status": "ready",
            "message": "Copernicus monthly training subsets finished syncing.",
            "months_synced": months_to_sync,
            "downloads": downloads,
            "output_directory": str(output_directory),
            "local_netcdf_count_before": len(before_files),
            "local_netcdf_count": len(after_files),
            "new_files_downloaded": max(0, len(after_files) - len(before_files)),
            "local_netcdf_files": [str(path) for path in after_files[:12]],
            "config": config,
        }

    def _config_for(self, connector_id: str):
        for item in self.trainer.list_required_apis():
            if item.get("id") == connector_id:
                return item
        return None

    def _preview_static(self, connector_id: str, message: str, extra: dict | None = None):
        config = self._config_for(connector_id)
        payload = {
            "connector_id": connector_id,
            "connector_name": config["name"] if config else connector_id,
            "status": "needs_config",
            "message": message,
            "config": config,
        }
        if extra:
            payload.update(extra)
        return payload

    def _preview_copernicus(self):
        config = self._config_for("copernicus_marine")
        note_settings = parse_copernicus_notes(config.get("notes", ""))
        example = {
            "dataset_id_envs": [
                "COPERNICUS_MONTHLY_PHYSICS_DATASET_ID",
                "COPERNICUS_ROUTING_DATASET_ID",
                "COPERNICUS_CURRENTS_DATASET_ID",
                "COPERNICUS_SALINITY_DATASET_ID",
                "COPERNICUS_TEMPERATURE_DATASET_ID",
                "COPERNICUS_SEA_LEVEL_DATASET_ID",
            ],
            "variables": {
                "monthly_training": ["uo", "vo", "so", "thetao"],
                "routing": ["uo", "vo", "zos"],
            },
            "min_longitude": -160,
            "max_longitude": -120,
            "min_latitude": 15,
            "max_latitude": 40,
            "notes_format": "monthly_physics_dataset_id=<id>;routing_dataset_id=<id>;min_longitude=-160;max_longitude=-120;min_latitude=15;max_latitude=40;min_depth=0;max_depth=1;path=Training_Data/Copernicus",
        }
        output_directory = resolve_copernicus_output_directory(
            note_settings.get("path", "").strip()
            or os.getenv("COPERNICUS_OUTPUT_DIR", "").strip()
            or default_copernicus_directory()
        )
        nc_files = sorted(output_directory.glob("*.nc")) if output_directory.exists() else []
        try:
            import copernicusmarine
        except ImportError:
            return {
                "connector_id": "copernicus_marine",
                "connector_name": config["name"],
                "status": "missing_dependency",
                "message": "Install `copernicusmarine` from requirements-ml-ingest.txt to run live Copernicus previews.",
                "request_template": example,
                "local_output_directory": str(output_directory),
                "local_netcdf_files": [str(path) for path in nc_files[:10]],
                "config": config,
            }

        credentials_detected = False
        try:
            resolve_copernicus_credentials()
            credentials_detected = True
        except CopernicusSyncError:
            credentials_detected = False

        return {
            "connector_id": "copernicus_marine",
            "connector_name": config["name"],
            "status": "ready",
            "message": "Copernicus Marine connector is ready for Toolbox-based downloads and local NetCDF enrichment.",
            "toolbox_version": getattr(copernicusmarine, "__version__", "unknown"),
            "credentials_detected": credentials_detected,
            "request_template": example,
            "local_output_directory": str(output_directory),
            "local_netcdf_count": len(nc_files),
            "local_netcdf_files": [str(path) for path in nc_files[:10]],
            "dataset_id_hints": {
                "monthly_training": note_settings.get("monthly_physics_dataset_id", "").strip()
                or os.getenv("COPERNICUS_MONTHLY_PHYSICS_DATASET_ID", "").strip(),
                "routing": note_settings.get("routing_dataset_id", "").strip()
                or os.getenv("COPERNICUS_ROUTING_DATASET_ID", "").strip(),
                "currents": note_settings.get("currents_dataset_id", "").strip()
                or os.getenv("COPERNICUS_CURRENTS_DATASET_ID", "").strip(),
                "salinity": note_settings.get("salinity_dataset_id", "").strip()
                or os.getenv("COPERNICUS_SALINITY_DATASET_ID", "").strip(),
                "temperature": note_settings.get("temperature_dataset_id", "").strip()
                or os.getenv("COPERNICUS_TEMPERATURE_DATASET_ID", "").strip(),
                "sea_level": note_settings.get("sea_level_dataset_id", "").strip()
                or os.getenv("COPERNICUS_SEA_LEVEL_DATASET_ID", "").strip(),
            },
            "config": config,
        }

    def _preview_era5(self):
        config = self._config_for("era5")
        local_path = os.getenv("ERA5_LOCAL_PATH", "").strip() or str(config.get("notes", "")).strip()
        era_directory = Path(local_path) if local_path else default_era5_directory()
        request_template = {
            "dataset": "reanalysis-era5-single-levels-timeseries",
            "variable_count": 17,
            "location": {"longitude": 0, "latitude": 0},
            "date": "2025-01-01/2025-01-31",
            "data_format": "csv",
        }
        try:
            frame = load_era5_monthly(era_directory).head(6)
            return {
                "connector_id": "era5",
                "connector_name": config["name"],
                "status": "ready",
                "message": "Local ERA5 CSV files were found and aggregated into monthly features.",
                "records": frame.fillna("").to_dict(orient="records"),
                "local_path": str(era_directory),
                "config": config,
            }
        except RealDataLoadError:
            pass

        try:
            import cdsapi
        except ImportError:
            return {
                "connector_id": "era5",
                "connector_name": config["name"],
                "status": "missing_dependency",
                "message": "Install `cdsapi` for live ERA5 downloads, or set ERA5_LOCAL_PATH / connector notes to your local CSV folder.",
                "request_template": request_template,
                "local_path": str(era_directory),
                "config": config,
            }

        return {
            "connector_id": "era5",
            "connector_name": config["name"],
            "status": "ready",
            "message": "ERA5 connector is ready. Use your local CSV folder if present, or authenticate via CDS API for remote downloads.",
            "client_type": cdsapi.Client.__name__,
            "credentials_hint": "CDS API typically uses ~/.cdsapirc for authentication.",
            "request_template": request_template,
            "local_path": str(era_directory),
            "config": config,
        }

    def _preview_hycom(self):
        config = self._config_for("noaa_hycom")
        example = {
            "base_url": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0",
            "variables": ["water_u", "water_v"],
            "depth_index": 0,
            "region": {"min_longitude": -160, "max_longitude": -120, "min_latitude": 15, "max_latitude": 40},
        }
        return {
            "connector_id": "noaa_hycom",
            "connector_name": config["name"],
            "status": "ready",
            "message": "HYCOM is available through THREDDS/OPeNDAP rather than ERDDAP. Use the request template below in the ingest connector.",
            "request_template": example,
            "config": config,
        }

    def _preview_world_port_index(self):
        config = self._config_for("world_port_index")
        feature_service = str(config.get("notes", "")).strip() or str(config.get("url", "")).strip() or DEFAULT_WORLD_PORT_INDEX_URL
        if "FeatureServer" not in feature_service:
            return self._preview_static(
                "world_port_index",
                message="Set this connector to a World Port Index ArcGIS FeatureServer endpoint.",
                extra={"feature_service": DEFAULT_WORLD_PORT_INDEX_URL},
            )
        try:
            ports = fetch_world_ports(
                feature_service_url=feature_service,
                min_lat=15,
                max_lat=40,
                min_lon=-160,
                max_lon=-120,
                limit=5,
            )
            return {
                "connector_id": "world_port_index",
                "connector_name": config["name"],
                "status": "ready",
                "message": "World Port Index sample loaded successfully.",
                "records": [port.__dict__ for port in ports],
                "feature_service": feature_service,
                "config": config,
            }
        except Exception as exc:
            return {
                "connector_id": "world_port_index",
                "connector_name": config["name"],
                "status": "error",
                "message": str(exc),
                "feature_service": feature_service,
                "config": config,
            }

    def _preview_global_fishing_watch(self):
        config = self._config_for("global_fishing_watch")
        try:
            sample = search_vessels("san", limit=2)
            return {
                "connector_id": "global_fishing_watch",
                "connector_name": config["name"],
                "status": "ready",
                "message": "Global Fishing Watch token is configured and vessel search is working.",
                "records": sample.get("entries", [])[:2],
                "api_docs": ["https://globalfishingwatch.org/our-apis/documentation"],
                "config": config,
            }
        except GlobalFishingWatchError as exc:
            return {
                "connector_id": "global_fishing_watch",
                "connector_name": config["name"],
                "status": "needs_config",
                "message": str(exc),
                "api_docs": ["https://globalfishingwatch.org/our-apis/documentation"],
                "config": config,
            }
        except Exception as exc:
            return {
                "connector_id": "global_fishing_watch",
                "connector_name": config["name"],
                "status": "error",
                "message": str(exc),
                "api_docs": ["https://globalfishingwatch.org/our-apis/documentation"],
                "config": config,
            }

    def _preview_trash_feed(self, connector_id: str, default_portal: str):
        config = self._config_for(connector_id)
        data_url = str(config.get("notes", "")).strip() or str(config.get("url", "")).strip()
        if data_url == default_portal or not data_url:
            return self._preview_static(
                connector_id,
                message="Configure this connector with a direct JSON, GeoJSON, or CSV data URL to preview measured trash observations.",
                extra={"portal": default_portal},
            )
        try:
            records = load_trash_observations(data_url, limit=5)
            return {
                "connector_id": connector_id,
                "connector_name": config["name"],
                "status": "ready",
                "message": "Trash observation sample loaded successfully.",
                "records": records,
                "config": config,
            }
        except TrashDataError as exc:
            return {
                "connector_id": connector_id,
                "connector_name": config["name"],
                "status": "error",
                "message": str(exc),
                "config": config,
            }

    def _preview_erddap_like(self, connector_id: str, default_search: str, default_url: str):
        config = self._config_for(connector_id)
        base_url = (config.get("url") or default_url).rstrip("/")
        search_term = config.get("notes") or default_search
        if "erddap" not in base_url:
            return {
                "connector_id": connector_id,
                "connector_name": config["name"],
                "status": "needs_config",
                "message": "Set this connector URL to an ERDDAP server root to search live datasets.",
                "config": config,
                "example_url": "https://coastwatch.pfeg.noaa.gov/erddap",
            }
        search_url = f"{base_url}/search/index.json?page=1&itemsPerPage=5&searchFor={search_term.replace(' ', '+')}"
        try:
            response = requests.get(search_url, timeout=20)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("table", {}).get("rows", [])
            preview_rows = rows[:5]
            return {
                "connector_id": connector_id,
                "connector_name": config["name"],
                "status": "ready",
                "message": f"Found {len(rows)} matching ERDDAP dataset rows.",
                "search_url": search_url,
                "rows": preview_rows,
                "config": config,
            }
        except Exception as exc:
            return {
                "connector_id": connector_id,
                "connector_name": config["name"],
                "status": "error",
                "message": str(exc),
                "search_url": search_url,
                "config": config,
            }

    def _preview_socat(self):
        config = self._config_for("socat")
        data_url = config.get("url", "").strip()
        if "/erddap/tabledap/" in data_url:
            query_hint = (
                "Use the SOCAT ERDDAP tabledap endpoint. If you provide only the dataset root, "
                "OceanPulse will auto-build a lightweight decimated query."
            )
        else:
            query_hint = None
        if not data_url.endswith((".zip", ".tsv", ".csv", ".txt")) and "/erddap/tabledap/" not in data_url:
            return {
                "connector_id": "socat",
                "connector_name": config["name"],
                "status": "needs_config",
                "message": "Set a SOCAT ERDDAP tabledap URL or a direct SOCAT release URL ending in .zip, .tsv, .txt, or .csv before previewing.",
                "config": config,
                "example_url": DEFAULT_SOCAT_ERDDAP_URL,
            }
        try:
            frame = load_socat_observations(data_url, sample_size=5)
            return {
                "connector_id": "socat",
                "connector_name": config["name"],
                "status": "ready",
                "message": "SOCAT sample loaded successfully.",
                "columns": list(frame.columns),
                "records": frame.fillna("").to_dict(orient="records"),
                "query_hint": query_hint,
                "config": config,
            }
        except RealDataLoadError as exc:
            return {
                "connector_id": "socat",
                "connector_name": config["name"],
                "status": "error",
                "message": str(exc),
                "config": config,
            }

    def _preview_noaa_gml(self):
        config = self._config_for("noaa_gml_co2")
        data_url = config.get("url", "").strip() or DEFAULT_NOAA_GML_CO2_URL
        if not data_url.endswith((".csv", ".txt")):
            return {
                "connector_id": "noaa_gml_co2",
                "connector_name": config["name"],
                "status": "needs_config",
                "message": "Set a direct NOAA GML monthly CO2 URL ending in .txt or .csv before previewing.",
                "config": config,
            }
        try:
            frame = load_noaa_gml_monthly(data_url).head(5)
            return {
                "connector_id": "noaa_gml_co2",
                "connector_name": config["name"],
                "status": "ready",
                "message": "NOAA GML sample loaded successfully.",
                "columns": list(frame.columns),
                "records": frame.fillna("").to_dict(orient="records"),
                "config": config,
            }
        except RealDataLoadError as exc:
            return {
                "connector_id": "noaa_gml_co2",
                "connector_name": config["name"],
                "status": "error",
                "message": str(exc),
                "config": config,
            }

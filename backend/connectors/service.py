from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from ingest.real_training_data import (
    DEFAULT_NOAA_GML_CO2_URL,
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
        if connector_id == "nasa_ocean_color":
            return self._preview_static(
                connector_id,
                message="NASA Ocean Color requires a product selection and often Earthdata-authenticated downloads. Configure the product URL first.",
            )
        return {"status": "error", "message": f"Unknown connector: {connector_id}"}

    def _config_for(self, connector_id: str):
        for item in self.trainer.list_required_apis():
            if item.get("id") == connector_id:
                return item
        return None

    def _preview_static(self, connector_id: str, message: str):
        config = self._config_for(connector_id)
        return {
            "connector_id": connector_id,
            "connector_name": config["name"] if config else connector_id,
            "status": "needs_config",
            "message": message,
            "config": config,
        }

    def _preview_copernicus(self):
        config = self._config_for("copernicus_marine")
        example = {
            "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            "variables": ["so", "thetao"],
            "min_longitude": -160,
            "max_longitude": -120,
            "min_latitude": 15,
            "max_latitude": 40,
        }
        try:
            import copernicusmarine
        except ImportError:
            return {
                "connector_id": "copernicus_marine",
                "connector_name": config["name"],
                "status": "missing_dependency",
                "message": "Install `copernicusmarine` from requirements-ml-ingest.txt to run live Copernicus previews.",
                "request_template": example,
                "config": config,
            }

        return {
            "connector_id": "copernicus_marine",
            "connector_name": config["name"],
            "status": "ready",
            "message": "Copernicus Marine connector is configured for Toolbox-based downloads. Use the example request below in the next ingest step.",
            "toolbox_version": getattr(copernicusmarine, "__version__", "unknown"),
            "credentials_detected": bool(os.getenv("COPERNICUS_USERNAME") and os.getenv("COPERNICUS_PASSWORD")),
            "request_template": example,
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
        if not data_url.endswith((".zip", ".tsv", ".csv", ".txt")):
            return {
                "connector_id": "socat",
                "connector_name": config["name"],
                "status": "needs_config",
                "message": "Set a direct SOCAT release URL ending in .zip, .tsv, .txt, or .csv before previewing.",
                "config": config,
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

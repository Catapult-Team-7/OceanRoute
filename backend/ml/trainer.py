from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pipeline.artifacts import write_data_manifest, write_training_manifest
from ingest.fetch_copernicus import (
    CopernicusSyncError,
    default_copernicus_directory,
    parse_copernicus_notes,
    sync_copernicus_currents,
    sync_copernicus_monthly_training,
    sync_copernicus_salinity,
    sync_copernicus_temperature,
)
from ingest.real_training_data import (
    DEFAULT_SOCAT_ERDDAP_URL,
    RealDataLoadError,
    build_real_training_dataset,
    default_era5_directory,
)

TORCH_IMPORT_ERROR = None
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
    from ml.model import OceanPulseLSTM, masked_huber_loss, physics_consistency_penalty
    from ml.preprocess import build_monthly_training_tensors
except Exception as exc:  # pragma: no cover - dependency availability branch
    torch = None
    DataLoader = None
    Dataset = object
    Subset = None
    OceanPulseLSTM = None
    masked_huber_loss = None
    physics_consistency_penalty = None
    build_monthly_training_tensors = None
    TORCH_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from db.demo_data import DemoOceanRepository


logger = logging.getLogger("oceanpulse.ml")
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "oceanpulse_latest.pt"
RESUME_CHECKPOINT_PATH = CHECKPOINT_DIR / "oceanpulse_resume.pt"
AUTOSAVE_EPOCH_INTERVAL = 10
DEFAULT_EXTERNAL_ERA5_PATH = "~/OceanPulseData/ERA"
RECOMMENDED_COPERNICUS_CURRENTS_DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m"
RECOMMENDED_COPERNICUS_SALINITY_DATASET = "cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m"
RECOMMENDED_COPERNICUS_TEMPERATURE_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m"
GLOBAL_COPERNICUS_REGION_NOTES = "min_longitude=-180;max_longitude=180;min_latitude=-80;max_latitude=80"
SURFACE_COPERNICUS_DEPTH_NOTES = "min_depth=0;max_depth=1"
DEFAULT_COPERNICUS_NOTES = (
    f"currents_dataset_id={RECOMMENDED_COPERNICUS_CURRENTS_DATASET};"
    f"salinity_dataset_id={RECOMMENDED_COPERNICUS_SALINITY_DATASET};"
    f"temperature_dataset_id={RECOMMENDED_COPERNICUS_TEMPERATURE_DATASET};"
    f"{GLOBAL_COPERNICUS_REGION_NOTES};"
    f"{SURFACE_COPERNICUS_DEPTH_NOTES};"
    "path=Training_Data/Copernicus"
)


def _smooth_prediction_grid(values: np.ndarray, passes: int = 2) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float32).copy()
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
        padded = np.pad(smoothed, ((1, 1), (1, 1)), mode="edge")
        next_grid = np.zeros_like(smoothed, dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                next_grid += padded[di : di + smoothed.shape[0], dj : dj + smoothed.shape[1]] * kernel[di, dj]
        smoothed = next_grid
    return smoothed


REAL_DATA_APIS = [
    {
        "id": "noaa_hycom",
        "name": "NOAA HYCOM",
        "purpose": "Ocean current vectors for advection-aware routing features.",
        "status": "planned",
        "enabled": False,
        "fields": ["water_u", "water_v", "surface currents"],
        "url": "https://tds.hycom.org",
        "env_var": "NOAA_HYCOM_URL",
        "notes": "",
    },
    {
        "id": "noaa_coraltemp",
        "name": "NOAA CoralTemp",
        "purpose": "Sea surface temperature for carbon exchange and route risk context.",
        "status": "planned",
        "enabled": False,
        "fields": ["sst", "daily thermal stress"],
        "url": "https://coralreefwatch.noaa.gov",
        "env_var": "NOAA_CORALTEMP_URL",
        "notes": "",
    },
    {
        "id": "copernicus_marine",
        "name": "Copernicus Marine",
        "purpose": "Salinity and marine state variables for flux estimation.",
        "status": "testing",
        "enabled": True,
        "fields": ["salinity", "surface currents", "biogeochemical grids"],
        "url": "https://marine.copernicus.eu",
        "env_var": "COPERNICUS_MARINE_URL",
        "notes": DEFAULT_COPERNICUS_NOTES,
    },
    {
        "id": "global_fishing_watch",
        "name": "Global Fishing Watch",
        "purpose": "AIS vessel presence, port visits, and vessel identity for route feasibility and supervision.",
        "status": "planned",
        "enabled": False,
        "fields": ["vessel presence", "port visits", "identity", "AIS gaps"],
        "url": "https://globalfishingwatch.org/our-apis/documentation",
        "env_var": "GLOBAL_FISHING_WATCH_URL",
        "notes": "",
    },
    {
        "id": "world_port_index",
        "name": "NGA World Port Index",
        "purpose": "Real port locations and metadata for route targets and nearest-port logic.",
        "status": "planned",
        "enabled": False,
        "fields": ["port name", "country", "coordinates", "harbor metadata"],
        "url": "https://vcps.nga.mil/nauticalpubs-feature/rest/services/WPI/World_Port_Index_Viewer/FeatureServer",
        "env_var": "WORLD_PORT_INDEX_URL",
        "notes": "",
    },
    {
        "id": "emodnet_litter",
        "name": "EMODnet Chemistry / Litter",
        "purpose": "Marine litter observations for replacing modeled recovery targets with measured debris context.",
        "status": "planned",
        "enabled": False,
        "fields": ["marine litter observations", "survey metadata"],
        "url": "https://emodnet.ec.europa.eu/en/chemistry",
        "env_var": "EMODNET_LITTER_URL",
        "notes": "",
    },
    {
        "id": "oceanscan",
        "name": "OceanScan",
        "purpose": "Additional marine debris and ocean monitoring products for trash accumulation evidence.",
        "status": "planned",
        "enabled": False,
        "fields": ["debris observations", "ocean monitoring products"],
        "url": "https://www.oceanscan.org",
        "env_var": "OCEANSCAN_URL",
        "notes": "",
    },
    {
        "id": "era5",
        "name": "ERA5",
        "purpose": "Wind forcing for gas transfer velocity and routing conditions.",
        "status": "testing",
        "enabled": True,
        "fields": ["10m wind", "surface pressure", "wave-relevant forcing"],
        "url": "https://cds.climate.copernicus.eu",
        "env_var": "ERA5_API_URL",
        "notes": DEFAULT_EXTERNAL_ERA5_PATH,
    },
    {
        "id": "socat",
        "name": "SOCAT",
        "purpose": "Ship-based pCO2 observations for target construction and validation.",
        "status": "connected",
        "enabled": True,
        "fields": ["time", "latitude", "longitude", "temp", "sal", "surface ocean fCO2"],
        "url": DEFAULT_SOCAT_ERDDAP_URL,
        "env_var": "SOCAT_DATA_URL",
        "notes": "ERDDAP live decimated dataset",
    },
    {
        "id": "noaa_gml_co2",
        "name": "NOAA GML CO2",
        "purpose": "Atmospheric pCO2 baseline feature.",
        "status": "connected",
        "enabled": True,
        "fields": ["monthly atmospheric CO2"],
        "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
        "env_var": "NOAA_GML_CO2_URL",
        "notes": "",
    },
]


def _normalize_copernicus_training_notes(notes: dict[str, str]) -> dict[str, str]:
    normalized = dict(notes)
    monthly_dataset_id = normalized.get("monthly_physics_dataset_id", "").strip()
    if "PT1H" in monthly_dataset_id or "PT6H" in monthly_dataset_id:
        normalized.pop("monthly_physics_dataset_id", None)

    normalized.setdefault(
        "currents_dataset_id",
        os.getenv("COPERNICUS_CURRENTS_DATASET_ID", "").strip() or RECOMMENDED_COPERNICUS_CURRENTS_DATASET,
    )
    normalized.setdefault(
        "salinity_dataset_id",
        os.getenv("COPERNICUS_SALINITY_DATASET_ID", "").strip() or RECOMMENDED_COPERNICUS_SALINITY_DATASET,
    )
    normalized.setdefault(
        "temperature_dataset_id",
        os.getenv("COPERNICUS_TEMPERATURE_DATASET_ID", "").strip() or RECOMMENDED_COPERNICUS_TEMPERATURE_DATASET,
    )
    normalized.setdefault("path", "Training_Data/Copernicus")
    return normalized


@dataclass
class TrainingState:
    status: str = "idle"
    run_id: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    loss_history: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    error: str | None = None
    model_ready: bool = False
    model_summary: dict = field(default_factory=dict)
    data_summary: dict = field(default_factory=dict)

    def snapshot(self):
        return {
            "status": self.status,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": round(self.progress, 3),
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss_history": self.loss_history[-30:],
            "metrics": self.metrics,
            "config": self.config,
            "error": self.error,
            "model_ready": self.model_ready,
            "model_summary": self.model_summary,
            "data_summary": self.data_summary,
        }


class MLTrainerService:
    def __init__(self, repo: "DemoOceanRepository"):
        self.repo = repo
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state = TrainingState(
            config={"epochs": 24, "learning_rate": 0.005, "month_window": 12, "resolution": "2deg"},
            
            model_summary={
                "mode": "real_monthly_convlstm",
                "target": "co2_flux",
                "features": [
                    "lat_norm",
                    "lon_norm",
                    "thetao",
                    "salinity",
                    "wind_speed",
                    "current_u",
                    "current_v",
                    "sea_level",
                    "pco2_atm",
                    "month_sin",
                    "month_cos",
                ],
            },
            data_summary={
                "source": "unconfigured_real_pipeline",
                "real_data_ready": False,
                "required_connectors": ["socat", "noaa_gml_co2", "era5"],
            },
        )
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.atmospheric_lookup: dict[tuple[int, int], float] = {}
        self._spatial_model = None
        self._spatial_model_meta = None
        self._spatial_model_cache_key = None
        self.api_registry = []
        for item in REAL_DATA_APIS:
            configured = dict(item)
            env_value = os.getenv(configured["env_var"], "").strip()
            if env_value:
                configured["url"] = env_value
                configured["enabled"] = True
                configured["status"] = "connected"
            if configured["id"] == "era5" and default_era5_directory().exists():
                configured["enabled"] = True
                configured["status"] = "connected"
                configured["notes"] = str(default_era5_directory())
            if configured["id"] == "copernicus_marine" and default_copernicus_directory().exists():
                configured["enabled"] = True
                configured["status"] = "connected"
                configured["notes"] = DEFAULT_COPERNICUS_NOTES
            self.api_registry.append(configured)
        self._bootstrap_from_checkpoint()

    def _bootstrap_from_checkpoint(self):
        checkpoint_path = LATEST_CHECKPOINT_PATH
        if not checkpoint_path.exists() or torch is None:
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            logger.exception("ml_checkpoint_bootstrap_failed path=%s", checkpoint_path)
            return

        self.state.model_ready = True
        self.state.model_summary = {
            "mode": "real_monthly_convlstm",
            "target": "co2_flux",
            "features": checkpoint.get("feature_names", self.state.model_summary.get("features", [])),
            "checkpoint_path": str(checkpoint_path),
        }
        self.state.data_summary = {
            **self.state.data_summary,
            **checkpoint.get("data_summary", {}),
        }
        self.state.metrics = {
            **self.state.metrics,
            "stage": "checkpoint_loaded",
            "detail": f"Loaded existing checkpoint from {checkpoint_path}.",
            "checkpoint_path": str(checkpoint_path),
        }
        if RESUME_CHECKPOINT_PATH.exists():
            self.state.metrics = {
                **self.state.metrics,
                "resume_checkpoint_path": str(RESUME_CHECKPOINT_PATH),
                "resume_available": True,
            }

    def list_required_apis(self):
        return [dict(item) for item in self.api_registry]

    def _merge_training_config(self, config: dict | None = None) -> dict:
        return {
            "epochs": int((config or {}).get("epochs", self.state.config.get("epochs", 24))),
            "learning_rate": float((config or {}).get("learning_rate", self.state.config.get("learning_rate", 0.005))),
            "month_window": int((config or {}).get("month_window", self.state.config.get("month_window", 12))),
            "resolution": str((config or {}).get("resolution", self.state.config.get("resolution", "2deg"))),
            "quick_test": bool((config or {}).get("quick_test", False)),
        }

    def _initialize_training_state(self, merged_config: dict):
        self.state = TrainingState(
            status="running",
            run_id=self.state.run_id + 1,
            started_at=datetime.now(timezone.utc).isoformat(),
            progress=0.0,
            current_epoch=0,
            total_epochs=merged_config["epochs"],
            config=merged_config,
            model_summary=self.state.model_summary,
            data_summary=self.state.data_summary,
        )
        write_training_manifest(
            {
                "status": "running",
                "run_id": self.state.run_id,
                "config": merged_config,
                "current_epoch": 0,
                "total_epochs": merged_config["epochs"],
                "checkpoint_path": str(LATEST_CHECKPOINT_PATH) if LATEST_CHECKPOINT_PATH.exists() else None,
                "resume_checkpoint_path": str(RESUME_CHECKPOINT_PATH) if RESUME_CHECKPOINT_PATH.exists() else None,
            }
        )

    def _config_by_id(self, connector_id: str):
        for item in self.api_registry:
            if item["id"] == connector_id:
                return item
        return None

    def _real_training_ready(self):
        required_ids = ("socat", "noaa_gml_co2")
        ready = True
        connectors = {}
        for connector_id in required_ids:
            config = self._config_by_id(connector_id) or {}
            enabled = bool(config.get("enabled"))
            has_url = bool(str(config.get("url", "")).strip())
            connectors[connector_id] = {
                "enabled": enabled,
                "configured_url": has_url,
                "status": config.get("status", "planned"),
            }
            ready = ready and enabled and has_url
        return ready, connectors

    def update_api_configs(self, updates: list[dict]):
        indexed = {item["id"]: dict(item) for item in self.api_registry}
        for update in updates:
            update_id = update.get("id")
            if not update_id and update.get("name"):
                update_id = next(
                    (item["id"] for item in self.api_registry if item.get("name") == update.get("name")),
                    None,
                )
            if update_id not in indexed:
                continue
            indexed[update_id].update(
                {
                    "enabled": bool(update.get("enabled", indexed[update_id]["enabled"])),
                    "status": str(update.get("status", indexed[update_id]["status"])),
                    "url": str(update.get("url", indexed[update_id]["url"])),
                    "notes": str(update.get("notes", indexed[update_id].get("notes", ""))),
                }
            )
        self.api_registry = list(indexed.values())
        return self.list_required_apis()

    def get_status(self):
        with self._lock:
            return self.state.snapshot()

    def expected_month_window(self) -> int:
        checkpoint_path = LATEST_CHECKPOINT_PATH
        if checkpoint_path.exists() and torch is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                return int(checkpoint.get("month_window", self.state.config.get("month_window", 12)))
            except Exception:
                return int(self.state.config.get("month_window", 12))
        return int(self.state.config.get("month_window", 12))

    def _load_spatial_model(self):
        if torch is None or OceanPulseLSTM is None:
            raise RealDataLoadError(
                "PyTorch spatial inference dependencies are not installed. Run `pip install -r backend/requirements-ml-ingest.txt`."
            )
        checkpoint_path = Path(self.state.model_summary.get("checkpoint_path") or LATEST_CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            raise RealDataLoadError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        cache_key = (str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
        if self._spatial_model is not None and self._spatial_model_cache_key == cache_key:
            return self._spatial_model, self._spatial_model_meta

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        feature_names = checkpoint.get("feature_names", self.state.model_summary.get("features", []))
        model = OceanPulseLSTM(in_channels=len(feature_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._spatial_model = model
        self._spatial_model_meta = checkpoint
        self._spatial_model_cache_key = cache_key
        return model, checkpoint

    def _save_model_checkpoint(
        self,
        *,
        model_state_dict: dict,
        feature_names: list[str],
        resolution: str,
        month_window: int,
        data_summary: dict,
        epoch: int | None = None,
    ) -> Path:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": model_state_dict,
            "feature_names": feature_names,
            "resolution": resolution,
            "month_window": month_window,
            "data_summary": data_summary,
        }
        if epoch is not None:
            payload["epoch"] = int(epoch)
        torch.save(payload, LATEST_CHECKPOINT_PATH)
        return LATEST_CHECKPOINT_PATH

    def _save_resume_checkpoint(
        self,
        *,
        epoch: int,
        total_epochs: int,
        config: dict,
        feature_names: list[str],
        data_summary: dict,
        model_state_dict: dict,
        optimizer_state_dict: dict,
        scheduler_state_dict: dict | None,
        best_state_dict: dict | None,
        best_val: float,
        stagnant_epochs: int,
        loss_history: list[float],
    ) -> Path:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": int(epoch),
                "total_epochs": int(total_epochs),
                "config": dict(config),
                "feature_names": list(feature_names),
                "data_summary": dict(data_summary),
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "scheduler_state_dict": scheduler_state_dict,
                "best_model_state_dict": best_state_dict,
                "best_val": float(best_val),
                "stagnant_epochs": int(stagnant_epochs),
                "loss_history": list(loss_history),
            },
            RESUME_CHECKPOINT_PATH,
        )
        return RESUME_CHECKPOINT_PATH

    def _load_resume_checkpoint(self):
        if torch is None or not RESUME_CHECKPOINT_PATH.exists():
            return None
        try:
            return torch.load(RESUME_CHECKPOINT_PATH, map_location="cpu")
        except Exception:
            logger.exception("ml_resume_checkpoint_load_failed path=%s", RESUME_CHECKPOINT_PATH)
            return None

    def predict_spatial_sequence(self, feature_sequence: np.ndarray, atmospheric_co2: float) -> np.ndarray:
        model, _ = self._load_spatial_model()
        with torch.no_grad():
            x_tensor = torch.tensor(feature_sequence[None, ...], dtype=torch.float32)
            atm_tensor = torch.tensor([[atmospheric_co2]], dtype=torch.float32)
            prediction = model(x_tensor, atm_tensor)
        return _smooth_prediction_grid(prediction[0, 0].detach().cpu().numpy(), passes=2)

    def start_training(self, config: dict | None = None):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.state.snapshot(), False
            merged_config = self._merge_training_config(config)
            self._initialize_training_state(merged_config)
            self._thread = threading.Thread(target=self._train_loop, args=(merged_config,), daemon=True)
            self._thread.start()
            return self.state.snapshot(), True

    def run_training_job(self, config: dict | None = None):
        merged_config = self._merge_training_config(config)
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Cannot run a blocking training job while background training is already active.")
            self._initialize_training_state(merged_config)
        self._train_loop(merged_config)
        with self._lock:
            return self.state.snapshot()

    def _build_dataset(self, month_window: int, resolution: str, quick_test: bool = False):
        real_ready, connector_state = self._real_training_ready()
        if not real_ready:
            raise RealDataLoadError(
                "Real training requires enabled SOCAT and NOAA GML connectors with valid source URLs."
            )

        socat_config = self._config_by_id("socat")
        noaa_config = self._config_by_id("noaa_gml_co2")
        era5_config = self._config_by_id("era5")
        era5_path = os.getenv("ERA5_LOCAL_PATH", "").strip() or None
        copernicus_path = os.getenv("COPERNICUS_LOCAL_PATH", "").strip() or None
        if era5_config and era5_config.get("enabled"):
            era5_path = str(era5_config.get("notes", "")).strip() or era5_path or str(default_era5_directory())
        elif not era5_path and default_era5_directory().exists():
            era5_path = str(default_era5_directory())
        copernicus_config = self._config_by_id("copernicus_marine")
        copernicus_notes = parse_copernicus_notes((copernicus_config or {}).get("notes", ""))
        if copernicus_config and copernicus_config.get("enabled"):
            copernicus_path = (
                copernicus_notes.get("path", "").strip()
                or copernicus_path
                or str(default_copernicus_directory())
            )
        elif not copernicus_path and default_copernicus_directory().exists():
            copernicus_path = str(default_copernicus_directory())

        real_dataset = build_real_training_dataset(
            socat_url=socat_config["url"],
            noaa_gml_url=noaa_config["url"],
            month_window=month_window,
            reference_now=self.repo.now,
            era_directory=era5_path,
            copernicus_directory=copernicus_path,
            sample_size=4000 if quick_test else 12000,
        )
        self.atmospheric_lookup = dict(real_dataset.atmospheric_lookup)
        summary = {
            **real_dataset.summary,
            "real_data_ready": True,
            "connector_state": connector_state,
        }
        return real_dataset.features, real_dataset.targets, summary

    def _feature_vector(self, row):
        month_angle = 2 * np.pi * row.timestamp.month / 12
        current_u = float(getattr(row, "current_u", 0.0) or 0.0)
        current_v = float(getattr(row, "current_v", 0.0) or 0.0)
        pco2_atm = float(self.atmospheric_lookup.get((row.timestamp.year, row.timestamp.month), 0.0))
        sea_level = float(getattr(row, "sea_level", 0.0) or 0.0)
        return np.array(
            [
                row.lat / 90,
                row.lon / 180,
                row.sst / 30,
                row.salinity / 40,
                row.wind_speed / 20,
                current_u / 3,
                current_v / 3,
                sea_level / 2,
                pco2_atm / 500,
                np.sin(month_angle),
                np.cos(month_angle),
            ],
            dtype=np.float64,
        )

    def score_row(self, row):
        observed_flux = float(row.co2_flux)
        predicted_flux = observed_flux
        if self.state.model_ready and self.weights is not None and self.feature_mean is not None and self.feature_std is not None:
            features = self._feature_vector(row).reshape(1, -1)
            normalized = (features - self.feature_mean) / self.feature_std
            predicted_flux = float((normalized @ self.weights + self.bias).item())

        weakening_score = max(0.0, observed_flux - predicted_flux)
        route_priority = max(0.0, (weakening_score * 1.8) + row.anomaly_score + (row.wind_speed / 20))
        return {
            "observed_flux": round(observed_flux, 4),
            "predicted_flux": round(predicted_flux, 4),
            "weakening_score": round(weakening_score, 4),
            "route_priority": round(min(route_priority, 5.0), 4),
            "inference_mode": "trained_regression" if self.state.model_ready else "observed_demo_baseline",
        }

    def _train_loop(self, config: dict):
        logger.info("ml_training_started run_id=%s config=%s", self.state.run_id, config)
        try:
            if TORCH_IMPORT_ERROR is not None:
                raise RealDataLoadError(
                    "PyTorch ML dependencies are not installed. Run `pip install -r backend/requirements-ml-ingest.txt` in backend/.venv."
                )
            with self._lock:
                self.state.metrics = {
                    "stage": "loading_real_observations",
                    "detail": "Loading SOCAT, NOAA GML, and configured real-source connectors.",
                }
            X, y, data_summary = self._build_dataset(
                config["month_window"], config["resolution"], config.get("quick_test", False)
            )
            with self._lock:
                self.state.data_summary = data_summary
                self.state.metrics = {
                    **self.state.metrics,
                    "stage": "building_monthly_tensors",
                    "detail": "Aligning monthly ERA5 and Copernicus grids into ConvLSTM tensors.",
                    "samples": int(len(X)),
                }
            copernicus_config = self._config_by_id("copernicus_marine")
            copernicus_notes = _normalize_copernicus_training_notes(
                parse_copernicus_notes((copernicus_config or {}).get("notes", ""))
            )
            copernicus_path = (
                copernicus_notes.get("path", "").strip()
                or os.getenv("COPERNICUS_LOCAL_PATH", "").strip()
                or str(default_copernicus_directory())
            )
            copernicus_dir = Path(copernicus_path)
            if not copernicus_dir.is_absolute():
                copernicus_dir = default_copernicus_directory().parents[1] / copernicus_dir
            if copernicus_config and copernicus_config.get("enabled") and not copernicus_dir.exists():
                if config.get("quick_test", False):
                    raise RealDataLoadError(
                        "Quick validation mode will not wait on a live Copernicus sync. "
                        f"Please sync Copernicus first so monthly NetCDF files exist in {copernicus_dir}, "
                        "or switch back to full training mode."
                    )
                sync_months = (
                    max(4, min(config["month_window"] + 1, 6))
                    if config.get("quick_test", False)
                    else max(6, min(config["month_window"] + 2, 15))
                )
                with self._lock:
                    self.state.metrics = {
                        **self.state.metrics,
                        "stage": "syncing_copernicus",
                        "detail": (
                            f"Copernicus directory missing. Syncing monthly training grids only "
                            f"for the last {sync_months} months into {copernicus_dir}."
                        ),
                    }
                try:
                    sync_steps = [
                        ("currents", sync_copernicus_currents),
                        ("salinity", sync_copernicus_salinity),
                        ("temperature", sync_copernicus_temperature),
                    ]
                    for step_index, (label, sync_fn) in enumerate(sync_steps, start=1):
                        with self._lock:
                            self.state.progress = 0.1 + (step_index - 1) * 0.06
                            self.state.metrics = {
                                **self.state.metrics,
                                "stage": "syncing_copernicus",
                                "detail": f"Downloading Copernicus {label} grid ({step_index} of {len(sync_steps)}).",
                                "download_progress": {
                                    "completed": step_index - 1,
                                    "total": len(sync_steps),
                                    "current": label,
                                },
                            }
                        sync_fn(months=sync_months, overrides=copernicus_notes)
                    with self._lock:
                        self.state.progress = 0.28
                        self.state.metrics = {
                            **self.state.metrics,
                            "stage": "syncing_copernicus",
                            "detail": f"Downloaded {len(sync_steps)} Copernicus monthly grids into {copernicus_dir}.",
                            "download_progress": {
                                "completed": len(sync_steps),
                                "total": len(sync_steps),
                                "current": "complete",
                            },
                        }
                except CopernicusSyncError as exc:
                    raise RealDataLoadError(
                        f"Copernicus data is required for tensor training and live sync failed: {exc}"
                    ) from exc
            tensor_build = build_monthly_training_tensors(
                socat_url=self._config_by_id("socat")["url"],
                noaa_gml_url=self._config_by_id("noaa_gml_co2")["url"],
                era_directory=(
                    str(self._config_by_id("era5").get("notes", "")).strip() or os.getenv("ERA5_LOCAL_PATH", "").strip()
                ),
                copernicus_directory=str(copernicus_dir),
                month_window=config["month_window"],
                resolution=config["resolution"],
                reference_now=self.repo.now,
            )
            write_data_manifest(
                {
                    "config": config,
                    "dataset": {
                        "tabular_samples": int(len(X)),
                        "target_samples": int(len(y)),
                        **data_summary,
                    },
                    "tensor_build": {
                        "tensor_dir": str(tensor_build.tensor_dir),
                        "x_path": str(tensor_build.x_path),
                        "y_path": str(tensor_build.y_path),
                        "mask_path": str(tensor_build.mask_path),
                        "atm_path": str(tensor_build.atm_path),
                        "metadata_path": str(tensor_build.metadata_path),
                        **tensor_build.summary,
                    },
                }
            )
            with self._lock:
                self.state.data_summary = {**data_summary, **tensor_build.summary}
                self.state.metrics = {
                    **self.state.metrics,
                    "stage": "initializing_convlstm",
                    "detail": "Monthly tensors built. Initializing model and data loaders.",
                    "tensor_samples": int(tensor_build.summary.get("sample_count", 0)),
                }
            dataset = _MonthlyTensorDataset(tensor_build)
            if len(dataset) < 2:
                raise RealDataLoadError(
                    f"Need at least 2 monthly tensor samples for training, found {len(dataset)}."
                )
            split_idx = max(1, int(len(dataset) * 0.8))
            if split_idx >= len(dataset):
                split_idx = len(dataset) - 1
            train_dataset = Subset(dataset, list(range(0, split_idx)))
            val_dataset = Subset(dataset, list(range(split_idx, len(dataset))))
            train_loader = DataLoader(train_dataset, batch_size=min(2, len(train_dataset)), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            train_indices = list(range(0, split_idx))
            train_mask = dataset.mask[train_indices]
            train_targets = dataset.y[train_indices]
            train_valid_targets = train_targets[train_mask > 0]
            baseline_flux = float(train_valid_targets.mean().item()) if train_valid_targets.numel() else 0.0

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = OceanPulseLSTM(in_channels=len(tensor_build.feature_names)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config["epochs"], 1))

            resume_checkpoint = self._load_resume_checkpoint()
            best_val = float("inf")
            best_state = None
            epochs = config["epochs"]
            patience = max(epochs + 1, 12)
            stagnant_epochs = 0
            lr = config["learning_rate"]
            start_epoch = 0
            best_mae = float("nan")
            best_r2 = float("nan")

            if resume_checkpoint:
                resume_config = resume_checkpoint.get("config", {})
                if (
                    int(resume_config.get("month_window", config["month_window"])) == int(config["month_window"])
                    and str(resume_config.get("resolution", config["resolution"])) == str(config["resolution"])
                ):
                    try:
                        model.load_state_dict(resume_checkpoint["model_state_dict"])
                        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
                        scheduler_state = resume_checkpoint.get("scheduler_state_dict")
                        if scheduler_state:
                            scheduler.load_state_dict(scheduler_state)
                        best_resume_state = resume_checkpoint.get("best_model_state_dict")
                        if best_resume_state:
                            best_state = best_resume_state
                            self._save_model_checkpoint(
                                model_state_dict=best_resume_state,
                                feature_names=tensor_build.feature_names,
                                resolution=config["resolution"],
                                month_window=config["month_window"],
                                data_summary={**data_summary, **tensor_build.summary},
                                epoch=int(resume_checkpoint.get("epoch", 0)),
                            )
                        best_val = float(resume_checkpoint.get("best_val", best_val))
                        stagnant_epochs = int(resume_checkpoint.get("stagnant_epochs", 0))
                        start_epoch = min(int(resume_checkpoint.get("epoch", 0)), epochs)
                        with self._lock:
                            self.state.current_epoch = start_epoch
                            self.state.progress = start_epoch / max(epochs, 1)
                            self.state.loss_history = list(resume_checkpoint.get("loss_history", []))
                            self.state.metrics = {
                                **self.state.metrics,
                                "stage": "resuming_training",
                                "detail": f"Resuming from saved checkpoint at epoch {start_epoch} of {epochs}.",
                                "resume_checkpoint_path": str(RESUME_CHECKPOINT_PATH),
                            }
                    except Exception:
                        logger.exception("ml_resume_restore_failed path=%s", RESUME_CHECKPOINT_PATH)
                        start_epoch = 0
                else:
                    logger.info(
                        "ml_resume_checkpoint_ignored path=%s reason=config_mismatch",
                        RESUME_CHECKPOINT_PATH,
                    )

            for epoch in range(start_epoch, epochs):
                model.train()
                train_losses: list[float] = []
                for batch in train_loader:
                    features = batch["x"].to(device)
                    targets = batch["y"].to(device)
                    mask = batch["mask"].to(device)
                    atm = batch["atm"].to(device)

                    predictions = model(features, atm)
                    loss = masked_huber_loss(predictions, targets, mask)
                    loss = loss + 0.05 * physics_consistency_penalty(predictions, features[:, -1, 0])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_losses.append(float(loss.item()))

                model.eval()
                val_losses: list[float] = []
                mae_values: list[float] = []
                val_predictions: list[np.ndarray] = []
                val_targets: list[np.ndarray] = []
                with torch.no_grad():
                    for batch in val_loader:
                        features = batch["x"].to(device)
                        targets = batch["y"].to(device)
                        mask = batch["mask"].to(device)
                        atm = batch["atm"].to(device)
                        predictions = model(features, atm)
                        val_loss = masked_huber_loss(predictions, targets, mask)
                        val_losses.append(float(val_loss.item()))
                        abs_error = torch.abs((predictions - targets) * mask).sum() / torch.clamp(mask.sum(), min=1.0)
                        mae_values.append(float(abs_error.item()))
                        valid = mask > 0
                        if torch.any(valid):
                            val_predictions.append(predictions[valid].detach().cpu().numpy())
                            val_targets.append(targets[valid].detach().cpu().numpy())

                train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
                val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
                mae = float(np.mean(mae_values)) if mae_values else float("nan")
                if val_predictions and val_targets:
                    y_pred = np.concatenate(val_predictions)
                    y_true = np.concatenate(val_targets)
                    ss_res = float(np.sum((y_pred - y_true) ** 2))
                    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
                    val_r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
                    baseline_pred = np.full_like(y_true, baseline_flux)
                    baseline_mae = float(np.mean(np.abs(baseline_pred - y_true)))
                    baseline_ss_res = float(np.sum((baseline_pred - y_true) ** 2))
                    baseline_r2 = 1 - baseline_ss_res / ss_tot if ss_tot else float("nan")
                else:
                    val_r2 = float("nan")
                    baseline_mae = float("nan")
                    baseline_r2 = float("nan")
                scheduler.step()

                if val_loss < best_val:
                    best_val = val_loss
                    stagnant_epochs = 0
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    best_mae = mae
                    best_r2 = val_r2
                else:
                    stagnant_epochs += 1

                with self._lock:
                    self.state.current_epoch = epoch + 1
                    self.state.progress = (epoch + 1) / epochs
                    self.state.loss_history.append(round(val_loss, 5))
                    self.state.metrics = {
                        "stage": "training_convlstm",
                        "detail": f"Epoch {epoch + 1} of {epochs}.",
                        "train_loss": round(train_loss, 5),
                        "val_loss": round(val_loss, 5),
                        "mae": round(mae, 5),
                        "r2": round(val_r2, 5) if np.isfinite(val_r2) else None,
                        "baseline_mae": round(baseline_mae, 5) if np.isfinite(baseline_mae) else None,
                        "baseline_r2": round(baseline_r2, 5) if np.isfinite(baseline_r2) else None,
                        "beats_baseline": bool(np.isfinite(mae) and np.isfinite(baseline_mae) and mae < baseline_mae),
                        "samples": int(len(dataset)),
                        "train_samples": int(len(train_dataset)),
                        "val_samples": int(len(val_dataset)),
                        "learning_rate": round(lr, 6),
                        "data_source": tensor_build.summary.get("source", "real_monthly_tensor_pipeline"),
                    }
                    self.state.data_summary = {**data_summary, **tensor_build.summary}
                time.sleep(0.05)

                completed_epoch = epoch + 1
                if completed_epoch % AUTOSAVE_EPOCH_INTERVAL == 0:
                    current_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    latest_best_state = best_state or current_state
                    self._save_model_checkpoint(
                        model_state_dict=latest_best_state,
                        feature_names=tensor_build.feature_names,
                        resolution=config["resolution"],
                        month_window=config["month_window"],
                        data_summary={**data_summary, **tensor_build.summary},
                        epoch=completed_epoch,
                    )
                    self._save_resume_checkpoint(
                        epoch=completed_epoch,
                        total_epochs=epochs,
                        config=config,
                        feature_names=tensor_build.feature_names,
                        data_summary={**data_summary, **tensor_build.summary},
                        model_state_dict=current_state,
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        best_state_dict=latest_best_state,
                        best_val=best_val,
                        stagnant_epochs=stagnant_epochs,
                        loss_history=self.state.loss_history,
                    )
                    with self._lock:
                        self.state.metrics = {
                            **self.state.metrics,
                            "checkpoint_path": str(LATEST_CHECKPOINT_PATH),
                            "resume_checkpoint_path": str(RESUME_CHECKPOINT_PATH),
                            "last_autosave_epoch": completed_epoch,
                            "detail": f"Epoch {completed_epoch} of {epochs}. Autosaved resumable checkpoint.",
                        }
                    write_training_manifest(
                        {
                            "status": "running",
                            "run_id": self.state.run_id,
                            "config": config,
                            "current_epoch": completed_epoch,
                            "total_epochs": epochs,
                            "checkpoint_path": str(LATEST_CHECKPOINT_PATH),
                            "resume_checkpoint_path": str(RESUME_CHECKPOINT_PATH),
                            "data_summary": {**data_summary, **tensor_build.summary},
                            "metrics": dict(self.state.metrics),
                        }
                    )

                if stagnant_epochs >= patience:
                    logger.info("ml_training_early_stopping run_id=%s epoch=%s", self.state.run_id, epoch + 1)
                    break

            if best_state is None:
                raise RuntimeError("Training finished without producing a valid checkpoint state.")

            checkpoint_path = self._save_model_checkpoint(
                model_state_dict=best_state,
                feature_names=tensor_build.feature_names,
                resolution=config["resolution"],
                month_window=config["month_window"],
                data_summary={**data_summary, **tensor_build.summary},
                epoch=self.state.current_epoch,
            )
            if RESUME_CHECKPOINT_PATH.exists():
                RESUME_CHECKPOINT_PATH.unlink()

            # Keep a lightweight tabular surrogate for any non-gridded scoring hooks.
            feature_mean = X.mean(axis=0, keepdims=True)
            feature_std = X.std(axis=0, keepdims=True) + 1e-6
            Xn = (X - feature_mean) / feature_std
            ridge = 1e-3 * np.eye(Xn.shape[1])
            weights = np.linalg.solve(Xn.T @ Xn + ridge, Xn.T @ y)
            bias = float(np.mean(y - (Xn @ weights)))
            val_target = y[max(1, int(len(y) * 0.8)) :]
            val_pred = (Xn[max(1, int(len(y) * 0.8)) :] @ weights) + bias
            surrogate_mae = float(np.mean(np.abs(val_pred - val_target))) if len(val_target) else 0.0
            ss_res = float(np.sum((val_pred - val_target) ** 2)) if len(val_target) else 0.0
            ss_tot = float(np.sum((val_target - val_target.mean()) ** 2)) if len(val_target) else 0.0
            surrogate_r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

            with self._lock:
                self.weights = weights
                self.bias = bias
                self.feature_mean = feature_mean
                self.feature_std = feature_std
                self.state.status = "completed"
                self.state.finished_at = datetime.now(timezone.utc).isoformat()
                self.state.model_ready = True
                self.state.metrics = {
                    **self.state.metrics,
                    "stage": "completed",
                    "detail": "Training completed and checkpoint saved.",
                    "mae": round(best_mae, 5) if np.isfinite(best_mae) else None,
                    "r2": round(best_r2, 5) if np.isfinite(best_r2) else None,
                    "baseline_mae": round(baseline_mae, 5) if np.isfinite(baseline_mae) else None,
                    "baseline_r2": round(baseline_r2, 5) if np.isfinite(baseline_r2) else None,
                    "beats_baseline": bool(np.isfinite(best_mae) and np.isfinite(baseline_mae) and best_mae < baseline_mae),
                    "surrogate_mae": round(surrogate_mae, 5),
                    "surrogate_r2": round(surrogate_r2, 5),
                    "checkpoint_path": str(checkpoint_path),
                }
                self.state.model_summary = {
                    "mode": "real_monthly_convlstm",
                    "target": data_summary.get("target_mode", "co2_flux"),
                    "features": tensor_build.feature_names,
                    "checkpoint_path": str(checkpoint_path),
                }
                self.state.data_summary = {**data_summary, **tensor_build.summary}
                self.state.error = None
            write_training_manifest(
                {
                    "status": "completed",
                    "run_id": self.state.run_id,
                    "config": config,
                    "current_epoch": self.state.current_epoch,
                    "total_epochs": epochs,
                    "checkpoint_path": str(checkpoint_path),
                    "resume_checkpoint_path": None,
                    "data_summary": {**data_summary, **tensor_build.summary},
                    "model_summary": dict(self.state.model_summary),
                    "metrics": dict(self.state.metrics),
                }
            )
            logger.info("ml_training_completed run_id=%s metrics=%s", self.state.run_id, self.state.metrics)
        except Exception as exc:
            logger.exception("ml_training_failed run_id=%s", self.state.run_id)
            with self._lock:
                self.state.status = "failed"
                self.state.finished_at = datetime.now(timezone.utc).isoformat()
                self.state.error = str(exc)
            write_training_manifest(
                {
                    "status": "failed",
                    "run_id": self.state.run_id,
                    "config": config,
                    "current_epoch": self.state.current_epoch,
                    "total_epochs": self.state.total_epochs,
                    "checkpoint_path": self.state.model_summary.get("checkpoint_path"),
                    "resume_checkpoint_path": str(RESUME_CHECKPOINT_PATH) if RESUME_CHECKPOINT_PATH.exists() else None,
                    "data_summary": dict(self.state.data_summary),
                    "metrics": dict(self.state.metrics),
                    "error": str(exc),
                }
            )


class _MonthlyTensorDataset(Dataset):
    def __init__(self, tensor_build):
        self.x = torch.load(tensor_build.x_path, map_location="cpu")
        self.y = torch.load(tensor_build.y_path, map_location="cpu")
        self.mask = torch.load(tensor_build.mask_path, map_location="cpu")
        self.atm = torch.load(tensor_build.atm_path, map_location="cpu")

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, index):
        return {
            "x": self.x[index],
            "y": self.y[index],
            "mask": self.mask[index],
            "atm": self.atm[index],
        }

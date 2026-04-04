from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np
from ingest.real_training_data import RealDataLoadError, build_real_training_dataset, default_era5_directory

if TYPE_CHECKING:
    from db.demo_data import DemoOceanRepository


logger = logging.getLogger("oceanpulse.ml")


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
        "status": "planned",
        "enabled": False,
        "fields": ["salinity", "surface currents", "biogeochemical grids"],
        "url": "https://marine.copernicus.eu",
        "env_var": "COPERNICUS_MARINE_URL",
        "notes": "",
    },
    {
        "id": "nasa_ocean_color",
        "name": "NASA MODIS / Ocean Color",
        "purpose": "Chlorophyll-a and optical indicators tied to biological uptake.",
        "status": "planned",
        "enabled": False,
        "fields": ["chlorophyll_a", "ocean color"],
        "url": "https://oceancolor.gsfc.nasa.gov",
        "env_var": "NASA_OCEANCOLOR_URL",
        "notes": "",
    },
    {
        "id": "era5",
        "name": "ERA5",
        "purpose": "Wind forcing for gas transfer velocity and routing conditions.",
        "status": "planned",
        "enabled": False,
        "fields": ["10m wind", "surface pressure", "wave-relevant forcing"],
        "url": "https://cds.climate.copernicus.eu",
        "env_var": "ERA5_API_URL",
        "notes": "",
    },
    {
        "id": "socat",
        "name": "SOCAT",
        "purpose": "Ship-based pCO2 observations for target construction and validation.",
        "status": "planned",
        "enabled": False,
        "fields": ["fCO2rec", "latitude", "longitude", "SST_C", "salinity"],
        "url": "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0304549/SOCATv2025.tsv",
        "env_var": "SOCAT_DATA_URL",
        "notes": "",
    },
    {
        "id": "noaa_gml_co2",
        "name": "NOAA GML CO2",
        "purpose": "Atmospheric pCO2 baseline feature.",
        "status": "planned",
        "enabled": False,
        "fields": ["monthly atmospheric CO2"],
        "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
        "env_var": "NOAA_GML_CO2_URL",
        "notes": "",
    },
]


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
            config={"epochs": 18, "learning_rate": 0.05, "month_window": 12, "resolution": "2deg"},
            model_summary={
                "mode": "demo_regression",
                "target": "co2_flux",
                "features": [
                    "lat_norm",
                    "lon_norm",
                    "sst",
                    "salinity",
                    "wind_speed",
                    "chl_a",
                    "month_sin",
                    "month_cos",
                ],
            },
            data_summary={
                "source": "synthetic_demo_grid",
                "real_data_ready": False,
                "required_connectors": ["socat", "noaa_gml_co2"],
            },
        )
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.atmospheric_lookup: dict[tuple[int, int], float] = {}
        self.api_registry = []
        for item in REAL_DATA_APIS:
            configured = dict(item)
            env_value = os.getenv(configured["env_var"], "").strip()
            if env_value:
                configured["url"] = env_value
                configured["enabled"] = True
                configured["status"] = "connected"
            self.api_registry.append(configured)

    def list_required_apis(self):
        return [dict(item) for item in self.api_registry]

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
        indexed = {item["name"]: dict(item) for item in self.api_registry}
        for update in updates:
            name = update.get("name")
            if name not in indexed:
                continue
            indexed[name].update(
                {
                    "enabled": bool(update.get("enabled", indexed[name]["enabled"])),
                    "status": str(update.get("status", indexed[name]["status"])),
                    "url": str(update.get("url", indexed[name]["url"])),
                    "notes": str(update.get("notes", indexed[name].get("notes", ""))),
                }
            )
        self.api_registry = list(indexed.values())
        return self.list_required_apis()

    def get_status(self):
        with self._lock:
            return self.state.snapshot()

    def start_training(self, config: dict | None = None):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.state.snapshot(), False
            merged_config = {
                "epochs": int((config or {}).get("epochs", self.state.config.get("epochs", 18))),
                "learning_rate": float((config or {}).get("learning_rate", self.state.config.get("learning_rate", 0.05))),
                "month_window": int((config or {}).get("month_window", self.state.config.get("month_window", 12))),
                "resolution": str((config or {}).get("resolution", self.state.config.get("resolution", "2deg"))),
            }
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
            self._thread = threading.Thread(target=self._train_loop, args=(merged_config,), daemon=True)
            self._thread.start()
            return self.state.snapshot(), True

    def _build_demo_dataset(self, month_window: int, resolution: str):
        rows = []
        current_month = self.repo.now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        for offset in range(month_window):
            stamp = current_month - timedelta(days=30 * offset)
            date_key = stamp.strftime("%Y-%m")
            rows.extend(self.repo.get_flux_grid(date=date_key, resolution=resolution, region="global"))

        X = []
        y = []
        for row in rows:
            month_angle = 2 * np.pi * row.timestamp.month / 12
            X.append(
                [
                    row.lat / 90,
                    row.lon / 180,
                    row.sst / 30,
                    row.salinity / 40,
                    row.wind_speed / 20,
                    row.chl_a / 5,
                    np.sin(month_angle),
                    np.cos(month_angle),
                ]
            )
            y.append(row.co2_flux)
        return (
            np.array(X, dtype=np.float64),
            np.array(y, dtype=np.float64).reshape(-1, 1),
            {
                "source": "synthetic_demo_grid",
                "sample_count": int(len(X)),
                "month_window": int(month_window),
                "resolution": resolution,
                "real_data_ready": False,
            },
        )

    def _build_dataset(self, month_window: int, resolution: str):
        real_ready, connector_state = self._real_training_ready()
        if real_ready:
            socat_config = self._config_by_id("socat")
            noaa_config = self._config_by_id("noaa_gml_co2")
            try:
                era5_config = self._config_by_id("era5")
                era5_path = os.getenv("ERA5_LOCAL_PATH", "").strip() or None
                if era5_config and era5_config.get("enabled"):
                    era5_path = str(era5_config.get("notes", "")).strip() or era5_path or str(default_era5_directory())
                elif not era5_path and default_era5_directory().exists():
                    era5_path = str(default_era5_directory())
                real_dataset = build_real_training_dataset(
                    socat_url=socat_config["url"],
                    noaa_gml_url=noaa_config["url"],
                    month_window=month_window,
                    reference_now=self.repo.now,
                    era_directory=era5_path,
                )
                self.atmospheric_lookup = dict(real_dataset.atmospheric_lookup)
                summary = {
                    **real_dataset.summary,
                    "real_data_ready": True,
                    "connector_state": connector_state,
                }
                return real_dataset.features, real_dataset.targets, summary
            except RealDataLoadError as exc:
                logger.warning("real_training_data_unavailable reason=%s", exc)
                fallback_summary = {
                    "source": "synthetic_demo_grid",
                    "real_data_ready": False,
                    "fallback_reason": str(exc),
                    "connector_state": connector_state,
                }
                X, y, demo_summary = self._build_demo_dataset(month_window, resolution)
                return X, y, {**demo_summary, **fallback_summary}

        X, y, demo_summary = self._build_demo_dataset(month_window, resolution)
        demo_summary["connector_state"] = connector_state
        return X, y, demo_summary

    def _feature_vector(self, row):
        month_angle = 2 * np.pi * row.timestamp.month / 12
        return np.array(
            [
                row.lat / 90,
                row.lon / 180,
                row.sst / 30,
                row.salinity / 40,
                row.wind_speed / 20,
                row.chl_a / 5,
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
            X, y, data_summary = self._build_dataset(config["month_window"], config["resolution"])
            feature_mean = X.mean(axis=0, keepdims=True)
            feature_std = X.std(axis=0, keepdims=True) + 1e-6
            Xn = (X - feature_mean) / feature_std

            split_idx = int(len(Xn) * 0.8)
            X_train, X_val = Xn[:split_idx], Xn[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            weights = np.zeros((X_train.shape[1], 1))
            bias = 0.0
            lr = config["learning_rate"]
            epochs = config["epochs"]
            losses = []

            for epoch in range(epochs):
                preds = X_train @ weights + bias
                error = preds - y_train
                loss = float(np.mean(error**2))
                grad_w = (2 / len(X_train)) * (X_train.T @ error)
                grad_b = float((2 / len(X_train)) * np.sum(error))
                weights -= lr * grad_w
                bias -= lr * grad_b

                val_preds = X_val @ weights + bias
                val_loss = float(np.mean((val_preds - y_val) ** 2))
                losses.append(round(loss, 5))

                with self._lock:
                    self.state.current_epoch = epoch + 1
                    self.state.progress = (epoch + 1) / epochs
                    self.state.loss_history.append(round(val_loss, 5))
                    self.state.metrics = {
                        "train_loss": round(loss, 5),
                        "val_loss": round(val_loss, 5),
                        "samples": int(len(X)),
                        "train_samples": int(len(X_train)),
                        "val_samples": int(len(X_val)),
                        "data_source": data_summary.get("source", "synthetic_demo_grid"),
                    }
                    self.state.data_summary = data_summary
                time.sleep(0.08)

            final_preds = X_val @ weights + bias
            mae = float(np.mean(np.abs(final_preds - y_val)))
            ss_res = float(np.sum((final_preds - y_val) ** 2))
            ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

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
                    "mae": round(mae, 5),
                    "r2": round(r2, 5),
                }
                self.state.model_summary = {
                    "mode": "real_observation_regression"
                    if data_summary.get("source") == "real_observation_sample"
                    else "demo_regression",
                    "target": data_summary.get("target_mode", "co2_flux"),
                    "features": self.state.model_summary.get("features", []),
                }
                self.state.data_summary = data_summary
                self.state.error = None
            logger.info("ml_training_completed run_id=%s metrics=%s", self.state.run_id, self.state.metrics)
        except Exception as exc:
            logger.exception("ml_training_failed run_id=%s", self.state.run_id)
            with self._lock:
                self.state.status = "failed"
                self.state.finished_at = datetime.now(timezone.utc).isoformat()
                self.state.error = str(exc)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

from ingest.real_training_data import RealDataLoadError
from pipeline.artifacts import load_best_published_map_bundle
from .models import AnomalyRecord, ForecastPoint, ForecastResponse, FluxPoint
from .real_products import build_provisional_real_grid_bundle, build_real_grid_bundle, build_real_point_bundle
from ml.trainer import MLTrainerService


REGION_BOUNDS = {
    "global": (-90, 90, -180, 180),
    "pacific": (-60, 65, 110, -70),
    "atlantic": (-60, 70, -80, 20),
    "indian": (-60, 30, 20, 120),
}


@dataclass
class DemoOceanRepository:
    now: datetime = datetime.now(timezone.utc)

    def __post_init__(self):
        self.trainer = MLTrainerService(self)
        self.last_grid_metadata = {
            "verified_map": False,
            "map_source": "real_grid_unavailable",
            "source_summary": "No verified CO2 ocean layers are available until the real gridded pipeline succeeds.",
        }
        self.last_real_anomalies: list[AnomalyRecord] = []
        self._grid_cache: dict[tuple, tuple[float, list[FluxPoint], dict, list[AnomalyRecord]]] = {}
        self._grid_cache_ttl_seconds = 20.0

    def _checkpoint_cache_token(self) -> tuple[str | None, int | None]:
        checkpoint_path = self.trainer.state.model_summary.get("checkpoint_path")
        if not checkpoint_path:
            return None, None
        path = Path(checkpoint_path)
        if not path.exists():
            return checkpoint_path, None
        return str(path), path.stat().st_mtime_ns

    def _monthly_timestamp(self, date_str: str | None) -> datetime:
        if not date_str:
            return self.now
        year, month = map(int, date_str.split("-"))
        return datetime(year, month, 1, tzinfo=timezone.utc)

    def _in_region(self, lat: float, lon: float, region: str) -> bool:
        min_lat, max_lat, min_lon, max_lon = REGION_BOUNDS.get(region, REGION_BOUNDS["global"])
        if min_lon <= max_lon:
            lon_ok = min_lon <= lon <= max_lon
        else:
            lon_ok = lon >= min_lon or lon <= max_lon
        return min_lat <= lat <= max_lat and lon_ok

    def get_flux_grid(self, date: str | None, resolution: str, region: str) -> list[FluxPoint]:
        cache_key = (date or "", resolution, region, *self._checkpoint_cache_token())
        cached = self._grid_cache.get(cache_key)
        now_ts = time.time()
        if cached and now_ts - cached[0] <= self._grid_cache_ttl_seconds:
            _, rows, metadata, anomalies = cached
            self.last_grid_metadata = metadata
            self.last_real_anomalies = anomalies
            return rows

        published_bundle = load_best_published_map_bundle(region=region, resolution=resolution, date=date)
        if published_bundle:
            self.last_grid_metadata = published_bundle.get("metadata", {})
            self.last_real_anomalies = [
                AnomalyRecord.model_validate(item) for item in published_bundle.get("anomalies", [])
            ]
            rows = [FluxPoint.model_validate(item) for item in published_bundle.get("rows", [])]
            self._grid_cache[cache_key] = (now_ts, rows, dict(self.last_grid_metadata), list(self.last_real_anomalies))
            return rows

        noaa_config = self.trainer._config_by_id("noaa_gml_co2") or {}
        socat_config = self.trainer._config_by_id("socat") or {}
        era_config = self.trainer._config_by_id("era5") or {}
        copernicus_config = self.trainer._config_by_id("copernicus_marine") or {}
        try:
            provisional_bundle = build_provisional_real_grid_bundle(
                date=date,
                resolution=resolution,
                region=region,
                trainer=self.trainer,
                noaa_gml_url=str(noaa_config.get("url", "")).strip(),
                era_directory=str(era_config.get("notes", "")).strip(),
                copernicus_directory=str(copernicus_config.get("notes", "")).split("path=", 1)[-1].split(";", 1)[0].strip()
                if "path=" in str(copernicus_config.get("notes", ""))
                else str(copernicus_config.get("notes", "")).strip(),
                reference_now=self.now,
            )
            self.last_grid_metadata = provisional_bundle.metadata
            self.last_real_anomalies = provisional_bundle.anomalies
            self._grid_cache[cache_key] = (
                now_ts,
                list(provisional_bundle.rows),
                dict(provisional_bundle.metadata),
                list(provisional_bundle.anomalies),
            )
            return provisional_bundle.rows
        except RealDataLoadError as exc:
            try:
                real_bundle = build_real_grid_bundle(
                    date=date,
                    resolution=resolution,
                    region=region,
                    trainer=self.trainer,
                    socat_url=str(socat_config.get("url", "")).strip(),
                    noaa_gml_url=str(noaa_config.get("url", "")).strip(),
                    era_directory=str(era_config.get("notes", "")).strip(),
                    copernicus_directory=str(copernicus_config.get("notes", "")).split("path=", 1)[-1].split(";", 1)[0].strip()
                    if "path=" in str(copernicus_config.get("notes", ""))
                    else str(copernicus_config.get("notes", "")).strip(),
                    reference_now=self.now,
                )
                real_bundle.metadata["source_summary"] = (
                    f"{real_bundle.metadata.get('source_summary', '')} Fast provisional serving was unavailable: {exc}"
                ).strip()
                self.last_grid_metadata = real_bundle.metadata
                self.last_real_anomalies = real_bundle.anomalies
                self._grid_cache[cache_key] = (
                    now_ts,
                    list(real_bundle.rows),
                    dict(real_bundle.metadata),
                    list(real_bundle.anomalies),
                )
                return real_bundle.rows
            except RealDataLoadError:
                self.last_grid_metadata = {
                    "verified_map": False,
                    "trained_model_ready": bool(self.trainer.state.model_ready),
                    "inference_mode": self.trainer.state.model_summary.get("mode", "real_monthly_convlstm"),
                    "map_source": "real_grid_unavailable",
                    "source_summary": (
                        "No verified CO2 ocean layers are available because the checkpoint-backed map build failed: "
                        f"{exc}"
                    ),
                }
                self.last_real_anomalies = []
                return []

    def get_recent_anomalies(self, threshold: float, limit: int, date: str | None = None) -> list[AnomalyRecord]:
        if self.last_real_anomalies:
            anomalies = [a for a in self.last_real_anomalies if a.anomaly_score >= threshold]
            return anomalies[:limit]
        return []

    def get_point_forecast(self, lat: float, lon: float, horizon_hours: int) -> ForecastResponse:
        noaa_config = self.trainer._config_by_id("noaa_gml_co2") or {}
        era_config = self.trainer._config_by_id("era5") or {}
        copernicus_config = self.trainer._config_by_id("copernicus_marine") or {}
        try:
            point_bundle = build_real_point_bundle(
                lat=lat,
                lon=lon,
                horizon_hours=horizon_hours,
                trainer=self.trainer,
                noaa_gml_url=str(noaa_config.get("url", "")).strip(),
                era_directory=str(era_config.get("notes", "")).strip(),
                copernicus_directory=str(copernicus_config.get("notes", "")).split("path=", 1)[-1].split(";", 1)[0].strip()
                if "path=" in str(copernicus_config.get("notes", ""))
                else str(copernicus_config.get("notes", "")).strip(),
            )
            return point_bundle.forecast
        except RealDataLoadError:
            pass
        raise RealDataLoadError("Point forecast is unavailable until real monthly drivers and a trained checkpoint are available.")

    def get_global_stats(self, date: str | None) -> dict:
        rows = self.get_flux_grid(date=date, resolution="2deg", region="global")
        if not rows:
            return {
                "date": date or self.now.strftime("%Y-%m"),
                "mean_flux": None,
                "sink_area_pct": None,
                "strongest_sink": None,
                "verified_map": self.last_grid_metadata.get("verified_map", False),
                "map_source": self.last_grid_metadata.get("map_source", "real_grid_unavailable"),
            }
        mean_flux = sum(row.co2_flux for row in rows) / max(len(rows), 1)
        sink_area_pct = sum(1 for row in rows if row.co2_flux < 0) / max(len(rows), 1) * 100
        strongest_sink = min(rows, key=lambda row: row.co2_flux)
        return {
            "date": date or self.now.strftime("%Y-%m"),
            "mean_flux": round(mean_flux, 3),
            "sink_area_pct": round(sink_area_pct, 1),
            "strongest_sink": {
                "lat": strongest_sink.lat,
                "lon": strongest_sink.lon,
                "flux": strongest_sink.co2_flux,
            },
            "verified_map": self.last_grid_metadata.get("verified_map", False),
            "map_source": self.last_grid_metadata.get("map_source", "real_grid_unavailable"),
        }

    def get_point_history(self, lat: float, lon: float, months: int = 12) -> list[dict]:
        noaa_config = self.trainer._config_by_id("noaa_gml_co2") or {}
        era_config = self.trainer._config_by_id("era5") or {}
        copernicus_config = self.trainer._config_by_id("copernicus_marine") or {}
        try:
            point_bundle = build_real_point_bundle(
                lat=lat,
                lon=lon,
                horizon_hours=72,
                trainer=self.trainer,
                noaa_gml_url=str(noaa_config.get("url", "")).strip(),
                era_directory=str(era_config.get("notes", "")).strip(),
                copernicus_directory=str(copernicus_config.get("notes", "")).split("path=", 1)[-1].split(";", 1)[0].strip()
                if "path=" in str(copernicus_config.get("notes", ""))
                else str(copernicus_config.get("notes", "")).strip(),
            )
            return point_bundle.history[-months:]
        except RealDataLoadError:
            return []

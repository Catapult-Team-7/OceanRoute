from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ingest.real_training_data import RealDataLoadError
from .models import AnomalyRecord, ForecastPoint, ForecastResponse, FluxPoint
from .real_products import build_real_grid_bundle, build_real_point_bundle
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
            "map_source": "synthetic_demo_grid",
            "source_summary": "Spatial ocean map still uses synthetic demo fields. Do not treat map layers as verified until real gridded ingestion is wired.",
        }
        self.last_real_anomalies: list[AnomalyRecord] = []

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

    def _base_flux(self, lat: float, lon: float, timestamp: datetime) -> float:
        seasonal = math.sin((timestamp.month / 12) * 2 * math.pi)
        gyre = math.cos(math.radians(lon / 2.5)) * 0.8
        lat_band = -2.6 * math.cos(math.radians(lat)) + 1.2
        equatorial = math.exp(-((lat / 12) ** 2)) * 1.1
        polar_sink = -0.9 * math.exp(-(((abs(lat) - 55) / 14) ** 2))
        return lat_band + gyre + equatorial + polar_sink + seasonal * 0.35

    def _anomaly_hotspots(self, timestamp: datetime) -> list[AnomalyRecord]:
        ts = timestamp.replace(day=15, hour=6, minute=0, second=0, microsecond=0)
        return [
            AnomalyRecord(
                id="north-pacific-gyre",
                lat=33.5,
                lon=-147.0,
                region_name="North Pacific Gyre",
                anomaly_score=0.89,
                deviation_pct=-34.2,
                detected_at=ts,
                severity="high",
            ),
            AnomalyRecord(
                id="south-atlantic-plume",
                lat=-24.0,
                lon=-8.0,
                region_name="South Atlantic Plume",
                anomaly_score=0.76,
                deviation_pct=-21.4,
                detected_at=ts - timedelta(hours=6),
                severity="medium",
            ),
            AnomalyRecord(
                id="arabian-sea-bloom",
                lat=16.0,
                lon=66.0,
                region_name="Arabian Sea Bloom",
                anomaly_score=0.93,
                deviation_pct=18.9,
                detected_at=ts - timedelta(hours=12),
                severity="critical",
            ),
        ]

    def get_flux_grid(self, date: str | None, resolution: str, region: str) -> list[FluxPoint]:
        noaa_config = self.trainer._config_by_id("noaa_gml_co2") or {}
        socat_config = self.trainer._config_by_id("socat") or {}
        era_config = self.trainer._config_by_id("era5") or {}
        copernicus_config = self.trainer._config_by_id("copernicus_marine") or {}
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
            self.last_grid_metadata = real_bundle.metadata
            self.last_real_anomalies = real_bundle.anomalies
            return real_bundle.rows
        except RealDataLoadError:
            self.last_grid_metadata = {
                "verified_map": False,
                "map_source": "synthetic_demo_grid",
                "source_summary": "Spatial ocean map still uses synthetic demo fields. Do not treat map layers as verified until real gridded ingestion is wired.",
            }
            self.last_real_anomalies = []

        timestamp = self._monthly_timestamp(date)
        step = {
            "0.25deg": 10,
            "0.5deg": 8,
            "1deg": 6,
            "2deg": 12,
        }.get(resolution, 6)
        anomalies = self._anomaly_hotspots(timestamp)
        rows: list[FluxPoint] = []
        for lat in range(-72, 73, step):
            for lon in range(-180, 181, step):
                if not self._in_region(lat, lon, region):
                    continue
                flux = self._base_flux(lat, lon, timestamp)
                anomaly_score = 0.0
                for anomaly in anomalies:
                    dist = math.hypot((lat - anomaly.lat) / 10, (lon - anomaly.lon) / 14)
                    impact = math.exp(-(dist**2))
                    anomaly_score = max(anomaly_score, anomaly.anomaly_score * impact)
                    flux += (anomaly.deviation_pct / 100) * 0.6 * impact
                sst = max(-1.5, 28 - abs(lat) * 0.32 + math.sin(math.radians(lon)) * 1.7)
                salinity = 34.2 + math.cos(math.radians(lon / 1.8)) * 1.2 - abs(lat) * 0.008
                wind_speed = 4.5 + abs(math.sin(math.radians(lat * 2))) * 6.0
                chl_a = max(0.02, 0.8 + math.cos(math.radians(lat * 3)) * 0.4)
                rows.append(
                    FluxPoint(
                        lat=float(lat),
                        lon=float(lon),
                        co2_flux=round(flux, 4),
                        sst=round(sst, 2),
                        salinity=round(salinity, 2),
                        wind_speed=round(wind_speed, 2),
                        chl_a=round(chl_a, 3),
                        anomaly_score=round(min(anomaly_score, 1.0), 4),
                        timestamp=timestamp,
                    )
                )
        return rows

    def get_recent_anomalies(self, threshold: float, limit: int, date: str | None = None) -> list[AnomalyRecord]:
        if self.last_grid_metadata.get("verified_map") and self.last_real_anomalies:
            anomalies = [a for a in self.last_real_anomalies if a.anomaly_score >= threshold]
            return anomalies[:limit]
        timestamp = self._monthly_timestamp(date)
        anomalies = [a for a in self._anomaly_hotspots(timestamp) if a.anomaly_score >= threshold]
        return anomalies[:limit]

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

        current = self._base_flux(lat, lon, self.now)
        steps = [hours for hours in (24, 48, 72) if hours <= max(horizon_hours, 24)]
        forecast: list[ForecastPoint] = []
        for hours in steps:
            drift = math.sin(math.radians(lon + hours)) * 0.12 + math.cos(math.radians(lat * 2)) * -0.09
            flux = current + drift * (hours / 24)
            spread = 0.32 + (hours / 72) * 0.28
            forecast.append(
                ForecastPoint(
                    hours_ahead=hours,
                    flux=round(flux, 3),
                    confidence_low=round(flux - spread, 3),
                    confidence_high=round(flux + spread, 3),
                )
            )
        return ForecastResponse(lat=lat, lon=lon, current_flux=round(current, 3), forecast=forecast)

    def get_global_stats(self, date: str | None) -> dict:
        rows = self.get_flux_grid(date=date, resolution="2deg", region="global")
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
            "map_source": self.last_grid_metadata.get("map_source", "synthetic_demo_grid"),
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
            pass

        items = []
        for offset in range(months - 1, -1, -1):
            ts = (self.now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=30 * offset))
            items.append({"date": ts.strftime("%Y-%m"), "flux": round(self._base_flux(lat, lon, ts), 3)})
        return items

from datetime import datetime

from pydantic import BaseModel, Field


class FluxPoint(BaseModel):
    lat: float
    lon: float
    co2_flux: float
    sst: float
    salinity: float
    wind_speed: float
    chl_a: float
    anomaly_score: float = 0.0
    observed_flux: float | None = None
    predicted_flux: float | None = None
    weakening_score: float | None = None
    route_priority: float | None = None
    timestamp: datetime
    source: str = "MODEL"


class ForecastPoint(BaseModel):
    hours_ahead: int = Field(..., ge=1)
    flux: float
    confidence_low: float
    confidence_high: float


class ForecastResponse(BaseModel):
    lat: float
    lon: float
    current_flux: float
    forecast: list[ForecastPoint]


class AnomalyRecord(BaseModel):
    id: str
    lat: float
    lon: float
    region_name: str
    anomaly_score: float
    deviation_pct: float
    detected_at: datetime
    severity: str
    is_active: bool = True


class HeatmapMetadata(BaseModel):
    date: str
    units: str
    mean_flux: float
    sink_area_pct: float
    inference_mode: str
    trained_model_ready: bool
    verified_map: bool = False
    map_source: str = "synthetic_demo_grid"
    source_summary: str = ""


class FeatureGeometry(BaseModel):
    type: str = "Point"
    coordinates: tuple[float, float]


class FeatureProperties(BaseModel):
    flux: float
    observed_flux: float | None = None
    predicted_flux: float
    sst: float
    anomaly_score: float
    weakening_score: float
    route_priority: float
    is_anomaly: bool


class FluxFeature(BaseModel):
    type: str = "Feature"
    geometry: FeatureGeometry
    properties: FeatureProperties


class HeatmapResponse(BaseModel):
    type: str = "FeatureCollection"
    metadata: HeatmapMetadata
    features: list[FluxFeature]

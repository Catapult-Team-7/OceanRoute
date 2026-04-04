from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


DebrisClass = Literal["low", "high"]
FoundStatus = Literal["found", "not_found"]
SourceMode = Literal["auto", "sample", "live"]
RecommendedMode = Literal["collection", "recon"]


DEBRIS_CLASS_METADATA: dict[DebrisClass, dict[str, object]] = {
    "low": {
        "label": "Low-windage fragments",
        "description": "Surface fragments and small floatables that mostly follow currents.",
        "windage_factor": 0.05,
    },
    "high": {
        "label": "High-windage floatables",
        "description": "Bottles, foam, and gear that respond more strongly to wind.",
        "windage_factor": 0.14,
    },
}


class DebrisClassInfo(BaseModel):
    code: DebrisClass
    label: str
    description: str
    windage_factor: float


class ForecastProvenance(BaseModel):
    generated_at: datetime
    horizon_hours: int
    source_mode_requested: SourceMode
    source_mode_used: Literal["sample", "live"]
    is_fallback: bool
    is_stale: bool
    age_minutes: int
    stale_after_minutes: int
    source_notes: list[str] = Field(default_factory=list)


class GridPoint(BaseModel):
    cell_id: str
    lat: float
    lon: float
    current_u: float
    current_v: float
    wind_u: float
    wind_v: float
    shoreline_proximity: float = Field(ge=0.0, le=1.0)
    restricted: bool = False
    water_temperature_c: float | None = None


class OperationalGridFrame(BaseModel):
    valid_at: datetime
    horizon_hour: int = Field(ge=1, le=72)
    grid: list[GridPoint]


class OperationalContext(BaseModel):
    generated_at: datetime
    pilot_region: str
    source_mode_requested: SourceMode
    source_mode_used: Literal["sample", "live"]
    is_fallback: bool = False
    source_notes: list[str] = Field(default_factory=list)
    frames: list[OperationalGridFrame] = Field(default_factory=list)


class DriftBaselineInput(BaseModel):
    run_id: str
    generated_at: datetime
    valid_at: datetime
    horizon_hour: int
    debris_class: DebrisClass
    diffusion_sigma: float = Field(default=0.12, ge=0.0)
    windage_factor: float = Field(default=0.05, ge=0.0)
    source_strength: float = Field(default=1.0, ge=0.0)
    grid: list[GridPoint]


class ResidualModelInput(BaseModel):
    run_id: str
    debris_class: DebrisClass
    horizon_hour: int
    baseline_density: dict[str, float]
    currents: dict[str, tuple[float, float]]
    winds: dict[str, tuple[float, float]]
    history_bias: dict[str, float] = Field(default_factory=dict)
    shoreline: dict[str, float] = Field(default_factory=dict)


class ForecastStep(BaseModel):
    valid_at: datetime
    horizon_hour: int = Field(ge=1, le=72)
    cell_id: str
    lat: float
    lon: float
    debris_class: DebrisClass
    probability: float = Field(ge=0.0, le=1.0)
    expected_kg_min: float = Field(ge=0.0)
    expected_kg_max: float = Field(ge=0.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    beaching_risk: float = Field(ge=0.0, le=1.0)
    restricted: bool = False


class HotspotSummary(BaseModel):
    valid_at: datetime
    horizon_hour: int
    cell_id: str
    lat: float
    lon: float
    debris_class: DebrisClass
    probability: float
    expected_kg_min: float
    expected_kg_max: float
    confidence: float


class ForecastRunRequest(BaseModel):
    horizon_hours: int = Field(default=24, ge=24, le=72)
    debris_classes: list[DebrisClass] = Field(default_factory=lambda: ["low", "high"])
    source_strength: float = Field(default=1.0, ge=0.0)
    seed: int = 42
    source_mode: SourceMode = "auto"


class ForecastRunResponse(BaseModel):
    run_id: str
    generated_at: datetime
    horizon_hours: int
    source_mode_requested: SourceMode
    source_mode_used: Literal["sample", "live"]
    is_fallback: bool
    is_stale: bool
    age_minutes: int
    stale_after_minutes: int
    steps_generated: int
    top_hotspots: list[HotspotSummary]
    provenance: ForecastProvenance
    summary: dict[str, float | int | str]


class ForecastSnapshot(BaseModel):
    run_id: str
    generated_at: datetime
    horizon_hours: int
    pilot_region: str
    source_mode_requested: SourceMode
    source_mode_used: Literal["sample", "live"]
    is_fallback: bool
    is_stale: bool
    age_minutes: int
    stale_after_minutes: int
    source_notes: list[str]
    debris_classes: list[DebrisClassInfo]
    steps: list[ForecastStep]
    top_hotspots: list[HotspotSummary]
    provenance: ForecastProvenance
    summary: dict[str, float | int | str]


class HotspotQueryResponse(BaseModel):
    run_id: str
    generated_at: datetime
    horizon_hours: int
    source_mode_requested: SourceMode
    source_mode_used: Literal["sample", "live"]
    is_fallback: bool
    is_stale: bool
    age_minutes: int
    stale_after_minutes: int
    filters: dict[str, float | int | str | None]
    hotspots: list[ForecastStep]
    top_hotspots: list[HotspotSummary]
    provenance: ForecastProvenance


class RouteCandidate(BaseModel):
    cell_id: str
    lat: float
    lon: float
    expected_kg_min: float = Field(ge=0.0)
    expected_kg_max: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    service_time_min: int = Field(default=10, ge=1)
    access_flag: bool = True

    @property
    def expected_mid_kg(self) -> float:
        return (self.expected_kg_min + self.expected_kg_max) / 2


class RouteObjectiveWeights(BaseModel):
    yield_weight: float = Field(default=1.0, ge=0.0)
    distance_weight: float = Field(default=0.12, ge=0.0)
    uncertainty_weight: float = Field(default=6.0, ge=0.0)
    fuel_weight: float = Field(default=0.55, ge=0.0)


class RouteOptimizeRequest(BaseModel):
    vessel_id: str = "sf-bay-pilot-vessel"
    depot_lat: float
    depot_lon: float
    mission_hours: float = Field(default=4.0, gt=0.25, le=12.0)
    vessel_speed_kmh: float = Field(default=18.0, gt=1.0, le=60.0)
    fuel_burn_lph: float = Field(default=12.0, gt=0.0, le=100.0)
    target_horizon_hour: int = Field(default=24, ge=1, le=72)
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    min_objective_score: float = Field(default=1.0)
    objective_weights: RouteObjectiveWeights = Field(default_factory=RouteObjectiveWeights)
    candidates: list[RouteCandidate] = Field(default_factory=list)
    must_visit_cell_ids: list[str] = Field(default_factory=list)
    avoid_cell_ids: list[str] = Field(default_factory=list)


class RouteLeg(BaseModel):
    from_cell: str
    to_cell: str
    distance_km: float
    travel_minutes: float


class RoutePlan(BaseModel):
    mission_id: str
    created_at: datetime
    vessel_id: str
    recommended_mode: RecommendedMode
    target_horizon_hour: int
    ordered_cell_ids: list[str]
    expected_kg_min: float
    expected_kg_max: float
    expected_distance_km: float
    expected_duration_min: float
    estimated_fuel_liters: float
    objective_score: float
    legs: list[RouteLeg]
    uncertainty_risk: float
    alternates: list[str] = Field(default_factory=list)
    forecast_run_id: str | None = None
    forecast_provenance: ForecastProvenance | None = None
    forecast_is_stale: bool = False
    metadata: dict[str, float | int | str] = Field(default_factory=dict)


class RoutingBenchmarkStrategy(BaseModel):
    strategy: Literal["nearest_hotspot", "highest_yield", "recon_aware"]
    recommended_mode: RecommendedMode
    ordered_cell_ids: list[str]
    expected_kg_min: float = Field(ge=0.0)
    expected_kg_max: float = Field(ge=0.0)
    expected_distance_km: float = Field(ge=0.0)
    uncertainty_penalty: float = Field(ge=0.0)
    hotspot_hit_rate_estimate: float = Field(ge=0.0, le=1.0)
    objective_score: float


class RoutingBenchmarkReport(BaseModel):
    generated_at: datetime
    forecast_run_id: str
    target_horizon_hour: int
    compared_strategies: list[RoutingBenchmarkStrategy]
    winning_strategy: Literal["nearest_hotspot", "highest_yield", "recon_aware"]


class ObservationUpload(BaseModel):
    mission_id: str | None = None
    observed_at: datetime
    lat: float
    lon: float
    found_status: FoundStatus | None = None
    debris_found: bool | None = None
    debris_class: DebrisClass | None = None
    estimated_kg: float = Field(default=0.0, ge=0.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    photo_url: str | None = None
    note: str | None = None
    route_deviation_reason: str | None = None

    @model_validator(mode="after")
    def normalize_found_status(self) -> "ObservationUpload":
        if self.found_status is None and self.debris_found is not None:
            self.found_status = "found" if self.debris_found else "not_found"
        if self.found_status is None:
            self.found_status = "found"
        return self


class MissionOutcome(BaseModel):
    mission_id: str
    completed_at: datetime
    vessel_id: str
    recommended_mode: RecommendedMode = "collection"
    predicted_kg_min: float = Field(default=0.0, ge=0.0)
    predicted_kg_max: float = Field(default=0.0, ge=0.0)
    collected_kg: float = Field(default=0.0, ge=0.0)
    vessel_distance_km: float = Field(default=0.0, ge=0.0)
    vessel_hours: float = Field(default=0.0, ge=0.0)
    hotspot_hits: int = Field(default=0, ge=0)
    hotspot_misses: int = Field(default=0, ge=0)
    false_search_km: float = Field(default=0.0, ge=0.0)
    fuel_liters: float = Field(default=0.0, ge=0.0)
    route_deviation_reason: str | None = None
    notes: str | None = None


class ImpactDashboard(BaseModel):
    total_missions: int
    total_collected_kg: float
    total_distance_km: float
    total_hours: float
    kg_per_vessel_km: float
    kg_per_hour: float
    hotspot_precision: float
    false_search_distance_km: float
    mission_hit_rate: float
    latest_updated_at: datetime | None = None

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


DebrisClass = Literal["low", "high"]
FoundStatus = Literal["found", "not_found"]
SourceMode = Literal["auto", "sample", "live"]
RecommendedMode = Literal["collection", "recon"]
ModelArchitecture = Literal["linear_residual", "temporal_unet", "convlstm"]
TrainingStatus = Literal["pending", "completed", "failed"]
BaselineEngine = Literal["pygnome", "custom_particle"]
ModelStage = Literal["candidate", "champion", "archived"]
TrainingScope = Literal["shared", "per_region"]
TrainingScopeRequest = Literal["shared", "per_region", "both"]
PromotionPolicy = Literal["auto", "candidate_only", "always_activate"]
ModelExportFormat = Literal["trace", "script", "json"]
BackfillMode = Literal["dataset_only", "live_parity"]


DEBRIS_CLASS_METADATA: dict[DebrisClass, dict[str, object]] = {
    "low": {
        "label": "Low-windage fragments",
        "description": "Surface fragments and small floatables that mostly follow currents.",
        "windage_factor": 0.01,
    },
    "high": {
        "label": "High-windage floatables",
        "description": "Bottles, foam, and gear that respond more strongly to wind.",
        "windage_factor": 0.025,
    },
}


class DebrisClassInfo(BaseModel):
    code: DebrisClass
    label: str
    description: str
    windage_factor: float


class RegionInfo(BaseModel):
    id: str
    name: str
    description: str
    bbox: dict[str, float]
    default_depot_lat: float
    default_depot_lon: float
    grid_dx_km: float
    grid_dy_km: float
    tags: list[str] = Field(default_factory=list)


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
    baseline_engine: BaselineEngine | None = None
    baseline_artifact_uri: str | None = None
    requested_model_id: str | None = None
    resolved_model_id: str | None = None
    model_id: str | None = None
    model_architecture: ModelArchitecture | None = None
    model_dataset_version: str | None = None
    model_stage: ModelStage | None = None
    training_scope: TrainingScope | None = None
    used_candidate_override: bool = False
    used_inference_fallback: bool = False
    inference_service_version: str | None = None
    prediction_artifact_uri: str | None = None


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


class GridSpec(BaseModel):
    crs: str
    width: int
    height: int
    bbox: dict[str, float]
    resolution_km: tuple[float, float]
    transform: list[float]
    shoreline_mask_uri: str | None = None
    restricted_mask_uri: str | None = None
    bathymetry_mask_uri: str | None = None


class ForcingRefs(BaseModel):
    current_source: str
    wind_source: str
    wave_source: str | None = None
    drifter_source: str | None = None
    shoreline_source: str | None = None
    bathymetry_source: str | None = None


class OperationalGridFrame(BaseModel):
    valid_at: datetime
    horizon_hour: int = Field(ge=1, le=72)
    grid: list[GridPoint]


class OperationalContext(BaseModel):
    generated_at: datetime
    pilot_region: str
    region: RegionInfo
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
    diffusion_sigma: float = Field(default=0.18, ge=0.0)
    windage_factor: float = Field(default=0.0, ge=0.0)
    stokes_drift_factor: float = Field(default=0.01, ge=0.0)
    ensemble_members: int = Field(default=10, ge=1, le=32)
    particles_per_member: int = Field(default=240, ge=16, le=5000)
    beaching_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
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
    ensemble_spread: dict[str, float] = Field(default_factory=dict)
    beaching_fraction: dict[str, float] = Field(default_factory=dict)


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
    baseline_density: float = Field(default=0.0, ge=0.0)
    ensemble_spread: float = Field(default=0.0, ge=0.0)
    beaching_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    stokes_drift_u: float = 0.0
    stokes_drift_v: float = 0.0
    windage_fraction: float = Field(default=0.0, ge=0.0)
    restricted: bool = False


class BaselineArtifact(BaseModel):
    artifact_id: str
    region_id: str
    run_id: str
    debris_class: DebrisClass
    generated_at: datetime
    forecast_valid_at: datetime
    horizon_hour: int
    grid_spec: GridSpec
    forcing_refs: ForcingRefs
    baseline_engine: BaselineEngine
    source_mode_requested: SourceMode
    source_mode_used: Literal["sample", "live"]
    is_fallback: bool
    source_notes: list[str] = Field(default_factory=list)
    density_uri: str
    current_u_uri: str | None = None
    current_v_uri: str | None = None
    wind_u_uri: str | None = None
    wind_v_uri: str | None = None
    ensemble_spread_uri: str
    beaching_fraction_uri: str
    stokes_u_uri: str
    stokes_v_uri: str
    stokes_magnitude_uri: str
    manifest_uri: str
    parquet_index_uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionArtifact(BaseModel):
    artifact_id: str
    forecast_run_id: str
    region_id: str
    debris_class: DebrisClass
    created_at: datetime
    model_id: str
    model_architecture: ModelArchitecture
    model_dataset_version: str
    training_scope: TrainingScope | None = None
    target_horizons: list[int] = Field(default_factory=lambda: [24, 48, 72])
    inference_service_version: str
    feature_artifact_uri: str
    hotspot_probability_uri: str
    expected_kg_uri: str
    uncertainty_uri: str | None = None
    feature_schema_path: str | None = None
    normalization_stats_path: str | None = None
    parquet_index_uri: str
    manifest_uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferencePredictRequest(BaseModel):
    region_id: str
    forecast_run_id: str
    debris_class: DebrisClass
    feature_artifact_uri: str
    model_id: str | None = None
    target_horizons: list[int] = Field(default_factory=lambda: [24, 48, 72])


class InferencePredictResponse(BaseModel):
    artifact: PredictionArtifact
    loaded_model_id: str
    used_fallback: bool = False


class ModelLoadRequest(BaseModel):
    model_id: str


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
    region_id: str | None = None
    horizon_hours: int = Field(default=24, ge=24, le=72)
    debris_classes: list[DebrisClass] = Field(default_factory=lambda: ["low", "high"])
    source_strength: float = Field(default=1.0, ge=0.0)
    seed: int = 42
    source_mode: SourceMode = "auto"
    model_id: str | None = None


class ForecastRunResponse(BaseModel):
    run_id: str
    generated_at: datetime
    horizon_hours: int
    region: RegionInfo
    source_mode_requested: SourceMode
    source_mode_used: Literal["sample", "live"]
    is_fallback: bool
    is_stale: bool
    age_minutes: int
    stale_after_minutes: int
    steps_generated: int
    top_hotspots: list[HotspotSummary]
    provenance: ForecastProvenance
    summary: dict[str, Any]


class ForecastSnapshot(BaseModel):
    run_id: str
    generated_at: datetime
    horizon_hours: int
    pilot_region: str
    region: RegionInfo
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
    summary: dict[str, Any]


class HotspotQueryResponse(BaseModel):
    run_id: str
    generated_at: datetime
    horizon_hours: int
    region: RegionInfo
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
    region_id: str | None = None
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
    region_id: str | None = None
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
    region_id: str
    forecast_run_id: str
    target_horizon_hour: int
    compared_strategies: list[RoutingBenchmarkStrategy]
    winning_strategy: Literal["nearest_hotspot", "highest_yield", "recon_aware"]


class DatasetBuildRequest(BaseModel):
    region_id: str
    label_type: Literal["hotspot_presence", "expected_kg"] = "hotspot_presence"
    max_forecast_runs: int = Field(default=10, ge=1, le=200)


class HistoricalBackfillRequest(BaseModel):
    region_id: str
    source_mode: SourceMode = "sample"
    days: int = Field(default=180, ge=1, le=366)
    mode: BackfillMode = "dataset_only"
    chunk_days: int = Field(default=28, ge=1, le=180)
    ensemble_members: int | None = Field(default=None, ge=1, le=32)
    particles_per_member: int | None = Field(default=None, ge=16, le=5000)
    debris_classes: list[DebrisClass] = Field(default_factory=lambda: ["low", "high"])


class HistoricalBackfillTiming(BaseModel):
    region_id: str
    chunk_index: int = Field(ge=1)
    timestamps_in_chunk: int = Field(ge=0)
    runs_created: int = Field(ge=0)
    baseline_artifacts_created: int = Field(ge=0)
    setup_ms: float = Field(ge=0.0)
    source_load_ms: float = Field(ge=0.0)
    baseline_ms: float = Field(ge=0.0)
    artifact_write_ms: float = Field(ge=0.0)
    db_write_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class HistoricalBackfillResponse(BaseModel):
    region_id: str
    source_mode: SourceMode
    days_backfilled: int
    mode: BackfillMode
    runs_created: int
    baseline_artifacts_created: int
    dataset_ready_run_count: int
    timestamps_planned: int
    timestamps_processed: int
    chunk_count: int
    timings: list[HistoricalBackfillTiming] = Field(default_factory=list)


class DatasetExportRequest(BaseModel):
    region_id: str | None = None
    region_ids: list[str] | None = None
    dataset_id: str | None = None
    max_forecast_runs: int = Field(default=180, ge=3, le=1000)
    lookback_hours: int = Field(default=12, ge=3, le=24)
    target_horizons: list[int] = Field(default_factory=lambda: [24, 48, 72])
    label_strategy: Literal["observed_or_proxy"] = "observed_or_proxy"


class DatasetArtifact(BaseModel):
    dataset_id: str
    region_id: str
    created_at: datetime
    dataset_version: str
    label_type: Literal["hotspot_presence", "expected_kg"]
    sample_count: int
    feature_count: int
    feature_names: list[str]
    artifact_path: str
    zarr_uri: str | None = None
    parquet_index_uri: str | None = None
    manifest_path: str | None = None
    splits_path: str | None = None
    feature_stats_path: str | None = None
    metadata_path: str | None = None
    split_counts: dict[str, int] = Field(default_factory=dict)
    schema_versions: dict[str, str] = Field(default_factory=dict)
    region_ids: list[str] = Field(default_factory=list)
    horizons: list[int] = Field(default_factory=list)
    input_channels: list[str] = Field(default_factory=list)
    target_channels: list[str] = Field(default_factory=list)
    tensor_shapes: dict[str, list[int]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelTrainRequest(BaseModel):
    region_id: str | None = None
    dataset_id: str
    architecture: ModelArchitecture = "linear_residual"
    activate: bool = True
    region_ids: list[str] | None = None
    horizons: list[int] = Field(default_factory=lambda: [24, 48, 72])
    training_scope: TrainingScope = "shared"
    device: str = "auto"
    promote_policy: PromotionPolicy = "auto"
    epochs: int = Field(default=2, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)


class ModelRegistryEntry(BaseModel):
    model_id: str
    region_id: str
    created_at: datetime
    architecture: ModelArchitecture
    status: TrainingStatus
    stage: ModelStage = "candidate"
    is_active: bool
    training_scope: TrainingScope = "per_region"
    artifact_path: str
    dataset_id: str
    dataset_version: str | None = None
    trained_regions: list[str] = Field(default_factory=list)
    compatible_regions: list[str] = Field(default_factory=list)
    horizons: list[int] = Field(default_factory=list)
    input_channels: list[str] = Field(default_factory=list)
    output_heads: list[str] = Field(default_factory=list)
    normalization_stats_path: str | None = None
    feature_schema_path: str | None = None
    best_checkpoint_path: str | None = None
    checkpoint_path: str | None = None
    export_artifact_path: str | None = None
    export_format: ModelExportFormat | None = None
    evaluation_path: str | None = None
    framework: str | None = None
    metrics: dict[str, Any]


class ModelTrainResponse(BaseModel):
    training_run_id: str
    model: ModelRegistryEntry
    metrics: dict[str, Any]


class ModelEvaluateResponse(BaseModel):
    model_id: str
    region_id: str
    stage: ModelStage
    artifact_path: str
    export_format: ModelExportFormat | None = None
    metrics: dict[str, Any]
    evaluation_path: str


class ModelExportResponse(BaseModel):
    model_id: str
    export_artifact_path: str
    format: Literal["torchscript", "json"]


class ModelPromotionResponse(BaseModel):
    promoted_model_id: str | None = None
    previous_model_id: str | None = None
    region_id: str
    reason: str
    promoted: bool


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

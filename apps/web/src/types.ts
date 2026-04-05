export type DebrisClass = "low" | "high";
export type RecommendedMode = "collection" | "recon";
export type SourceMode = "auto" | "sample" | "live";
export type BaselineEngine = "pygnome" | "custom_particle";
export type ModelArchitecture = "linear_residual" | "temporal_unet" | "convlstm";
export type TrainingScope = "shared" | "per_region";
export type ModelStage = "candidate" | "champion" | "archived";

export interface HealthStatus {
  status: string;
  pilot_region: string;
  default_horizon_hours: number;
  scheduler_enabled: boolean;
  database_url: string;
  ingest_mode: string;
  forecast_stale_after_minutes: number;
  default_depot_lat: number;
  default_depot_lon: number;
}

export interface DebrisClassInfo {
  code: DebrisClass;
  label: string;
  description: string;
  windage_factor: number;
}

export interface RegionInfo {
  id: string;
  name: string;
  description: string;
  bbox: Record<string, number>;
  default_depot_lat: number;
  default_depot_lon: number;
  grid_dx_km: number;
  grid_dy_km: number;
  tags: string[];
}

export interface ForecastStep {
  valid_at: string;
  horizon_hour: number;
  cell_id: string;
  lat: number;
  lon: number;
  debris_class: DebrisClass;
  probability: number;
  expected_kg_min: number;
  expected_kg_max: number;
  uncertainty: number;
  confidence: number;
  beaching_risk: number;
  baseline_density: number;
  ensemble_spread: number;
  beaching_fraction: number;
  stokes_drift_u: number;
  stokes_drift_v: number;
  windage_fraction: number;
  restricted: boolean;
}

export interface HotspotSummary {
  valid_at: string;
  horizon_hour: number;
  cell_id: string;
  lat: number;
  lon: number;
  debris_class: DebrisClass;
  probability: number;
  expected_kg_min: number;
  expected_kg_max: number;
  confidence: number;
}

export interface ForecastProvenance {
  generated_at: string;
  horizon_hours: number;
  source_mode_requested: SourceMode;
  source_mode_used: "sample" | "live";
  is_fallback: boolean;
  is_stale: boolean;
  age_minutes: number;
  stale_after_minutes: number;
  source_notes: string[];
  baseline_engine?: BaselineEngine | null;
  baseline_artifact_uri?: string | null;
  requested_model_id?: string | null;
  resolved_model_id?: string | null;
  model_id?: string | null;
  model_architecture?: ModelArchitecture | null;
  model_dataset_version?: string | null;
  model_stage?: ModelStage | null;
  training_scope?: TrainingScope | null;
  used_candidate_override: boolean;
  used_inference_fallback: boolean;
  inference_service_version?: string | null;
  prediction_artifact_uri?: string | null;
}

export interface ForecastSnapshot {
  run_id: string;
  generated_at: string;
  horizon_hours: number;
  pilot_region: string;
  region: RegionInfo;
  source_mode_requested: SourceMode;
  source_mode_used: "sample" | "live";
  is_fallback: boolean;
  is_stale: boolean;
  age_minutes: number;
  stale_after_minutes: number;
  source_notes: string[];
  debris_classes: DebrisClassInfo[];
  steps: ForecastStep[];
  top_hotspots: HotspotSummary[];
  provenance: ForecastProvenance;
  summary: Record<string, number | string | boolean>;
}

export interface RouteLeg {
  from_cell: string;
  to_cell: string;
  distance_km: number;
  travel_minutes: number;
}

export interface RoutePlan {
  mission_id: string;
  created_at: string;
  region_id: string | null;
  vessel_id: string;
  recommended_mode: RecommendedMode;
  target_horizon_hour: number;
  ordered_cell_ids: string[];
  expected_kg_min: number;
  expected_kg_max: number;
  expected_distance_km: number;
  expected_duration_min: number;
  estimated_fuel_liters: number;
  objective_score: number;
  uncertainty_risk: number;
  alternates: string[];
  legs: RouteLeg[];
  forecast_run_id: string | null;
  forecast_provenance: ForecastProvenance | null;
  forecast_is_stale: boolean;
  metadata: Record<string, number | string | boolean>;
}

export interface ImpactDashboard {
  total_missions: number;
  total_collected_kg: number;
  total_distance_km: number;
  total_hours: number;
  kg_per_vessel_km: number;
  kg_per_hour: number;
  hotspot_precision: number;
  false_search_distance_km: number;
  mission_hit_rate: number;
  latest_updated_at: string | null;
}

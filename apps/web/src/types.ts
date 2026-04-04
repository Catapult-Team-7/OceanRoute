export type DebrisClass = "low" | "high";
export type RecommendedMode = "collection" | "recon";
export type SourceMode = "auto" | "sample" | "live";

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
}

export interface ForecastSnapshot {
  run_id: string;
  generated_at: string;
  horizon_hours: number;
  pilot_region: string;
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
  summary: Record<string, number | string>;
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
  metadata: Record<string, number | string>;
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

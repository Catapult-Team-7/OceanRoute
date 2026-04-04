import type {
  DebrisClass,
  ForecastSnapshot,
  HealthStatus,
  HotspotSummary,
  ImpactDashboard,
  RoutePlan,
  RoutingBenchmarkReport,
} from "./types";

export const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === "true";
export const DEMO_NOTICE =
  "Hosted demo mode is using seeded San Francisco Bay artifacts and is read-only on GitHub Pages.";
export const FALLBACK_NOTICE =
  "The live API is unavailable right now, so the site has switched to the seeded San Francisco Bay launch demo.";

export interface DemoFilters {
  horizonHour: number;
  debrisClass: DebrisClass | "all";
  minConfidence: number;
}

interface DemoBundle {
  health: HealthStatus;
  impact: ImpactDashboard;
  forecast: ForecastSnapshot;
  benchmark: RoutingBenchmarkReport;
  route: RoutePlan;
}

export const DEMO_HEALTH: HealthStatus = {
  status: "ok",
  pilot_region: "sf_bay_estuary",
  default_horizon_hours: 24,
  scheduler_enabled: false,
  database_url: "static",
  ingest_mode: "demo",
  forecast_stale_after_minutes: 180,
  default_depot_lat: 37.8066,
  default_depot_lon: -122.4659,
};

export const DEMO_IMPACT: ImpactDashboard = {
  total_missions: 1,
  total_collected_kg: 68.098,
  total_distance_km: 42.869,
  total_hours: 3.9815,
  kg_per_vessel_km: 1.5885,
  kg_per_hour: 17.1036,
  hotspot_precision: 1,
  false_search_distance_km: 0.6,
  mission_hit_rate: 1,
  latest_updated_at: "2026-04-05T02:10:08.813928Z",
};

function demoAssetPath(fileName: string): string {
  return `${import.meta.env.BASE_URL}demo/${fileName}`;
}

async function requestDemoJson<T>(fileName: string): Promise<T> {
  const response = await fetch(demoAssetPath(fileName));
  if (!response.ok) {
    throw new Error(`Failed to load demo asset: ${fileName}`);
  }
  return response.json() as Promise<T>;
}

function toHotspotSummary(step: ForecastSnapshot["steps"][number]): HotspotSummary {
  return {
    valid_at: step.valid_at,
    horizon_hour: step.horizon_hour,
    cell_id: step.cell_id,
    lat: step.lat,
    lon: step.lon,
    debris_class: step.debris_class,
    probability: step.probability,
    expected_kg_min: step.expected_kg_min,
    expected_kg_max: step.expected_kg_max,
    confidence: step.confidence,
  };
}

export function filterDemoForecast(snapshot: ForecastSnapshot, filters: DemoFilters): ForecastSnapshot {
  const filteredSteps = snapshot.steps.filter((step) => {
    if (step.horizon_hour !== filters.horizonHour) {
      return false;
    }
    if (filters.debrisClass !== "all" && step.debris_class !== filters.debrisClass) {
      return false;
    }
    return step.confidence >= filters.minConfidence;
  });

  const topHotspots = [...filteredSteps]
    .sort((left, right) => {
      const maxDelta = right.expected_kg_max - left.expected_kg_max;
      if (maxDelta !== 0) {
        return maxDelta;
      }
      return right.confidence - left.confidence;
    })
    .slice(0, 8)
    .map(toHotspotSummary);

  const meanConfidence =
    filteredSteps.length > 0
      ? filteredSteps.reduce((total, step) => total + step.confidence, 0) / filteredSteps.length
      : 0;

  return {
    ...snapshot,
    steps: filteredSteps,
    top_hotspots: topHotspots,
    summary: {
      ...snapshot.summary,
      step_count: filteredSteps.length,
      top_expected_kg_max: topHotspots[0]?.expected_kg_max ?? 0,
      mean_confidence: meanConfidence,
      active_horizon_hour: filters.horizonHour,
      active_debris_class: filters.debrisClass,
      active_confidence_floor: filters.minConfidence,
    },
  };
}

export async function loadDemoBundle(): Promise<DemoBundle> {
  const [forecast, benchmark, route] = await Promise.all([
    requestDemoJson<ForecastSnapshot>("forecast.json"),
    requestDemoJson<RoutingBenchmarkReport>("benchmark.json"),
    requestDemoJson<RoutePlan>("route.json"),
  ]);

  return {
    health: DEMO_HEALTH,
    impact: DEMO_IMPACT,
    forecast,
    benchmark,
    route,
  };
}

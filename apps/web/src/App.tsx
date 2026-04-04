import { FormEvent, useEffect, useState } from "react";
import type { LatLngExpression } from "leaflet";
import { CircleMarker, MapContainer, Popup, Polyline, TileLayer } from "react-leaflet";
import { DEMO_NOTICE, FALLBACK_NOTICE, STATIC_DEMO, filterDemoForecast, loadDemoBundle } from "./demo";
import type {
  DebrisClass,
  ForecastProvenance,
  ForecastSnapshot,
  ForecastStep,
  HealthStatus,
  ImpactDashboard,
  RecommendedMode,
  RoutePlan,
  RoutingBenchmarkReport,
  RoutingBenchmarkStrategy,
} from "./types";

const PRODUCT_NAME = "SeaSweep";
const API_BASE = (() => {
  const configured = import.meta.env.VITE_API_BASE?.trim();
  return configured ? configured.replace(/\/$/, "") : "/api";
})();
const DEFAULT_CENTER: [number, number] = [37.8066, -122.4659];

const STRATEGY_LABELS: Record<RoutingBenchmarkStrategy["strategy"], string> = {
  nearest_hotspot: "Nearest hotspot",
  highest_yield: "Highest-yield sweep",
  recon_aware: "Recon-aware optimizer",
};

const LAUNCH_PILLARS = [
  {
    index: "01",
    title: "Forecast debris before crews leave the dock",
    copy:
      "SeaSweep turns public current and marine-state data into pilot-ready hotspot forecasts so cleanup teams can launch toward likely accumulation zones instead of burning hours searching.",
  },
  {
    index: "02",
    title: "Turn hotspots into a collection plan",
    copy:
      "The same interface compares route options, shows uncertainty, and helps operators choose between a collection mission and a quicker recon pass.",
  },
  {
    index: "03",
    title: "Prove impact with a shared ledger",
    copy:
      "Mission feedback, collection results, and benchmark reports stay in the same system so funders, ports, and research partners can see what changed on the water.",
  },
];

const PARTNER_FITS = [
  {
    label: "For NGOs and community fleets",
    detail:
      "A lightweight operations layer that makes predictive cleanup planning accessible to smaller teams, not just well-funded programs.",
  },
  {
    label: "For ports and municipal partners",
    detail:
      "A way to prioritize debris response zones, communicate mission rationale, and keep a transparent record of what was recovered.",
  },
  {
    label: "For model and data collaborators",
    detail:
      "A launch-ready product surface where better current models, debris labels, and routing logic can be integrated without building another website from scratch.",
  },
];

const MODEL_CONNECTORS = [
  {
    name: "SOCAT",
    status: "connected",
    purpose: "Ship-based observations that can support labels, calibration, and validation windows.",
  },
  {
    name: "NOAA GML CO2",
    status: "connected",
    purpose: "Atmospheric baselines for hybrid model variants and environmental context.",
  },
  {
    name: "ERA5",
    status: "testing",
    purpose: "Wind and forcing context for routing, drift, and next-step model experiments.",
  },
  {
    name: "Copernicus Marine",
    status: "testing",
    purpose: "Currents, salinity, and temperature grids for higher-fidelity drift modeling.",
  },
  {
    name: "HYCOM / NOAA currents",
    status: "planned",
    purpose: "Alternate current products and regional redundancy for route supervision.",
  },
  {
    name: "Debris observations",
    status: "planned",
    purpose: "True cleanup labels for hotspot supervision instead of hand-authored priors.",
  },
];

const MODEL_PRIORITIES = [
  "Global or tiled ERA5 fields for wind, pressure, and wave context.",
  "Copernicus or HYCOM current fields aligned to the mission mesh.",
  "Ocean color or chlorophyll layers for richer environmental drivers.",
  "Observed debris concentration labels to train directly on cleanup outcomes.",
];

const HOSTING_NOTES = [
  "GitHub Pages acts as the SeaSweep landing layer and the fallback judge demo using exported forecast, route, and benchmark artifacts.",
  "A live API deployment can power fresh forecast runs, route writes, and database-backed feedback without changing the frontend code.",
  "The Vite base path and API base are build-time configurable, so the same app can ship to repo pages now and a live hackathon endpoint later.",
];

const DEMO_FLOW_STEPS = [
  {
    label: "Predict",
    detail: "Run a fresh debris forecast and show confidence-aware hotspots on the map.",
  },
  {
    label: "Plan",
    detail: "Turn the forecast into a route recommendation and benchmark it against simpler strategies.",
  },
  {
    label: "Prove",
    detail: "Save crew feedback and update the impact ledger so the system closes the loop.",
  },
];

type DebrisFilter = DebrisClass | "all";
type AppMode = "booting" | "live" | "demo";

interface ForecastFilters {
  horizonHour: number;
  debrisClass: DebrisFilter;
  minConfidence: number;
}

interface RouteFormState {
  missionHours: number;
  vesselSpeedKmh: number;
  fuelBurnLph: number;
  depotLat: number;
  depotLon: number;
}

interface FeedbackState {
  foundStatus: "found" | "not_found";
  estimatedKg: number;
  collectedKg: number;
  photoUrl: string;
  note: string;
  routeDeviationReason: string;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function formatRange(min: number, max: number): string {
  return `${min.toFixed(1)}-${max.toFixed(1)} kg`;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function markerColor(step: ForecastStep): string {
  if (step.confidence >= 0.75) {
    return "#0b8f72";
  }
  if (step.confidence >= 0.5) {
    return "#b88921";
  }
  return "#b04d3c";
}

function buildRoutePolyline(
  route: RoutePlan | null,
  forecast: ForecastSnapshot | null,
  routeForm: RouteFormState,
): LatLngExpression[] {
  if (!route || route.recommended_mode !== "collection" || !forecast) {
    return [];
  }

  const stepByKey = new Map<string, ForecastStep>();
  forecast.steps.forEach((step) => {
    stepByKey.set(`${step.cell_id}:${step.debris_class}`, step);
    if (!stepByKey.has(step.cell_id)) {
      stepByKey.set(step.cell_id, step);
    }
  });

  const coordinates: LatLngExpression[] = [[routeForm.depotLat, routeForm.depotLon]];
  route.ordered_cell_ids.forEach((cellId) => {
    const step = stepByKey.get(cellId) ?? stepByKey.get(cellId.split(":")[0]);
    if (step) {
      coordinates.push([step.lat, step.lon]);
    }
  });
  coordinates.push([routeForm.depotLat, routeForm.depotLon]);
  return coordinates;
}

function topLine(snapshot: ForecastSnapshot | null): string {
  if (!snapshot || snapshot.top_hotspots.length === 0) {
    return "No active hotspot forecast";
  }
  const top = snapshot.top_hotspots[0];
  return `${top.cell_id.replaceAll("_", " ")} ${formatRange(top.expected_kg_min, top.expected_kg_max)}`;
}

function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatOptionalTimestamp(value: string | null | undefined): string {
  return value ? formatTimestamp(value) : "Awaiting first mission";
}

function formatFreshness(ageMinutes: number): string {
  if (ageMinutes < 60) {
    return `${ageMinutes}m old`;
  }
  const hours = Math.floor(ageMinutes / 60);
  const minutes = ageMinutes % 60;
  return `${hours}h ${minutes}m old`;
}

function sourceLabel(provenance: ForecastProvenance): string {
  const transition = `${provenance.source_mode_requested} -> ${provenance.source_mode_used}`;
  return provenance.is_fallback ? `${transition} fallback` : transition;
}

function confidenceBand(snapshot: ForecastSnapshot | null): string {
  if (!snapshot) {
    return "Unknown";
  }
  const meanConfidence = snapshot.summary.mean_confidence;
  if (typeof meanConfidence !== "number") {
    return "Unknown";
  }
  if (meanConfidence >= 0.75) {
    return "High";
  }
  if (meanConfidence >= 0.5) {
    return "Medium";
  }
  return "Recon";
}

function strategyLabel(strategy: RoutingBenchmarkStrategy["strategy"] | undefined): string {
  if (!strategy) {
    return "Awaiting benchmark";
  }
  return STRATEGY_LABELS[strategy];
}

function benchmarkSummary(benchmark: RoutingBenchmarkReport | null): string {
  if (!benchmark?.compared_strategies?.length) {
    return "No benchmark report yet";
  }
  const winner = benchmark.compared_strategies.find((entry) => entry.strategy === benchmark.winning_strategy);
  if (!winner) {
    return strategyLabel(benchmark.winning_strategy);
  }
  return `${strategyLabel(winner.strategy)} | score ${winner.objective_score.toFixed(1)}`;
}

function apiBaseLabel(base: string): string {
  if (base.startsWith("http://") || base.startsWith("https://")) {
    return base;
  }
  return `relative ${base}`;
}

export function App() {
  const [appMode, setAppMode] = useState<AppMode>("booting");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [forecast, setForecast] = useState<ForecastSnapshot | null>(null);
  const [impact, setImpact] = useState<ImpactDashboard | null>(null);
  const [benchmark, setBenchmark] = useState<RoutingBenchmarkReport | null>(null);
  const [route, setRoute] = useState<RoutePlan | null>(null);
  const [demoSourceForecast, setDemoSourceForecast] = useState<ForecastSnapshot | null>(null);
  const [filters, setFilters] = useState<ForecastFilters>({
    horizonHour: 24,
    debrisClass: "all",
    minConfidence: 0.25,
  });
  const [routeForm, setRouteForm] = useState<RouteFormState>({
    missionHours: 4,
    vesselSpeedKmh: 18,
    fuelBurnLph: 12,
    depotLat: DEFAULT_CENTER[0],
    depotLon: DEFAULT_CENTER[1],
  });
  const [feedback, setFeedback] = useState<FeedbackState>({
    foundStatus: "found",
    estimatedKg: 6,
    collectedKg: 8,
    photoUrl: "",
    note: "",
    routeDeviationReason: "",
  });
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [loadingForecast, setLoadingForecast] = useState(true);
  const [loadingBenchmark, setLoadingBenchmark] = useState(true);
  const [loadingRoute, setLoadingRoute] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [forecastMissing, setForecastMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [feedbackMessage, setFeedbackMessage] = useState("");

  async function activateDemoMode(message: string) {
    const bundle = await loadDemoBundle();
    const filteredForecast = filterDemoForecast(bundle.forecast, filters);

    setAppMode("demo");
    setDemoSourceForecast(bundle.forecast);
    setHealth(bundle.health);
    setImpact(bundle.impact);
    setBenchmark(bundle.benchmark);
    setRoute(bundle.route);
    setForecast(filteredForecast);
    setForecastMissing(filteredForecast.steps.length === 0);
    setRouteForm((current) => ({
      ...current,
      depotLat: bundle.health.default_depot_lat,
      depotLon: bundle.health.default_depot_lon,
    }));
    setStatusMessage(message);
    setError(null);
    setLoadingHealth(false);
    setLoadingForecast(false);
    setLoadingBenchmark(false);
  }

  async function loadImpactDashboard() {
    try {
      const dashboard = await requestJson<ImpactDashboard>("/impact/dashboard");
      setImpact(dashboard);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to load impact dashboard.");
    }
  }

  async function loadBenchmarkReport() {
    setLoadingBenchmark(true);
    try {
      const report = await requestJson<RoutingBenchmarkReport>("/impact/benchmarks/latest");
      setBenchmark(report);
    } catch (caught) {
      if (caught instanceof Error && caught.message.includes("No forecast")) {
        setBenchmark(null);
      } else {
        setError(caught instanceof Error ? caught.message : "Failed to load routing benchmark.");
      }
    } finally {
      setLoadingBenchmark(false);
    }
  }

  async function loadForecastSnapshot(activeFilters: ForecastFilters) {
    setLoadingForecast(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      params.set("horizon_hour", String(activeFilters.horizonHour));
      params.set("min_confidence", String(activeFilters.minConfidence));
      if (activeFilters.debrisClass !== "all") {
        params.set("class", activeFilters.debrisClass);
      }
      const snapshot = await requestJson<ForecastSnapshot>(`/forecast/latest?${params.toString()}`);
      setForecast(snapshot);
      setForecastMissing(false);
    } catch (caught) {
      if (caught instanceof Error && caught.message.includes("No forecast")) {
        setForecast(null);
        setForecastMissing(true);
      } else {
        setError(caught instanceof Error ? caught.message : "Failed to load forecast.");
      }
    } finally {
      setLoadingForecast(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      if (STATIC_DEMO) {
        try {
          await activateDemoMode(DEMO_NOTICE);
        } catch (caught) {
          if (!cancelled) {
            setError(caught instanceof Error ? caught.message : "Failed to load the hosted SeaSweep demo.");
            setLoadingHealth(false);
            setLoadingForecast(false);
            setLoadingBenchmark(false);
          }
        }
        return;
      }

      try {
        const status = await requestJson<HealthStatus>("/health");
        if (cancelled) {
          return;
        }
        setAppMode("live");
        setHealth(status);
        setFilters((current) => ({ ...current, horizonHour: status.default_horizon_hours }));
        setRouteForm((current) => ({
          ...current,
          depotLat: status.default_depot_lat,
          depotLon: status.default_depot_lon,
        }));
        setStatusMessage("");
      } catch (caught) {
        if (cancelled) {
          return;
        }
        try {
          await activateDemoMode(FALLBACK_NOTICE);
        } catch {
          setError(caught instanceof Error ? caught.message : "Failed to load system status.");
          setLoadingForecast(false);
          setLoadingBenchmark(false);
        }
      } finally {
        if (!cancelled) {
          setLoadingHealth(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (appMode !== "live") {
      return;
    }
    void loadImpactDashboard();
    void loadBenchmarkReport();
  }, [appMode]);

  useEffect(() => {
    if (appMode !== "live") {
      return;
    }
    void loadForecastSnapshot(filters);
  }, [appMode, filters.horizonHour, filters.debrisClass, filters.minConfidence]);

  useEffect(() => {
    if (appMode !== "demo" || !demoSourceForecast) {
      return;
    }
    const filteredForecast = filterDemoForecast(demoSourceForecast, filters);
    setForecast(filteredForecast);
    setForecastMissing(filteredForecast.steps.length === 0);
  }, [appMode, demoSourceForecast, filters.horizonHour, filters.debrisClass, filters.minConfidence]);

  async function handleRunForecast() {
    setLoadingForecast(true);
    setError(null);

    if (appMode === "demo") {
      try {
        await activateDemoMode(DEMO_NOTICE);
        setStatusMessage("SeaSweep demo refreshed from the seeded San Francisco Bay scenario.");
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Failed to refresh the hosted demo.");
        setLoadingForecast(false);
      }
      return;
    }

    try {
      await requestJson("/forecast/run", {
        method: "POST",
        body: JSON.stringify({
          horizon_hours: filters.horizonHour,
          debris_classes: ["low", "high"],
          source_mode: "auto",
        }),
      });
      await loadForecastSnapshot(filters);
      await loadBenchmarkReport();
      setStatusMessage("Forecast refreshed from the live operations API.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to run forecast.");
      setLoadingForecast(false);
    }
  }

  async function handleOptimizeRoute() {
    setLoadingRoute(true);
    setError(null);

    if (appMode === "demo") {
      try {
        const bundle = await loadDemoBundle();
        setRoute(bundle.route);
        setBenchmark(bundle.benchmark);
        setStatusMessage("Showing the seeded collection route used in the SeaSweep judge demo.");
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Failed to load the seeded route.");
      } finally {
        setLoadingRoute(false);
      }
      return;
    }

    try {
      const nextRoute = await requestJson<RoutePlan>("/route/optimize", {
        method: "POST",
        body: JSON.stringify({
          depot_lat: routeForm.depotLat,
          depot_lon: routeForm.depotLon,
          mission_hours: routeForm.missionHours,
          vessel_speed_kmh: routeForm.vesselSpeedKmh,
          fuel_burn_lph: routeForm.fuelBurnLph,
          target_horizon_hour: filters.horizonHour,
          min_confidence: filters.minConfidence,
        }),
      });
      setRoute(nextRoute);
      setFeedbackMessage("");
      setStatusMessage("Route updated from the live operations API.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to optimize route.");
    } finally {
      setLoadingRoute(false);
    }
  }

  async function handleSubmitFeedback(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!route) {
      return;
    }

    if (appMode === "demo") {
      setFeedbackMessage(
        "This GitHub Pages SeaSweep demo is read-only. Connect the live API deployment to save mission feedback.",
      );
      return;
    }

    setSubmittingFeedback(true);
    setError(null);
    try {
      await requestJson("/observations/upload", {
        method: "POST",
        body: JSON.stringify({
          mission_id: route.mission_id,
          observed_at: new Date().toISOString(),
          lat: routeForm.depotLat,
          lon: routeForm.depotLon,
          found_status: feedback.foundStatus,
          debris_class: filters.debrisClass === "all" ? "low" : filters.debrisClass,
          estimated_kg: feedback.estimatedKg,
          confidence: 0.8,
          photo_url: feedback.photoUrl || null,
          note: feedback.note || null,
          route_deviation_reason: feedback.routeDeviationReason || null,
        }),
      });
      await requestJson("/cleanup/log", {
        method: "POST",
        body: JSON.stringify({
          mission_id: route.mission_id,
          completed_at: new Date().toISOString(),
          vessel_id: route.vessel_id,
          recommended_mode: route.recommended_mode as RecommendedMode,
          predicted_kg_min: route.expected_kg_min,
          predicted_kg_max: route.expected_kg_max,
          collected_kg: feedback.collectedKg,
          vessel_distance_km: route.expected_distance_km,
          vessel_hours: Math.max(route.expected_duration_min / 60, 0.5),
          hotspot_hits: feedback.foundStatus === "found" ? 1 : 0,
          hotspot_misses: feedback.foundStatus === "not_found" ? 1 : 0,
          false_search_km: feedback.foundStatus === "not_found" ? route.expected_distance_km : 0,
          fuel_liters: route.estimated_fuel_liters,
          route_deviation_reason: feedback.routeDeviationReason || null,
          notes: feedback.note || null,
        }),
      });
      await loadImpactDashboard();
      await loadBenchmarkReport();
      setFeedbackMessage("Mission feedback saved to the impact ledger.");
      setStatusMessage("Impact ledger updated from crew feedback.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to submit mission feedback.");
    } finally {
      setSubmittingFeedback(false);
    }
  }

  const routePolyline = buildRoutePolyline(route, forecast, routeForm);
  const activeProvenance = forecast?.provenance ?? null;
  const routeProvenance = route?.forecast_provenance ?? null;
  const routeReason =
    route && typeof route.metadata.reason === "string" ? route.metadata.reason : "No recon reason supplied.";
  const winningStrategy = benchmark?.winning_strategy;
  const isDemoMode = appMode === "demo";
  const modeLabel = loadingHealth ? "Checking runtime" : isDemoMode ? "Pages demo mode" : "Live API mode";
  const modeDetail = loadingHealth
    ? "Confirming whether SeaSweep is using seeded artifacts or the live API."
    : isDemoMode
      ? "GitHub Pages is showing the seeded San Francisco Bay scenario so the story always works on stage."
      : "This frontend is calling the external API directly for live forecasts, routing, and feedback writes.";
  const apiLabel = isDemoMode ? "Seeded mission data" : apiBaseLabel(API_BASE);
  const forecastButtonLabel = isDemoMode
    ? loadingForecast
      ? "Refreshing demo..."
      : "Refresh demo scenario"
    : loadingForecast
      ? "Refreshing forecast..."
      : "Run fresh forecast";
  const optimizeRouteLabel = isDemoMode
    ? loadingRoute
      ? "Loading seeded route..."
      : "Load seeded route"
    : loadingRoute
      ? "Optimizing..."
      : "Optimize route";
  const readinessCards = [
    {
      label: "Forecast engine",
      value: activeProvenance ? sourceLabel(activeProvenance) : "Waiting for first run",
      detail: activeProvenance
        ? `${formatFreshness(activeProvenance.age_minutes)} with ${confidenceBand(forecast)} confidence`
        : "Run or seed a forecast to populate the operations stack.",
    },
    {
      label: "Mission plan",
      value: route ? route.recommended_mode : strategyLabel(winningStrategy),
      detail: route
        ? `${route.ordered_cell_ids.length || route.alternates.length} destination(s) staged for this mission window.`
        : benchmarkSummary(benchmark),
    },
    {
      label: "Impact ledger",
      value: impact ? `${impact.total_missions} logged mission(s)` : "Ledger warming up",
      detail: impact
        ? `${formatPercent(impact.hotspot_precision)} hotspot precision with ${impact.total_collected_kg.toFixed(1)} kg recorded.`
        : "Mission feedback will unlock precision and hit-rate reporting.",
    },
    {
      label: "Demo mode",
      value: isDemoMode ? "Pages landing + seeded story" : "Pages landing + live API handoff",
      detail: isDemoMode
        ? "Seeded artifacts, read-only controls, and zero backend dependency for partner walkthroughs."
        : "Fresh forecasts, route writes, and database-backed feedback are available in the live stack.",
    },
  ];
  const summaryCards = [
    {
      label: "Pilot region",
      value: health?.pilot_region ?? "sf_bay_estuary",
      detail: loadingHealth ? "Checking runtime status..." : "San Francisco Bay pilot coverage.",
    },
    {
      label: "Top hotspot",
      value: topLine(forecast),
      detail: activeProvenance
        ? `${formatFreshness(activeProvenance.age_minutes)} | ${confidenceBand(forecast)} confidence`
        : "Load the demo or run a forecast to populate hotspot ranking.",
    },
    {
      label: "Benchmark winner",
      value: benchmark ? strategyLabel(benchmark.winning_strategy) : "Awaiting benchmark",
      detail: loadingBenchmark
        ? "Comparing route strategies..."
        : benchmark
          ? `h+${benchmark.target_horizon_hour} | ${benchmark.compared_strategies.length} strategies compared`
          : "Benchmark report will appear after the first forecast cycle.",
    },
    {
      label: "Kg per vessel-km",
      value: impact ? impact.kg_per_vessel_km.toFixed(2) : "0.00",
      detail: impact
        ? `${formatPercent(impact.mission_hit_rate)} mission hit rate`
        : "Impact ledger waiting for field feedback.",
    },
  ];
  const exportLinks = isDemoMode
    ? [
        { href: `${import.meta.env.BASE_URL}demo/forecast.json`, label: "Forecast JSON" },
        { href: `${import.meta.env.BASE_URL}demo/route.json`, label: "Route JSON" },
      ]
    : [
        { href: `${API_BASE}/export/pdf-brief`, label: "PDF brief" },
        { href: `${API_BASE}/export/geojson?horizon_hour=${filters.horizonHour}`, label: "GeoJSON" },
      ];

  return (
    <div className="site-shell">
      <div className="page-shell">
        <header className="site-header">
          <a className="brand" href="#top" aria-label="OceanRoute home">
            <span className="brand-mark" aria-hidden="true"></span>
            <span>OceanRoute</span>
          </a>
          <nav className="site-nav" aria-label="Primary">
            <a href="#platform">Platform</a>
            <a href="#operations">Demo</a>
            <a href="#roadmap">Roadmap</a>
            <a href="#impact">Impact</a>
          </nav>
          <a className="button button-ghost" href="#operations">
            Explore the pilot
          </a>
        </header>

        <main id="top">
          <section className="hero section">
            <div className="hero-copy-block reveal">
              <p className="eyebrow">Cleanup intelligence for public-interest fleets</p>
              <h1>Help cleanup crews spend less time searching and more time collecting.</h1>
              <p className="lead">
                OceanRoute turns public ocean data into debris hotspot forecasts, route plans, and transparent mission
                reporting for NGOs, port partners, and local fleets. This launch experience is built to show the full
                cleanup workflow and stay ready for the next model handoff.
              </p>
              <div className="hero-actions">
                <a className="button" href="#operations">
                  View the launch demo
                </a>
                <button className="button button-secondary" onClick={handleRunForecast} type="button" disabled={loadingForecast}>
                  {forecastButtonLabel}
                </button>
              </div>
              <div className="hero-notes">
                <span>24-72h hotspot forecast window</span>
                <span>Route planning plus impact reporting</span>
                <span>{isDemoMode ? "Hosted Pages demo" : "Live API connected"}</span>
              </div>
            </div>

            <div className="hero-panel-stack reveal">
              <article className="signal-card">
                <p className="signal-label">Launch snapshot</p>
                <div className="signal-grid">
                  {readinessCards.map((card) => (
                    <article key={card.label}>
                      <span>{card.label}</span>
                      <strong>{card.value}</strong>
                      <small>{card.detail}</small>
                    </article>
                  ))}
                </div>
              </article>

              <article className="route-card">
                <div className="route-card__header">
                  <div>
                    <p className="route-kicker">From hotspot to field log</p>
                    <h2>Pilot mission preview</h2>
                  </div>
                  <span className="status-pill status-teal">{isDemoMode ? "Hosted demo" : "Live stack"}</span>
                </div>
                <div className="route-map" aria-hidden="true">
                  <span className="route-point route-point--a"></span>
                  <span className="route-point route-point--b"></span>
                  <span className="route-point route-point--c"></span>
                  <span className="route-line route-line--one"></span>
                  <span className="route-line route-line--two"></span>
                  <span className="heat heat--one"></span>
                  <span className="heat heat--two"></span>
                  <span className="heat heat--three"></span>
                </div>
                <div className="route-legend">
                  <div>
                    <span className="legend-swatch legend-swatch--hot"></span>
                    {topLine(forecast)}
                  </div>
                  <div>
                    <span className="legend-swatch legend-swatch--route"></span>
                    {route ? `${route.recommended_mode} mission plan ready` : benchmarkSummary(benchmark)}
                  </div>
                </div>
              </article>
            </div>
          </section>

          <section className="stats-strip" aria-label="Platform summary">
            {summaryCards.map((card) => (
              <article className="stat-card reveal" key={card.label}>
                <span className="summary-label">{card.label}</span>
                <strong>{card.value}</strong>
                <p>{card.detail}</p>
              </article>
            ))}
          </section>

          {statusMessage ? <div className="message-banner message-banner-info">{statusMessage}</div> : null}
          {error ? <div className="message-banner message-banner-error">{error}</div> : null}

          <section id="platform" className="section">
            <div className="section-heading reveal">
              <p className="eyebrow">What OceanRoute delivers at launch</p>
              <h2>A single product surface for forecasting, routing, and reporting cleanup work.</h2>
              <p>
                The launch site is designed to communicate the mission clearly to partners while still showing a real
                operational workflow that can evolve with better models and richer data.
              </p>
            </div>

            <div className="problem-grid">
              {LAUNCH_PILLARS.map((pillar) => (
                <article className="panel-card reveal" key={pillar.index}>
                  <span className="card-index">{pillar.index}</span>
                  <h3>{pillar.title}</h3>
                  <p>{pillar.copy}</p>
                </article>
              ))}
            </div>

            <div className="comparison-grid">
              {PARTNER_FITS.map((item) => (
                <article className="comparison-card reveal" key={item.label}>
                  <h3>{item.label}</h3>
                  <p>{item.detail}</p>
                </article>
              ))}
            </div>
          </section>

          <section id="operations" className="section section-tint">
            <div className="section-heading reveal">
              <p className="eyebrow">Launch demo</p>
              <h2>Forecast hotspots, preview a route, and show the trust signals behind every recommendation.</h2>
              <p>
                This is the part partners can explore immediately. In the hosted Pages version it runs on seeded demo
                artifacts, and in the live deployment it connects to the API for fresh runs and persisted feedback.
              </p>
            </div>

            <div className="operations-grid">
              <section className="workspace-card workspace-card-map">
                <div className="workspace-header">
                  <div>
                    <h3>Operations map</h3>
                    <p>Confidence-aware hotspots for the selected forecast horizon.</p>
                  </div>
                  <div className="filter-grid">
                    <label>
                      Horizon
                      <select
                        aria-label="Forecast horizon"
                        value={filters.horizonHour}
                        onChange={(event) =>
                          setFilters((current) => ({ ...current, horizonHour: Number(event.target.value) }))
                        }
                      >
                        <option value={24}>24h</option>
                        <option value={48}>48h</option>
                        <option value={72}>72h</option>
                      </select>
                    </label>
                    <label>
                      Debris class
                      <select
                        aria-label="Debris class filter"
                        value={filters.debrisClass}
                        onChange={(event) =>
                          setFilters((current) => ({
                            ...current,
                            debrisClass: event.target.value as DebrisFilter,
                          }))
                        }
                      >
                        <option value="all">All</option>
                        <option value="low">Low-windage</option>
                        <option value="high">High-windage</option>
                      </select>
                    </label>
                    <label>
                      Confidence floor
                      <input
                        aria-label="Confidence floor"
                        type="range"
                        min="0"
                        max="0.8"
                        step="0.05"
                        value={filters.minConfidence}
                        onChange={(event) =>
                          setFilters((current) => ({
                            ...current,
                            minConfidence: Number(event.target.value),
                          }))
                        }
                      />
                      <span>{formatPercent(filters.minConfidence)}</span>
                    </label>
                  </div>
                </div>

                {activeProvenance ? (
                  <div className="pill-row" aria-label="Forecast trust summary">
                    <span className="status-pill status-teal">{sourceLabel(activeProvenance)}</span>
                    <span className="status-pill">Generated {formatTimestamp(activeProvenance.generated_at)}</span>
                    <span className={`status-pill ${activeProvenance.is_stale ? "status-warm" : "status-foam"}`}>
                      {activeProvenance.is_stale ? "stale" : "fresh"} within {activeProvenance.stale_after_minutes}m
                    </span>
                    <span className="status-pill">Confidence {confidenceBand(forecast)}</span>
                  </div>
                ) : null}

                <div className="legend-row">
                  <span className="legend-chip legend-high">High confidence</span>
                  <span className="legend-chip legend-medium">Usable with caution</span>
                  <span className="legend-chip legend-low">Recon only</span>
                </div>

                <div className="map-shell">
                  <MapContainer center={DEFAULT_CENTER} zoom={10} scrollWheelZoom className="leaflet-map">
                    <TileLayer
                      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    {forecast?.steps.map((step) => (
                      <CircleMarker
                        key={`${step.cell_id}-${step.debris_class}-${step.horizon_hour}`}
                        center={[step.lat, step.lon]}
                        radius={6 + Math.round(step.probability * 10)}
                        pathOptions={{ color: markerColor(step), fillOpacity: 0.48, weight: 1.2 }}
                      >
                        <Popup>
                          <strong>{step.cell_id.replaceAll("_", " ")}</strong>
                          <br />
                          {step.debris_class} debris, h+{step.horizon_hour}
                          <br />
                          {formatRange(step.expected_kg_min, step.expected_kg_max)}
                          <br />
                          Confidence {formatPercent(step.confidence)}
                        </Popup>
                      </CircleMarker>
                    ))}
                    {routePolyline.length > 1 ? (
                      <Polyline pathOptions={{ color: "#2b736d", weight: 4 }} positions={routePolyline} />
                    ) : null}
                  </MapContainer>
                </div>

                {forecastMissing ? (
                  <p className="empty-state">No hotspots match the active demo filters yet. Widen the horizon or lower the confidence floor.</p>
                ) : null}

                {forecast?.source_notes.length ? (
                  <div className="source-notes">
                    {forecast.source_notes.map((note) => (
                      <p key={note}>{note}</p>
                    ))}
                  </div>
                ) : null}
              </section>

              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Hotspot ranking</h3>
                    <p>Top cells after filtering by horizon and confidence.</p>
                  </div>
                  <span className="status-pill status-foam">{forecast?.steps.length ?? 0} cells</span>
                </div>

                <div className="hotspot-list" aria-label="Hotspot ranking">
                  {forecast?.top_hotspots.map((hotspot) => (
                    <article className="hotspot-card" key={`${hotspot.cell_id}-${hotspot.debris_class}-${hotspot.horizon_hour}`}>
                      <div>
                        <h3>{hotspot.cell_id.replaceAll("_", " ")}</h3>
                        <p>
                          {hotspot.debris_class} debris, h+{hotspot.horizon_hour}, confidence{" "}
                          {formatPercent(hotspot.confidence)}
                        </p>
                      </div>
                      <strong>{formatRange(hotspot.expected_kg_min, hotspot.expected_kg_max)}</strong>
                    </article>
                  ))}
                  {!forecast?.top_hotspots.length && !loadingForecast ? (
                    <p className="empty-state">No hotspots match the current filters.</p>
                  ) : null}
                </div>

                <div className="section-divider"></div>

                <div className="workspace-header">
                  <div>
                    <h3>Mission planner</h3>
                    <p>{isDemoMode ? "Load the seeded launch route or explore the live route optimizer." : "Optimize a single-vessel route or return a recon recommendation."}</p>
                  </div>
                  <button
                    className="button button-secondary"
                    onClick={handleOptimizeRoute}
                    type="button"
                    disabled={loadingRoute || !forecast}
                  >
                    {optimizeRouteLabel}
                  </button>
                </div>

                <div className="form-grid compact-grid">
                  <label>
                    Mission hours
                    <input
                      aria-label="Mission hours"
                      type="number"
                      min="1"
                      max="12"
                      step="0.5"
                      value={routeForm.missionHours}
                      onChange={(event) =>
                        setRouteForm((current) => ({ ...current, missionHours: Number(event.target.value) }))
                      }
                    />
                  </label>
                  <label>
                    Vessel speed (km/h)
                    <input
                      aria-label="Vessel speed"
                      type="number"
                      min="4"
                      max="40"
                      step="1"
                      value={routeForm.vesselSpeedKmh}
                      onChange={(event) =>
                        setRouteForm((current) => ({ ...current, vesselSpeedKmh: Number(event.target.value) }))
                      }
                    />
                  </label>
                  <label>
                    Fuel burn (L/h)
                    <input
                      aria-label="Fuel burn"
                      type="number"
                      min="1"
                      max="60"
                      step="1"
                      value={routeForm.fuelBurnLph}
                      onChange={(event) =>
                        setRouteForm((current) => ({ ...current, fuelBurnLph: Number(event.target.value) }))
                      }
                    />
                  </label>
                </div>
              </section>

              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Route summary</h3>
                    <p>Collection route when the expected yield clears the threshold, recon otherwise.</p>
                  </div>
                  <span className={`status-pill status-mode-${route?.recommended_mode ?? "idle"}`}>
                    {route ? route.recommended_mode : "idle"}
                  </span>
                </div>

                {route ? (
                  <>
                    <div className="metric-grid">
                      <article className="metric-card">
                        <span>Yield range</span>
                        <strong>{formatRange(route.expected_kg_min, route.expected_kg_max)}</strong>
                      </article>
                      <article className="metric-card">
                        <span>Distance</span>
                        <strong>{route.expected_distance_km.toFixed(1)} km</strong>
                      </article>
                      <article className="metric-card">
                        <span>Fuel</span>
                        <strong>{route.estimated_fuel_liters.toFixed(1)} L</strong>
                      </article>
                      <article className="metric-card">
                        <span>Risk-adjusted confidence</span>
                        <strong>{formatPercent(Math.max(0, 1 - route.uncertainty_risk))}</strong>
                      </article>
                    </div>

                    <div className="detail-stack">
                      <div>
                        <h4>Forecast provenance</h4>
                        {routeProvenance ? (
                          <p>
                            {sourceLabel(routeProvenance)} | generated {formatTimestamp(routeProvenance.generated_at)} |
                            h+{route.target_horizon_hour} | {route.forecast_is_stale ? "stale" : "fresh"} | confidence{" "}
                            {confidenceBand(forecast)}
                          </p>
                        ) : (
                          <p>Route was optimized from ad hoc candidates rather than a stored forecast run.</p>
                        )}
                      </div>

                      <div>
                        <h4>Ordered cells</h4>
                        {route.ordered_cell_ids.length ? (
                          <ol className="sequence-list">
                            {route.ordered_cell_ids.map((cellId) => (
                              <li key={cellId}>{cellId}</li>
                            ))}
                          </ol>
                        ) : (
                          <p className="empty-state">{routeReason}</p>
                        )}
                      </div>

                      <div>
                        <h4>Alternates</h4>
                        <p>{route.alternates.join(", ") || "No alternates available."}</p>
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="empty-state">{isDemoMode ? "Load the seeded route to preview the launch mission." : "Optimize a route after loading a forecast."}</p>
                )}
              </section>
            </div>
          </section>

          <section id="roadmap" className="section">
            <div className="section-heading reveal">
              <p className="eyebrow">Model and deployment roadmap</p>
              <h2>Launch with a stable demo surface now, then keep layering in better models and data.</h2>
              <p>
                The product is already structured around the eventual model handoff. Connectors, benchmark comparison,
                and launch-hosting choices are all visible in the same place.
              </p>
            </div>

            <div className="model-grid">
              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Readiness rails</h3>
                    <p>The end-to-end loop that the next model can plug into.</p>
                  </div>
                </div>
                <div className="readiness-grid">
                  {readinessCards.map((card) => (
                    <article className="metric-card" key={card.label}>
                      <span>{card.label}</span>
                      <strong>{card.value}</strong>
                      <p>{card.detail}</p>
                    </article>
                  ))}
                </div>
              </section>

              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Connector blueprint</h3>
                    <p>Useful targets for the next model wiring pass.</p>
                  </div>
                </div>
                <div className="connector-grid">
                  {MODEL_CONNECTORS.map((connector) => (
                    <article className="connector-card" key={connector.name}>
                      <div className="connector-header">
                        <h4>{connector.name}</h4>
                        <span className={`status-pill status-${connector.status}`}>{connector.status}</span>
                      </div>
                      <p>{connector.purpose}</p>
                    </article>
                  ))}
                </div>
              </section>

              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Routing benchmark</h3>
                    <p>Compare baseline strategies before attaching a richer predictive model.</p>
                  </div>
                  <span className="status-pill status-foam">
                    {benchmark ? `Winner: ${strategyLabel(benchmark.winning_strategy)}` : "Awaiting benchmark"}
                  </span>
                </div>

                {benchmark?.compared_strategies?.length ? (
                  <div className="benchmark-grid">
                    {benchmark.compared_strategies.map((entry) => (
                      <article
                        className={`benchmark-card ${entry.strategy === benchmark.winning_strategy ? "benchmark-card--winner" : ""}`}
                        key={entry.strategy}
                      >
                        <div className="connector-header">
                          <h4>{strategyLabel(entry.strategy)}</h4>
                          <span className={`status-pill status-mode-${entry.recommended_mode}`}>{entry.recommended_mode}</span>
                        </div>
                        <div className="benchmark-metrics">
                          <span>Objective</span>
                          <strong>{entry.objective_score.toFixed(2)}</strong>
                          <span>Distance</span>
                          <strong>{entry.expected_distance_km.toFixed(1)} km</strong>
                          <span>Hit rate</span>
                          <strong>{formatPercent(entry.hotspot_hit_rate_estimate)}</strong>
                        </div>
                        <p>{entry.ordered_cell_ids.join(", ") || "Recon fallback path."}</p>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="empty-state">
                    {loadingBenchmark
                      ? "Loading benchmark comparison..."
                      : "Run a forecast cycle to compare route strategies."}
                  </p>
                )}
              </section>

              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Next model priorities</h3>
                    <p>The highest-signal additions for the next integration pass.</p>
                  </div>
                </div>
                <ul className="priority-list">
                  {MODEL_PRIORITIES.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
                <div className="api-surface">
                  <span>API surface ready now</span>
                  <code>/api/forecast/latest</code>
                  <code>/api/route/optimize</code>
                  <code>/api/impact/dashboard</code>
                  <code>/api/impact/benchmarks/latest</code>
                  <code>/api/export/pdf-brief</code>
                </div>
              </section>

              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Launch hosting path</h3>
                    <p>How the launch site is packaged today and how it grows into the live stack.</p>
                  </div>
                </div>
                <ul className="priority-list">
                  {HOSTING_NOTES.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </section>
            </div>
          </section>

          <section id="impact" className="section section-tint">
            <div className="section-heading reveal">
              <p className="eyebrow">Impact and reporting</p>
              <h2>Keep mission verification, sponsor reporting, and future model tuning in one ledger.</h2>
              <p>
                This is where operations becomes learning. Field teams can verify what they found, while the same data
                powers impact reporting and the next round of model improvement.
              </p>
            </div>

            <div className="impact-layout">
              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Mission feedback</h3>
                    <p>Write back what the crew found so the forecast and impact ledger can learn.</p>
                  </div>
                </div>
                {isDemoMode ? (
                  <p className="workspace-note">
                    The GitHub Pages launch demo is read-only. Use the live API deployment to save feedback and write to
                    the impact ledger.
                  </p>
                ) : null}
                <form className="feedback-form" onSubmit={handleSubmitFeedback}>
                  <label>
                    Found status
                    <select
                      aria-label="Found status"
                      value={feedback.foundStatus}
                      onChange={(event) =>
                        setFeedback((current) => ({
                          ...current,
                          foundStatus: event.target.value as "found" | "not_found",
                        }))
                      }
                    >
                      <option value="found">Found debris</option>
                      <option value="not_found">No debris</option>
                    </select>
                  </label>
                  <label>
                    Estimated kg observed
                    <input
                      aria-label="Estimated kilograms"
                      type="number"
                      min="0"
                      step="0.5"
                      value={feedback.estimatedKg}
                      onChange={(event) =>
                        setFeedback((current) => ({ ...current, estimatedKg: Number(event.target.value) }))
                      }
                    />
                  </label>
                  <label>
                    Collected kg
                    <input
                      aria-label="Collected kilograms"
                      type="number"
                      min="0"
                      step="0.5"
                      value={feedback.collectedKg}
                      onChange={(event) =>
                        setFeedback((current) => ({ ...current, collectedKg: Number(event.target.value) }))
                      }
                    />
                  </label>
                  <label>
                    Photo URL
                    <input
                      aria-label="Photo URL"
                      type="url"
                      value={feedback.photoUrl}
                      onChange={(event) => setFeedback((current) => ({ ...current, photoUrl: event.target.value }))}
                    />
                  </label>
                  <label>
                    Route deviation reason
                    <textarea
                      aria-label="Route deviation reason"
                      value={feedback.routeDeviationReason}
                      onChange={(event) =>
                        setFeedback((current) => ({
                          ...current,
                          routeDeviationReason: event.target.value,
                        }))
                      }
                    />
                  </label>
                  <label>
                    Crew notes
                    <textarea
                      aria-label="Crew notes"
                      value={feedback.note}
                      onChange={(event) => setFeedback((current) => ({ ...current, note: event.target.value }))}
                    />
                  </label>
                  <button className="button" type="submit" disabled={isDemoMode || !route || submittingFeedback}>
                    {isDemoMode ? "Read-only in launch demo" : submittingFeedback ? "Saving feedback..." : "Save mission feedback"}
                  </button>
                  {feedbackMessage ? <p className="message-banner message-banner-success">{feedbackMessage}</p> : null}
                </form>
              </section>

              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Impact ledger</h3>
                    <p>Operational metrics over logged missions and exports for downstream reporting.</p>
                  </div>
                </div>

                <div className="metric-grid">
                  <article className="metric-card">
                    <span>Kg per hour</span>
                    <strong>{impact ? impact.kg_per_hour.toFixed(2) : "0.00"}</strong>
                    <p>Average collection productivity across logged missions.</p>
                  </article>
                  <article className="metric-card">
                    <span>Hotspot precision</span>
                    <strong>{impact ? formatPercent(impact.hotspot_precision) : "0%"}</strong>
                    <p>How often forecasted hotspots actually turned into useful cleanup targets.</p>
                  </article>
                  <article className="metric-card">
                    <span>Mission hit rate</span>
                    <strong>{impact ? formatPercent(impact.mission_hit_rate) : "0%"}</strong>
                    <p>Share of logged missions that produced debris recovery.</p>
                  </article>
                  <article className="metric-card">
                    <span>False-search km</span>
                    <strong>{impact ? impact.false_search_distance_km.toFixed(1) : "0.0"}</strong>
                    <p>Distance spent looking without collection, useful for route and model tuning.</p>
                  </article>
                </div>

                <div className="detail-stack">
                  <div>
                    <h4>Latest ledger update</h4>
                    <p>{formatOptionalTimestamp(impact?.latest_updated_at)}</p>
                  </div>
                  <div>
                    <h4>Mission exports</h4>
                    <p className="export-links">
                      {exportLinks.map((link) => (
                        <a href={link.href} key={link.label} target="_blank" rel="noreferrer">
                          {link.label}
                        </a>
                      ))}
                    </p>
                  </div>
                </div>
              </section>
            </div>
          </section>
        </main>

        <footer className="site-footer">
          <p>OceanRoute is built to launch as a shareable static demo now and grow into a live cleanup operations stack.</p>
          <p>{isDemoMode ? "This hosted view is using seeded artifacts from the San Francisco Bay pilot scenario." : "This view is connected to the live API-backed pilot stack."}</p>
        </footer>
      </div>
    </div>
  );
}

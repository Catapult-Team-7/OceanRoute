import { FormEvent, useEffect, useState } from "react";
import type { LatLngExpression } from "leaflet";
import { CircleMarker, MapContainer, Popup, Polyline, TileLayer } from "react-leaflet";
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

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";
const DEFAULT_CENTER: [number, number] = [37.8066, -122.4659];

const STRATEGY_LABELS: Record<RoutingBenchmarkStrategy["strategy"], string> = {
  nearest_hotspot: "Nearest hotspot",
  highest_yield: "Highest-yield sweep",
  recon_aware: "Recon-aware optimizer",
};

const PLATFORM_PILLARS = [
  {
    index: "01",
    title: "Lead with the mission story",
    copy:
      "The website now opens with the clearer public-interest story so partners, donors, and crews immediately understand why cleanup missions need better intelligence.",
  },
  {
    index: "02",
    title: "Keep the working operations stack",
    copy:
      "The experience stays wired to live forecast, route, feedback, and export flows so the site is more than a pitch deck. It is still the real operations surface.",
  },
  {
    index: "03",
    title: "Make room for the next model",
    copy:
      "A dedicated model integration studio keeps connector targets, benchmark comparison, and the next datasets visible in the same place the crews already work.",
  },
];

const OPERATION_LOOP = [
  {
    label: "Ingest and trust",
    detail: "Forecast freshness, fallback mode, and confidence stay visible so crews know when they are acting on a strong signal.",
  },
  {
    label: "Plan the mission",
    detail: "Route optimization and recon fallback remain one click away from the live hotspot map.",
  },
  {
    label: "Close the loop",
    detail: "Mission observations and cleanup logs feed the impact ledger so model tuning and sponsor reporting share the same source of truth.",
  },
];

const MODEL_CONNECTORS = [
  {
    name: "SOCAT",
    status: "connected",
    purpose: "Ship-based ocean carbon observations for labels and validation windows.",
  },
  {
    name: "NOAA GML CO2",
    status: "connected",
    purpose: "Atmospheric CO2 baseline features for residual or hybrid model variants.",
  },
  {
    name: "ERA5",
    status: "testing",
    purpose: "Wind and forcing context for routing, drift, and flux-aware model experiments.",
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

type DebrisFilter = DebrisClass | "all";

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

export function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [forecast, setForecast] = useState<ForecastSnapshot | null>(null);
  const [impact, setImpact] = useState<ImpactDashboard | null>(null);
  const [benchmark, setBenchmark] = useState<RoutingBenchmarkReport | null>(null);
  const [route, setRoute] = useState<RoutePlan | null>(null);
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
  const [feedbackMessage, setFeedbackMessage] = useState("");

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
    void (async () => {
      try {
        const status = await requestJson<HealthStatus>("/health");
        setHealth(status);
        setFilters((current) => ({ ...current, horizonHour: status.default_horizon_hours }));
        setRouteForm((current) => ({
          ...current,
          depotLat: status.default_depot_lat,
          depotLon: status.default_depot_lon,
        }));
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Failed to load system status.");
      } finally {
        setLoadingHealth(false);
      }
    })();
  }, []);

  useEffect(() => {
    void loadImpactDashboard();
    void loadBenchmarkReport();
  }, []);

  useEffect(() => {
    void loadForecastSnapshot(filters);
  }, [filters.horizonHour, filters.debrisClass, filters.minConfidence]);

  async function handleRunForecast() {
    setLoadingForecast(true);
    setError(null);
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
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to run forecast.");
      setLoadingForecast(false);
    }
  }

  async function handleOptimizeRoute() {
    setLoadingRoute(true);
    setError(null);
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
  const readinessCards = [
    {
      label: "Forecast engine",
      value: activeProvenance ? sourceLabel(activeProvenance) : "Waiting for first run",
      detail: activeProvenance
        ? `${formatFreshness(activeProvenance.age_minutes)} with ${confidenceBand(forecast)} confidence`
        : "Run or seed a forecast to populate the operations stack.",
    },
    {
      label: "Route intelligence",
      value: route ? route.recommended_mode : strategyLabel(winningStrategy),
      detail: route
        ? `${route.ordered_cell_ids.length || route.alternates.length} destination(s) staged for this mission window.`
        : benchmarkSummary(benchmark),
    },
    {
      label: "Impact loop",
      value: impact ? `${impact.total_missions} logged mission(s)` : "Ledger warming up",
      detail: impact
        ? `${formatPercent(impact.hotspot_precision)} hotspot precision with ${impact.total_collected_kg.toFixed(1)} kg recorded.`
        : "Mission feedback will unlock precision and hit-rate reporting.",
    },
    {
      label: "Model hook",
      value: "Integration-ready",
      detail: "Connector targets, benchmark comparison, and export endpoints now live on the same website.",
    },
  ];
  const summaryCards = [
    {
      label: "Pilot region",
      value: health?.pilot_region ?? "sf_bay_estuary",
      detail: loadingHealth ? "Checking backend status..." : "Single-region Bay operations mode.",
    },
    {
      label: "Top hotspot",
      value: topLine(forecast),
      detail: activeProvenance
        ? `${formatFreshness(activeProvenance.age_minutes)} | ${confidenceBand(forecast)} confidence`
        : "Run a forecast to populate hotspot ranking.",
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
            <a href="#operations">Operations</a>
            <a href="#model-lab">Model</a>
            <a href="#impact">Impact</a>
          </nav>
          <a className="button button-ghost" href="#impact">
            Ready for integration
          </a>
        </header>

        <main id="top">
          <section className="hero section">
            <div className="hero-copy-block reveal">
              <p className="eyebrow">Unified branch for routing, ops, and model integration</p>
              <h1>One OceanRoute website for the full cleanup workflow.</h1>
              <p className="lead">
                This build brings the mission story, the working forecast and routing desk, and the model-integration
                studio into one product surface so we can plug the next model into a single website instead of juggling
                separate prototypes.
              </p>
              <div className="hero-actions">
                <a className="button" href="#operations">
                  Explore operations
                </a>
                <button className="button button-secondary" onClick={handleRunForecast} type="button" disabled={loadingForecast}>
                  {loadingForecast ? "Refreshing forecast..." : "Run fresh forecast"}
                </button>
              </div>
              <div className="hero-notes">
                <span>{health?.ingest_mode ?? "auto"} ingest</span>
                <span>{health?.scheduler_enabled ? "scheduled forecast loop" : "manual forecast loop"}</span>
                <span>API + website merged on one branch</span>
              </div>
            </div>

            <div className="hero-panel-stack reveal">
              <article className="signal-card">
                <p className="signal-label">Integration snapshot</p>
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
                    <p className="route-kicker">From forecast to field log</p>
                    <h2>Unified mission loop</h2>
                  </div>
                  <span className="status-pill status-teal">{benchmark ? strategyLabel(benchmark.winning_strategy) : "Wiring"}</span>
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
                    {route ? `${route.recommended_mode} mission plan live` : benchmarkSummary(benchmark)}
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

          {error ? <div className="message-banner message-banner-error">{error}</div> : null}

          <section id="platform" className="section">
            <div className="section-heading reveal">
              <p className="eyebrow">Why this merge works</p>
              <h2>Three branches now read like one coherent platform instead of parallel prototypes.</h2>
              <p>
                The integrated website keeps the public-facing narrative understandable while preserving the operational
                dashboard and surfacing a model-integration studio for the next phase of the product.
              </p>
            </div>

            <div className="problem-grid">
              {PLATFORM_PILLARS.map((pillar) => (
                <article className="panel-card reveal" key={pillar.index}>
                  <span className="card-index">{pillar.index}</span>
                  <h3>{pillar.title}</h3>
                  <p>{pillar.copy}</p>
                </article>
              ))}
            </div>

            <div className="comparison-grid">
              {OPERATION_LOOP.map((step) => (
                <article className="comparison-card reveal" key={step.label}>
                  <h3>{step.label}</h3>
                  <p>{step.detail}</p>
                </article>
              ))}
            </div>
          </section>

          <section id="operations" className="section section-tint">
            <div className="section-heading reveal">
              <p className="eyebrow">Operations desk</p>
              <h2>Forecast hotspots, shape a route, and keep trust signals visible in the same flow.</h2>
              <p>
                The working Bay dashboard stays intact here, but it now lives inside a calmer website shell that is
                easier to present to partners while crews still get the controls they need.
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
                  <p className="empty-state">No forecast stored yet. Run the first forecast cycle to populate the map.</p>
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
                    <p>Optimize a single-vessel route or return a recon recommendation.</p>
                  </div>
                  <button
                    className="button button-secondary"
                    onClick={handleOptimizeRoute}
                    type="button"
                    disabled={loadingRoute || !forecast}
                  >
                    {loadingRoute ? "Optimizing..." : "Optimize route"}
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
                  <p className="empty-state">Optimize a route after loading a forecast.</p>
                )}
              </section>
            </div>
          </section>

          <section id="model-lab" className="section">
            <div className="section-heading reveal">
              <p className="eyebrow">Model integration studio</p>
              <h2>Bring the model roadmap forward without spinning up a second website.</h2>
              <p>
                This section keeps the product honest about what is already operational, what connectors we still need,
                and how the routing benchmark can guide the next model integration pass.
              </p>
            </div>

            <div className="model-grid">
              <section className="workspace-card">
                <div className="workspace-header">
                  <div>
                    <h3>Readiness rails</h3>
                    <p>The end-to-end loop we can integrate the next model into.</p>
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
                    <p>Compare baseline strategies before we attach a richer predictive model.</p>
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
            </div>
          </section>

          <section id="impact" className="section section-tint">
            <div className="section-heading reveal">
              <p className="eyebrow">Impact and feedback</p>
              <h2>Keep mission verification, sponsor reporting, and future model tuning in one ledger.</h2>
              <p>
                This is where operations becomes learning. Field crews log what they actually found, and the same data
                powers both impact reporting and the next round of model improvement.
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
                  <button className="button" type="submit" disabled={!route || submittingFeedback}>
                    {submittingFeedback ? "Saving feedback..." : "Save mission feedback"}
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
                      <a href={`${API_BASE}/export/pdf-brief`} target="_blank" rel="noreferrer">
                        PDF brief
                      </a>
                      <a href={`${API_BASE}/export/geojson?horizon_hour=${filters.horizonHour}`} target="_blank" rel="noreferrer">
                        GeoJSON
                      </a>
                    </p>
                  </div>
                </div>
              </section>
            </div>
          </section>
        </main>

        <footer className="site-footer">
          <p>Unified integration branch blending the mission story, the live operations desk, and the model roadmap.</p>
          <p>Built for one future model hook instead of three competing web surfaces.</p>
        </footer>
      </div>
    </div>
  );
}

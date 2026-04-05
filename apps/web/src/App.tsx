import { FormEvent, useEffect, useState } from "react";
import { CircleMarker, MapContainer, Popup, Polyline, TileLayer } from "react-leaflet";
import type { LatLngExpression } from "leaflet";
import type {
  DebrisClass,
  ForecastProvenance,
  ForecastSnapshot,
  ForecastStep,
  HealthStatus,
  ImpactDashboard,
  RecommendedMode,
  RegionInfo,
  RoutePlan,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";
const DEFAULT_CENTER: [number, number] = [37.8066, -122.4659];

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
    return "#e6a700";
  }
  return "#c44536";
}

function buildRoutePolyline(route: RoutePlan | null, forecast: ForecastSnapshot | null, routeForm: RouteFormState): LatLngExpression[] {
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

function baselineLabel(provenance: ForecastProvenance): string {
  if (!provenance.baseline_engine) {
    return "Baseline pending";
  }
  return provenance.baseline_engine === "pygnome" ? "Baseline PyGNOME" : "Baseline custom particle";
}

function modelLabel(provenance: ForecastProvenance): string {
  if (!provenance.model_id || !provenance.model_architecture) {
    return "Baseline-only forecast";
  }
  const scopeLabel = provenance.training_scope === "shared" ? "shared" : provenance.training_scope === "per_region" ? "regional" : null;
  const stageLabel = provenance.model_stage ? ` [${provenance.model_stage}]` : "";
  const overrideLabel = provenance.used_candidate_override ? " candidate override" : "";
  const fallbackLabel = provenance.used_inference_fallback ? " fallback used" : "";
  if (provenance.model_dataset_version) {
    return `Model ${provenance.model_architecture} ${provenance.model_dataset_version}${scopeLabel ? ` (${scopeLabel})` : ""}${stageLabel}${overrideLabel}${fallbackLabel}`;
  }
  return `Model ${provenance.model_architecture}${scopeLabel ? ` (${scopeLabel})` : ""}${stageLabel}${overrideLabel}${fallbackLabel}`;
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

export function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [regions, setRegions] = useState<RegionInfo[]>([]);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [forecast, setForecast] = useState<ForecastSnapshot | null>(null);
  const [impact, setImpact] = useState<ImpactDashboard | null>(null);
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
  const [loadingRoute, setLoadingRoute] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [forecastMissing, setForecastMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedbackMessage, setFeedbackMessage] = useState<string>("");
  const [actionMessage, setActionMessage] = useState<string>("");
  const [view, setView] = useState<"home" | "app" | "records">("home");
  type MissionRecord = {
    mission_id: string;
    date: string;
    collected_kg: number;
    distance_km: number;
    hours: number;
    mode: string;
    notes?: string | null;
  };
  const [recentMissions, setRecentMissions] = useState<MissionRecord[]>([]);
  const LOCAL_MISSIONS_KEY = "sea_sweep_missions_v1";
  const [showNewMissionForm, setShowNewMissionForm] = useState(false);
  const [newMission, setNewMission] = useState<Partial<MissionRecord>>({
    date: new Date().toISOString(),
    collected_kg: 0,
    distance_km: 0,
    hours: 1,
    mode: "collection",
    notes: "",
  });
  const [missionFilters, setMissionFilters] = useState<{
    mode: "all" | "collection" | "recon";
    minKg: number;
    from?: string | null;
    to?: string | null;
  }>({ mode: "all", minKg: 0, from: null, to: null });

  useEffect(() => {
    // load persisted missions
    try {
      const raw = localStorage.getItem(LOCAL_MISSIONS_KEY);
      if (raw) {
        setRecentMissions(JSON.parse(raw));
      }
    } catch (e) {
      // ignore parse errors
    }
  }, []);

  // If backend is available and not demo, try loading server-side missions
  useEffect(() => {
    if (!health || (health.ingest_mode ?? "live") === "demo") return;
    void (async () => {
      try {
        const serverMissions = await requestJson<MissionRecord[]>('/missions');
        if (Array.isArray(serverMissions) && serverMissions.length > 0) {
          setRecentMissions(serverMissions);
        }
      } catch (e) {
        // ignore backend fetch errors; local copy remains
      }
    })();
  }, [health]);

  useEffect(() => {
    try {
      localStorage.setItem(LOCAL_MISSIONS_KEY, JSON.stringify(recentMissions));
    } catch (e) {
      // ignore storage errors
    }
  }, [recentMissions]);

  function addMission(payload: Partial<MissionRecord>) {
    const mission: MissionRecord = {
      mission_id: payload.mission_id ?? `M-${Math.floor(Math.random() * 9000) + 1000}`,
      date: payload.date ?? new Date().toISOString(),
      collected_kg: Number(payload.collected_kg ?? 0),
      distance_km: Number(payload.distance_km ?? 0),
      hours: Number(payload.hours ?? 0.5),
      mode: payload.mode ?? "collection",
      notes: payload.notes ?? null,
    };
    // Optimistically add locally
    setRecentMissions((cur) => [mission, ...cur]);
    setShowNewMissionForm(false);
    setNewMission({ date: new Date().toISOString(), collected_kg: 0, distance_km: 0, hours: 1, mode: "collection", notes: "" });

    // Try to persist to backend when not in demo mode
    (async () => {
      try {
        if (health && (health.ingest_mode ?? "live") !== "demo") {
          await requestJson('/missions', {
            method: 'POST',
            body: JSON.stringify(mission),
          });
        }
      } catch (err) {
        // If backend persist fails, keep local copy and notify user
        setActionMessage('Saved mission locally (failed to persist to server)');
        setTimeout(() => setActionMessage(''), 4000);
      }
    })();
  }

  function handleNewMissionSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    addMission({
      mission_id: newMission.mission_id,
      date: newMission.date,
      collected_kg: Number(newMission.collected_kg ?? 0),
      distance_km: Number(newMission.distance_km ?? 0),
      hours: Number(newMission.hours ?? 0.5),
      mode: (newMission.mode as "collection" | "recon") ?? "collection",
      notes: newMission.notes ?? null,
    });
  }

  const filteredMissions = recentMissions.filter((m) => {
    if (missionFilters.mode !== "all" && m.mode !== missionFilters.mode) return false;
    if (missionFilters.minKg && m.collected_kg < missionFilters.minKg) return false;
    if (missionFilters.from) {
      const from = new Date(missionFilters.from);
      if (new Date(m.date) < from) return false;
    }
    if (missionFilters.to) {
      const to = new Date(missionFilters.to);
      if (new Date(m.date) > to) return false;
    }
    return true;
  });

  useEffect(() => {
    void (async () => {
      try {
        const status = await requestJson<HealthStatus>("/health");
        setHealth(status);
        setSelectedRegionId(status.pilot_region);
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
    void (async () => {
      try {
        const availableRegions = await requestJson<RegionInfo[]>("/regions");
        setRegions(availableRegions);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Failed to load regions.");
      }
    })();
  }, []);


  // Observe reveal-on-scroll elements whenever the visible view changes so newly-rendered
  // elements (like the hero on the Home view) get observed and animate when they enter.
  useEffect(() => {
    const els = Array.from(document.querySelectorAll<HTMLElement>(".reveal-on-scroll"));
    // Reset any existing reveal state so the animation can replay when entering view.
    els.forEach((el) => el.classList.remove("reveal"));

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("reveal");
          }
        });
      },
      { threshold: 0.12 },
    );

    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [view]);
  useEffect(() => {
    if (!selectedRegionId || regions.length === 0) {
      return;
    }
    const selectedRegion = regions.find((region) => region.id === selectedRegionId);
    if (!selectedRegion) {
      return;
    }
    setRoute(null);
    setRouteForm((current) => ({
      ...current,
      depotLat: selectedRegion.default_depot_lat,
      depotLon: selectedRegion.default_depot_lon,
    }));
  }, [selectedRegionId, regions]);

  useEffect(() => {
    void (async () => {
      try {
        const dashboard = await requestJson<ImpactDashboard>("/impact/dashboard");
        setImpact(dashboard);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Failed to load impact dashboard.");
      }
    })();
  }, []);

  useEffect(() => {
    void (async () => {
      if (!selectedRegionId) {
        return;
      }
      setLoadingForecast(true);
      setError(null);
      try {
        const params = new URLSearchParams();
        params.set("region_id", selectedRegionId);
        params.set("horizon_hour", String(filters.horizonHour));
        params.set("min_confidence", String(filters.minConfidence));
        if (filters.debrisClass !== "all") {
          params.set("class", filters.debrisClass);
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
    })();
  }, [filters.horizonHour, filters.debrisClass, filters.minConfidence, selectedRegionId]);

  async function handleRunForecast() {
    if (!selectedRegionId) {
      return;
    }
    setLoadingForecast(true);
    setError(null);
    setActionMessage("");
    try {
      await requestJson("/forecast/run", {
        method: "POST",
        body: JSON.stringify({
          region_id: selectedRegionId,
          horizon_hours: 72,
          debris_classes: ["low", "high"],
          source_mode: "auto",
        }),
      });
      const params = new URLSearchParams();
      params.set("region_id", selectedRegionId);
      params.set("horizon_hour", String(filters.horizonHour));
      params.set("min_confidence", String(filters.minConfidence));
      if (filters.debrisClass !== "all") {
        params.set("class", filters.debrisClass);
      }
      const snapshot = await requestJson<ForecastSnapshot>(`/forecast/latest?${params.toString()}`);
      setForecast(snapshot);
      setRoute(null);
      setForecastMissing(false);
      setActionMessage(
        `Forecast refreshed for ${snapshot.region.name} at ${formatTimestamp(snapshot.generated_at)} using ${modelLabel(
          snapshot.provenance,
        )}.`,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to run forecast.");
    } finally {
      setLoadingForecast(false);
    }
  }

  async function handleOptimizeRoute() {
    if (!selectedRegionId) {
      return;
    }
    setLoadingRoute(true);
    setError(null);
    setActionMessage("");
    try {
      const nextRoute = await requestJson<RoutePlan>("/route/optimize", {
        method: "POST",
        body: JSON.stringify({
          region_id: selectedRegionId,
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
      setActionMessage(
        nextRoute.recommended_mode === "collection"
          ? `Collection route created with ${nextRoute.ordered_cell_ids.length} stop(s).`
          : `Recon plan returned. No collection route cleared the objective threshold.`,
      );
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
      const dashboard = await requestJson<ImpactDashboard>("/impact/dashboard");
      setImpact(dashboard);
      setFeedbackMessage("Mission feedback saved to the impact ledger.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to submit mission feedback.");
    } finally {
      setSubmittingFeedback(false);
    }
  }

  function seedDemoMissions() {
    if (recentMissions.length > 0) return;
    const now = new Date();
    const baseKg = impact?.total_collected_kg ?? 42;
    const baseDist = impact?.total_distance_km ?? 12.3;
    const baseHours = Math.max(Math.round((impact?.total_hours ?? 3.2) * 10) / 10, 0.5);
    const sample: MissionRecord[] = [0, 1, 2].map((i) => ({
      mission_id: `M-${Math.floor(Math.random() * 9000) + 1000}`,
      date: new Date(now.getTime() - i * 86400000).toISOString(),
      collected_kg: Math.round((baseKg / 3) * (1 - i * 0.08) * 10) / 10,
      distance_km: Math.round((baseDist / 3) * (1 + i * 0.2) * 10) / 10,
      hours: Math.round((baseHours / 3) * (1 + i * 0.15) * 10) / 10,
      mode: i % 2 === 0 ? "collection" : "recon",
      notes: i === 0 ? "Sufficient debris found; crew collected samples." : i === 1 ? "Search only; no collection made." : "Light collection with spot checks.",
    }));
    setRecentMissions(sample);
  }

  const routePolyline = buildRoutePolyline(route, forecast, routeForm);
  const activeProvenance = forecast?.provenance ?? null;
  const routeProvenance = route?.forecast_provenance ?? null;
  const selectedRegion = regions.find((region) => region.id === selectedRegionId) ?? forecast?.region ?? null;
  const mapCenter: [number, number] = selectedRegion
    ? [selectedRegion.default_depot_lat, selectedRegion.default_depot_lon]
    : DEFAULT_CENTER;

  return (
    <div className="app-shell">
      <nav className="top-nav" role="navigation" aria-label="Main">
        <div className="nav-left">
          <p className="nav-brand">SeaSweep</p>
        </div>
        <div className="nav-right">
          <button
            type="button"
            className={`nav-button ${view === "home" ? "active" : ""}`}
            onClick={() => setView("home")}
            aria-pressed={view === "home"}
          >
            <span className="nav-icon" aria-hidden>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 11.5L12 4l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1v-8.5z" fill="currentColor" />
              </svg>
            </span>
            <span className="nav-label">Home</span>
          </button>

          <button
            type="button"
            className={`nav-button ${view === "app" ? "active" : ""}`}
            onClick={() => setView("app")}
            aria-pressed={view === "app"}
          >
            <span className="nav-icon" aria-hidden>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="3" width="8" height="8" rx="1" fill="currentColor" />
                <rect x="13" y="3" width="8" height="8" rx="1" fill="currentColor" />
                <rect x="3" y="13" width="8" height="8" rx="1" fill="currentColor" />
                <rect x="13" y="13" width="8" height="8" rx="1" fill="currentColor" />
              </svg>
            </span>
            <span className="nav-label">Dashboard</span>
          </button>
          <button
            type="button"
            className={`nav-button ${view === "records" ? "active" : ""}`}
            onClick={() => setView("records")}
            aria-pressed={view === "records"}
            title="Records & ledger"
          >
            <span className="nav-icon" aria-hidden>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 5h16v2H4V5zm0 4h10v2H4V9zm0 4h16v6H4v-6z" fill="currentColor" />
              </svg>
            </span>
            <span className="nav-label">Records</span>
          </button>
        </div>
      </nav>

      

      {error ? <div className="error-banner">{error}</div> : null}
      {actionMessage ? <div className="success-banner">{actionMessage}</div> : null}

      {view === "home" ? (
        <>
          <header className="hero reveal-on-scroll">
            <div>
              <p className="eyebrow">SeaSweep</p>
              <h1>Protecting our coasts by sweeping debris — operationally</h1>
              <p className="hero-copy">
                SeaSweep connects forecasts, routes, and field feedback into a single operator dashboard so teams can
                find, collect, and learn from floating debris more effectively. Fast forecasts, clear uncertainty, and
                mission-grade routes for real crews.
              </p>
            </div>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => setView("app")} type="button">
                Get started
              </button>
              <div className="hero-meta">
                <span className="status-pill">{health?.ingest_mode ?? "auto"} ingest</span>
                <span className="status-pill">{health?.scheduler_enabled ? "scheduled" : "manual"}</span>
              </div>
            </div>
          </header>

          <section className="objectives reveal-on-scroll">
            <div className="about-inner">
              <h2>Our objective</h2>
              <p>
                SeaSweep's mission is to make coastal cleanup faster and data-driven. We focus on delivering
                operational forecasts, clear uncertainty signals for decision making, and practical tools that turn
                forecasts into short, single-vessel missions that crews can execute today.
              </p>
              <div className="kpi-grid">
                <article>
                  <strong>Target reduction</strong>
                  <p>Reduce floating debris in pilot regions by <strong>30%</strong> per season (operational target)</p>
                </article>
                <article>
                  <strong>Response time</strong>
                  <p>Enable mission planning in under <strong>15 minutes</strong> from forecast to route</p>
                </article>
                <article>
                  <strong>Mission ROI</strong>
                  <p>Maximize kilograms collected per vessel-hour while limiting false-search distance</p>
                </article>
              </div>
              <h3>How it works</h3>
              <ol>
                <li>Run a forecast for your pilot region and horizon.</li>
                <li>Filter by confidence and debris class to focus operations.</li>
                <li>Optimize a single-vessel route and collect mission feedback to improve models.</li>
              </ol>
            </div>
            {/* New mission form relocated to Records page */}
          </section>

          <section className="about reveal-on-scroll">
            <div className="about-inner">
              <h2>About SeaSweep</h2>
              <p>
                SeaSweep is dedicated to reducing marine debris by making field operations smarter. We provide
                confidence-aware forecasts, mission planning tools, and a feedback loop that turns field observations
                into measurable impact.
              </p>
              <ul className="mission-list">
                <li>Fast, actionable forecasts for coastal operators</li>
                <li>Route planning tuned for single-vessel collection missions</li>
                <li>Operational feedback and impact accounting to improve models</li>
              </ul>
            </div>
          </section>
        </>
      ) : (
        <>
          <section className="summary-strip">
            <article className="summary-card">
              <span className="summary-label">Pilot region</span>
              <strong>{selectedRegion?.name ?? health?.pilot_region ?? "sf_bay_estuary"}</strong>
              <p>{loadingHealth ? "Checking backend status..." : "Active regional forecast workspace."}</p>
            </article>
            <article className="summary-card">
              <span className="summary-label">Top hotspot</span>
              <strong>{topLine(forecast)}</strong>
              <p>
                {activeProvenance
                  ? `${sourceLabel(activeProvenance)} | ${formatFreshness(activeProvenance.age_minutes)}`
                  : "Run a forecast to populate the ranking."}
              </p>
            </article>
            <article className="summary-card">
              <span className="summary-label">Kg per vessel-km</span>
              <strong>{impact ? impact.kg_per_vessel_km.toFixed(2) : "0.00"}</strong>
              <p>{impact ? `${impact.total_missions} logged mission(s).` : "Impact ledger waiting for field feedback."}</p>
            </article>
            <article className="summary-card">
              <span className="summary-label">Exports</span>
              <strong>PDF + GeoJSON</strong>
              <p className="export-links">
                <a href={`${API_BASE}/export/pdf-brief`} target="_blank" rel="noreferrer">
                  PDF brief
                </a>
                <a href={`${API_BASE}/export/geojson?horizon_hour=${filters.horizonHour}`} target="_blank" rel="noreferrer">
                  GeoJSON
                </a>
              </p>
            </article>
          </section>

          {/* Dashboard banner containing broadcast/run action and quick provenance info */}
          <div className="dashboard-banner">
            <div className="dashboard-banner-inner">
              <button
                className="primary-button"
                onClick={handleRunForecast}
                type="button"
                disabled={loadingForecast}
              >
                {loadingForecast ? "Broadcasting..." : "Broadcast forecast"}
              </button>
              <div className="banner-meta">
                {activeProvenance ? (
                  <span>Last run: {formatTimestamp(activeProvenance.generated_at)} — {modelLabel(activeProvenance)}</span>
                ) : (
                  <span>No stored forecast yet. Run a broadcast to populate data.</span>
                )}
              </div>
            </div>
          </div>

          <main className="dashboard-grid">
        <section className="panel panel-map">
          <div className="panel-header">
            <div>
              <h2>Operations map</h2>
              <p>Confidence-aware hotspots for the selected forecast horizon.</p>
            </div>
            <div className="filter-row">
              <label>
                Area
                <select
                  aria-label="Region selector"
                  value={selectedRegionId ?? ""}
                  onChange={(event) => setSelectedRegionId(event.target.value)}
                >
                  {regions.map((region) => (
                    <option key={region.id} value={region.id}>
                      {region.name}
                    </option>
                  ))}
                </select>
              </label>
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
                <div className="range-control">
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
                  <span className="range-value">{formatPercent(filters.minConfidence)}</span>
                </div>
              </label>
            </div>
          </div>
          {activeProvenance ? (
            <div className="trust-strip" aria-label="Forecast trust summary">
              <span className="status-pill trust-pill">{sourceLabel(activeProvenance)}</span>
              <span className="status-pill trust-pill">
                Generated {formatTimestamp(activeProvenance.generated_at)}
              </span>
              <span className="status-pill trust-pill">{baselineLabel(activeProvenance)}</span>
              <span className="status-pill trust-pill">{modelLabel(activeProvenance)}</span>
              <span className={`status-pill trust-pill ${activeProvenance.is_stale ? "trust-pill-stale" : ""}`}>
                {activeProvenance.is_stale ? "stale" : "fresh"} within {activeProvenance.stale_after_minutes}m
              </span>
              <span className="status-pill trust-pill">Confidence {confidenceBand(forecast)}</span>
            </div>
          ) : null}
          <div className="legend-row">
            <span className="legend-chip legend-high">High confidence</span>
            <span className="legend-chip legend-medium">Usable with caution</span>
            <span className="legend-chip legend-low">Recon only</span>
          </div>
          <div className="map-frame">
            <MapContainer key={`${selectedRegionId ?? "default"}-${forecast?.run_id ?? "no-forecast"}`} center={mapCenter} zoom={10} scrollWheelZoom className="leaflet-map">
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
              {routePolyline.length > 1 ? <Polyline pathOptions={{ color: "#0a5c8f", weight: 4 }} positions={routePolyline} /> : null}
            </MapContainer>
          </div>
          {forecastMissing ? <p className="empty-state">No forecast stored yet. Run the first forecast cycle to populate the map.</p> : null}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Hotspot ranking</h2>
              <p>Top cells after filtering by horizon and confidence.</p>
            </div>
            <span className="panel-badge">{forecast?.steps.length ?? 0} cells</span>
          </div>
          <div className="hotspot-list" aria-label="Hotspot ranking">
            {forecast?.top_hotspots.map((hotspot) => (
              <article className="hotspot-card" key={`${hotspot.cell_id}-${hotspot.debris_class}-${hotspot.horizon_hour}`}>
                <div>
                  <h3>{hotspot.cell_id.replaceAll("_", " ")}</h3>
                  <p>
                    {hotspot.debris_class} debris, h+{hotspot.horizon_hour}, confidence {formatPercent(hotspot.confidence)}
                  </p>
                </div>
                <strong>{formatRange(hotspot.expected_kg_min, hotspot.expected_kg_max)}</strong>
              </article>
            ))}
            {!forecast?.top_hotspots.length && !loadingForecast ? (
              <p className="empty-state">No hotspots match the current filters.</p>
            ) : null}
          </div>

          <div className="route-form">
            <div className="panel-header route-header">
              <div>
                <h2>Mission planner</h2>
                <p>Optimize a single-vessel route or return a recon recommendation.</p>
              </div>
              <button className="secondary-button" onClick={handleOptimizeRoute} type="button" disabled={loadingRoute || !forecast}>
                {loadingRoute ? "Optimizing..." : "Optimize route"}
              </button>
            </div>
            <div className="route-grid">
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
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Route summary</h2>
              <p>Collection route when the expected yield clears the threshold, recon otherwise.</p>
            </div>
            <span className={`route-mode route-mode-${route?.recommended_mode ?? "idle"}`}>
              {route ? route.recommended_mode : "idle"}
            </span>
          </div>
          {route ? (
            <>
              <div className="summary-metrics">
                <div>
                  <span>Yield range</span>
                  <strong>{formatRange(route.expected_kg_min, route.expected_kg_max)}</strong>
                </div>
                <div>
                  <span>Distance</span>
                  <strong>{route.expected_distance_km.toFixed(1)} km</strong>
                </div>
                <div>
                  <span>Fuel</span>
                  <strong>{route.estimated_fuel_liters.toFixed(1)} L</strong>
                </div>
                <div>
                  <span>Risk</span>
                  <strong>{formatPercent(1 - route.uncertainty_risk)}</strong>
                </div>
              </div>
              <div className="route-sequence">
                <h3>Forecast provenance</h3>
                {routeProvenance ? (
                  <div>
                    <p>
                      {sourceLabel(routeProvenance)} | generated {formatTimestamp(routeProvenance.generated_at)} | h+
                      {route.target_horizon_hour} | {route.forecast_is_stale ? "stale" : "fresh"} | confidence{" "}
                      {confidenceBand(forecast)}
                    </p>
                    <p>
                      {baselineLabel(routeProvenance)} | {modelLabel(routeProvenance)}
                    </p>
                  </div>
                ) : (
                  <p>Route was optimized from ad hoc candidates rather than a stored forecast run.</p>
                )}
              </div>
              <div className="route-sequence">
                <h3>Ordered cells</h3>
                <ol>
                  {route.ordered_cell_ids.map((cellId) => (
                    <li key={cellId}>{cellId}</li>
                  ))}
                </ol>
                {!route.ordered_cell_ids.length ? <p className="empty-state">Recon mode. Start with alternates instead of a full route.</p> : null}
              </div>
              <div className="route-sequence">
                <h3>Alternates</h3>
                <p>{route.alternates.join(", ") || "No alternates available."}</p>
              </div>
            </>
          ) : (
            <p className="empty-state">Optimize a route after loading a forecast.</p>
          )}
        </section>

        {/* Mission feedback and Impact ledger moved to Records page */}
      </main>
        </>
      )}
      {view === "records" ? (
        <main className="records-grid">
          <div className="records-intro reveal-on-scroll">
            <div className="about-inner">
              <h2>Records & Impact ledger</h2>
              <p>
                A single place to review logged missions, add crew feedback, and track operational impact. Entries
                here feed the model training loop and the impact dashboard.
              </p>
              <div className="records-stats">
                <div>
                  <strong>{impact?.total_missions ?? 0}</strong>
                  <span>Logged missions</span>
                </div>
                <div>
                  <strong>{impact ? impact.kg_per_vessel_km.toFixed(2) : "0.00"}</strong>
                  <span>Kg / vessel-km</span>
                </div>
                <div>
                  <strong>{impact ? impact.kg_per_hour.toFixed(2) : "0.00"}</strong>
                  <span>Kg / hour</span>
                </div>
              </div>
            </div>
          </div>
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>Mission feedback</h2>
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
                  onChange={(event) => setFeedback((current) => ({ ...current, estimatedKg: Number(event.target.value) }))}
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
                  onChange={(event) => setFeedback((current) => ({ ...current, collectedKg: Number(event.target.value) }))}
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
                  onChange={(event) => setFeedback((current) => ({ ...current, routeDeviationReason: event.target.value }))}
                />
              </label>
              <label>
                Crew notes
                <textarea aria-label="Crew notes" value={feedback.note} onChange={(event) => setFeedback((current) => ({ ...current, note: event.target.value }))} />
              </label>
              <button className="primary-button" type="submit" disabled={!route || submittingFeedback}>
                {submittingFeedback ? "Saving feedback..." : "Save mission feedback"}
              </button>
              {feedbackMessage ? <p className="success-banner">{feedbackMessage}</p> : null}
            </form>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>Impact ledger</h2>
                <p>Operational metrics over logged missions.</p>
              </div>
            </div>
            <div className="impact-grid">
              <article>
                <span>Kg per hour</span>
                <strong>{impact ? impact.kg_per_hour.toFixed(2) : "0.00"}</strong>
              </article>
              <article>
                <span>Hotspot precision</span>
                <strong>{impact ? formatPercent(impact.hotspot_precision) : "0%"}</strong>
              </article>
              <article>
                <span>Mission hit rate</span>
                <strong>{impact ? formatPercent(impact.mission_hit_rate) : "0%"}</strong>
              </article>
              <article>
                <span>False-search km</span>
                <strong>{impact ? impact.false_search_distance_km.toFixed(1) : "0.0"}</strong>
              </article>
            </div>
            {forecast?.source_notes.length ? (
              <div className="source-notes">
                {forecast.source_notes.map((note) => (
                  <p key={note}>{note}</p>
                ))}
              </div>
            ) : null}
          </section>
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>Logged missions</h2>
                <p>Recent missions and quick details — use this to review past activity.</p>
              </div>
              <div>
                {recentMissions.length === 0 ? (
                  <button className="secondary-button" type="button" onClick={seedDemoMissions}>
                    Load demo missions
                  </button>
                ) : null}
              </div>
            </div>
            <div className="missions-toolbar">
              <div className="filter-row">
                <label>
                  Mode
                  <select
                    value={missionFilters.mode}
                    onChange={(e) => setMissionFilters((c) => ({ ...c, mode: e.target.value as any }))}
                  >
                    <option value="all">All</option>
                    <option value="collection">Collection</option>
                    <option value="recon">Recon</option>
                  </select>
                </label>
                <label>
                  Min kg
                  <input
                    type="number"
                    min="0"
                    value={missionFilters.minKg}
                    onChange={(e) => setMissionFilters((c) => ({ ...c, minKg: Number(e.target.value) }))}
                  />
                </label>
                <label>
                  From
                  <input
                    type="date"
                    value={missionFilters.from ?? ""}
                    onChange={(e) => setMissionFilters((c) => ({ ...c, from: e.target.value || null }))}
                  />
                </label>
                <label>
                  To
                  <input
                    type="date"
                    value={missionFilters.to ?? ""}
                    onChange={(e) => setMissionFilters((c) => ({ ...c, to: e.target.value || null }))}
                  />
                </label>
                <div>
                  <button className="secondary-button" type="button" onClick={() => setMissionFilters({ mode: "all", minKg: 0, from: null, to: null })}>
                    Clear
                  </button>
                </div>
              </div>
              <div style={{ marginTop: 10 }}>
                {showNewMissionForm ? (
                  <button className="secondary-button" type="button" onClick={() => setShowNewMissionForm(false)}>
                    Cancel
                  </button>
                ) : (
                  <button className="primary-button" type="button" onClick={() => setShowNewMissionForm(true)}>
                    New mission
                  </button>
                )}
              </div>
            </div>
            <div className="missions-list">
              {filteredMissions.length === 0 ? (
                <p className="empty-state">No missions match filters. Load demo missions or add one.</p>
              ) : (
                filteredMissions.map((m) => (
                  <article className="mission-card" key={m.mission_id}>
                    <div className="mission-row">
                      <div>
                        <strong>{m.mission_id}</strong>
                        <div className="muted">{new Date(m.date).toLocaleString()}</div>
                      </div>
                      <div className="mission-stats">
                        <span>{m.collected_kg} kg</span>
                        <span>{m.distance_km} km</span>
                        <span>{m.hours}h</span>
                      </div>
                    </div>
                    <p className="mission-notes">{m.notes}</p>
                  </article>
                ))
              )}
            </div>
          </section>
        </main>
      ) : null}
    </div>
  );
}

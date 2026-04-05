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

  const routePolyline = buildRoutePolyline(route, forecast, routeForm);
  const activeProvenance = forecast?.provenance ?? null;
  const routeProvenance = route?.forecast_provenance ?? null;
  const selectedRegion = regions.find((region) => region.id === selectedRegionId) ?? forecast?.region ?? null;
  const mapCenter: [number, number] = selectedRegion
    ? [selectedRegion.default_depot_lat, selectedRegion.default_depot_lon]
    : DEFAULT_CENTER;

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">OceanRoute v1</p>
          <h1>Regional debris response desk</h1>
          <p className="hero-copy">
            Tactical cleanup support for the next shift across the active pilot regions. Forecast probable debris
            concentration zones, see uncertainty, and turn the best cells into a route or reconnaissance plan.
          </p>
        </div>
        <div className="hero-actions">
          <button className="primary-button" onClick={handleRunForecast} type="button" disabled={loadingForecast}>
            {loadingForecast ? "Refreshing forecast..." : "Run fresh forecast"}
          </button>
          <div className="hero-meta">
            <span className="status-pill">{health?.ingest_mode ?? "auto"} ingest</span>
            <span className="status-pill">{health?.scheduler_enabled ? "scheduled" : "manual"}</span>
          </div>
        </div>
      </header>

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

      {error ? <div className="error-banner">{error}</div> : null}
      {actionMessage ? <div className="success-banner">{actionMessage}</div> : null}

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
                onChange={(event) =>
                  setFeedback((current) => ({ ...current, photoUrl: event.target.value }))
                }
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
                onChange={(event) =>
                  setFeedback((current) => ({ ...current, note: event.target.value }))
                }
              />
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
      </main>
    </div>
  );
}

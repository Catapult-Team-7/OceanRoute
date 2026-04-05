import { MissionMapCanvas } from "./MissionMapCanvas";

import {
  baselineLabel,
  confidenceBand,
  formatFreshness,
  formatNumber,
  formatPercent,
  formatRange,
  formatTimestamp,
  modelLabel,
  routeReason,
  sourceLabel,
} from "../lib/mission-utils";
import type {
  ForecastFilters,
  ForecastSnapshot,
  RegionInfo,
  RouteFormState,
  RoutePlan,
} from "../types";

interface MissionPageProps {
  regions: RegionInfo[];
  selectedRegion: RegionInfo | null;
  selectedRegionId: string | null;
  onSelectRegion: (regionId: string) => void;
  filters: ForecastFilters;
  onUpdateFilters: (patch: Partial<ForecastFilters>) => void;
  forecast: ForecastSnapshot | null;
  forecastMissing: boolean;
  loadingForecast: boolean;
  onRunForecast: () => void;
  route: RoutePlan | null;
  loadingRoute: boolean;
  routeForm: RouteFormState;
  onUpdateRouteForm: (patch: Partial<RouteFormState>) => void;
  onOptimizeRoute: () => void;
}

export function MissionPage({
  regions,
  selectedRegion,
  selectedRegionId,
  onSelectRegion,
  filters,
  onUpdateFilters,
  forecast,
  forecastMissing,
  loadingForecast,
  onRunForecast,
  route,
  loadingRoute,
  routeForm,
  onUpdateRouteForm,
  onOptimizeRoute,
}: MissionPageProps) {
  const provenance = forecast?.provenance ?? null;
  const routeCells = route?.recommended_mode === "collection" ? route.ordered_cell_ids : route?.alternates ?? [];
  const plannerReason = routeReason(route);

  return (
    <section className="mission-layout">
      <div className="mission-controls panel-card">
        <div>
          <p className="section-kicker">Mission map</p>
          <h2>Operational forecast cockpit</h2>
          <p>Run fresh 72-hour forecasts, tune the hotspot view, and send the current ranking into a single-vessel route plan.</p>
        </div>
        <div className="control-grid">
          <label>
            Area
            <select value={selectedRegionId ?? ""} onChange={(event) => onSelectRegion(event.target.value)}>
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
              value={filters.horizonHour}
              onChange={(event) => onUpdateFilters({ horizonHour: Number(event.target.value) })}
            >
              <option value={24}>24h</option>
              <option value={48}>48h</option>
              <option value={72}>72h</option>
            </select>
          </label>
          <label>
            Debris class
            <select
              value={filters.debrisClass}
              onChange={(event) => onUpdateFilters({ debrisClass: event.target.value as ForecastFilters["debrisClass"] })}
            >
              <option value="all">All</option>
              <option value="low">Low-windage</option>
              <option value="high">High-windage</option>
            </select>
          </label>
          <label>
            Confidence floor
            <input
              type="range"
              min="0"
              max="0.8"
              step="0.05"
              value={filters.minConfidence}
              onChange={(event) => onUpdateFilters({ minConfidence: Number(event.target.value) })}
            />
            <span className="input-note">{formatPercent(filters.minConfidence)}</span>
          </label>
        </div>
        <div className="action-row">
          <button className="primary-button" type="button" onClick={onRunForecast} disabled={loadingForecast || !selectedRegionId}>
            {loadingForecast ? "Running forecast..." : `Run fresh forecast for ${selectedRegion?.name ?? "area"}`}
          </button>
          <button className="secondary-button" type="button" onClick={onOptimizeRoute} disabled={loadingRoute || !forecast}>
            {loadingRoute ? "Optimizing..." : "Optimize route"}
          </button>
        </div>
      </div>

      <div className="mission-grid">
        <article className="panel-card map-panel-card">
          <div className="panel-header">
            <div>
              <h3>Mission Map</h3>
              <p>Deck.GL route overlays, top cells, and trust context for the current region.</p>
            </div>
            {provenance ? (
              <div className="status-cluster">
                <span className="status-pill">{sourceLabel(provenance)}</span>
                <span className="status-pill">{confidenceBand(forecast)}</span>
                <span className="status-pill">{provenance.is_stale ? "stale" : "fresh"}</span>
              </div>
            ) : null}
          </div>
          {forecastMissing ? (
            <div className="empty-state">
              <strong>No forecast for {selectedRegion?.name ?? "this area"} yet.</strong>
              <p>Run a fresh forecast for this region to populate 24h, 48h, and 72h hotspot layers.</p>
              <button className="primary-button" type="button" onClick={onRunForecast}>
                Run fresh forecast for this area
              </button>
            </div>
          ) : (
            <>
              <MissionMapCanvas forecast={forecast} route={route} selectedRegion={selectedRegion} routeForm={routeForm} />
              <div className="map-caption">
                <span>Hotspots are sized by expected kilograms and colored by confidence.</span>
                <span>Collection routes draw solid cyan. Recon previews draw dashed amber.</span>
              </div>
            </>
          )}
        </article>

        <aside className="mission-sidebar">
          <article className="panel-card">
            <div className="panel-header">
              <div>
                <h3>Hotspot ranking</h3>
                <p>Top cells after filtering by horizon and confidence.</p>
              </div>
              <span className="panel-badge">{forecast?.top_hotspots.length ?? 0} cells</span>
            </div>
            {forecast?.top_hotspots.length ? (
              <div className="stack-list">
                {forecast.top_hotspots.slice(0, 10).map((step) => (
                  <article key={`${step.cell_id}-${step.debris_class}`} className="rank-card">
                    <div>
                      <strong>{step.cell_id.replaceAll("_", " ")}</strong>
                      <p>
                        {step.debris_class} debris, h+{step.horizon_hour}, confidence {formatPercent(step.confidence)}
                      </p>
                    </div>
                    <span>{formatRange(step.expected_kg_min, step.expected_kg_max)}</span>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-copy">No hotspots match the current filters.</p>
            )}
          </article>

          <article className="panel-card">
            <div className="panel-header">
              <div>
                <h3>Mission planner</h3>
                <p>Operator-tuned objective weights favor collection when debris is meaningful.</p>
              </div>
              <span className={`route-mode-chip ${route?.recommended_mode === "recon" ? "is-recon" : "is-collection"}`}>
                {route?.recommended_mode ?? "pending"}
              </span>
            </div>
            <div className="control-grid compact-grid">
              <label>
                Mission hours
                <input
                  type="number"
                  min="1"
                  max="12"
                  step="0.5"
                  value={routeForm.missionHours}
                  onChange={(event) => onUpdateRouteForm({ missionHours: Number(event.target.value) })}
                />
              </label>
              <label>
                Vessel speed (km/h)
                <input
                  type="number"
                  min="2"
                  max="60"
                  value={routeForm.vesselSpeedKmh}
                  onChange={(event) => onUpdateRouteForm({ vesselSpeedKmh: Number(event.target.value) })}
                />
              </label>
              <label>
                Fuel burn (L/h)
                <input
                  type="number"
                  min="1"
                  max="60"
                  value={routeForm.fuelBurnLph}
                  onChange={(event) => onUpdateRouteForm({ fuelBurnLph: Number(event.target.value) })}
                />
              </label>
            </div>
            {route ? (
              <div className="route-summary">
                <div className="mini-metric-grid">
                  <article>
                    <span>Yield range</span>
                    <strong>{formatRange(route.expected_kg_min, route.expected_kg_max)}</strong>
                  </article>
                  <article>
                    <span>Distance</span>
                    <strong>{formatNumber(route.expected_distance_km)} km</strong>
                  </article>
                  <article>
                    <span>Fuel</span>
                    <strong>{formatNumber(route.estimated_fuel_liters)} L</strong>
                  </article>
                  <article>
                    <span>Risk</span>
                    <strong>{formatPercent(route.uncertainty_risk)}</strong>
                  </article>
                </div>
                {plannerReason ? <p className="info-note">Planner note: {plannerReason}</p> : null}
                <div className="ordered-cells">
                  <span>{route.recommended_mode === "collection" ? "Ordered cells" : "Alternates"}</span>
                  <strong>{routeCells.length ? routeCells.join(", ") : "No route cells returned"}</strong>
                </div>
              </div>
            ) : (
              <p className="empty-copy">Run route optimization to get a collection circuit or recon preview.</p>
            )}
          </article>

          <article className="panel-card">
            <div className="panel-header">
              <div>
                <h3>Forecast provenance</h3>
                <p>Trust strip for source mode, baseline, and model involvement.</p>
              </div>
            </div>
            {provenance ? (
              <div className="stack-list">
                <article className="meta-card">
                  <span>Generated</span>
                  <strong>{formatTimestamp(provenance.generated_at)}</strong>
                  <p>{formatFreshness(provenance.age_minutes)}</p>
                </article>
                <article className="meta-card">
                  <span>Source mode</span>
                  <strong>{sourceLabel(provenance)}</strong>
                  <p>{provenance.source_notes.join(" ") || "No source notes recorded."}</p>
                </article>
                <article className="meta-card">
                  <span>Runtime</span>
                  <strong>{modelLabel(provenance)}</strong>
                  <p>{baselineLabel(provenance)}</p>
                </article>
                {provenance.model_fallback_reason ? (
                  <article className="meta-card">
                    <span>Fallback reason</span>
                    <strong>{provenance.model_fallback_reason}</strong>
                  </article>
                ) : null}
              </div>
            ) : (
              <p className="empty-copy">No forecast provenance yet. Fresh forecasts will populate source, model, and fallback details here.</p>
            )}
          </article>
        </aside>
      </div>
    </section>
  );
}

import type { FormEvent } from "react";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { formatFreshness, formatNumber, formatPercent, formatRange, formatTimestamp, modelLabel, sourceLabel } from "../lib/mission-utils";
import type { FeedbackState, ForecastSnapshot, ImpactDashboard, MissionRecord, RoutePlan } from "../types";

interface ProgressPageProps {
  impact: ImpactDashboard | null;
  missions: MissionRecord[];
  forecast: ForecastSnapshot | null;
  route: RoutePlan | null;
  feedback: FeedbackState;
  onChangeFeedback: (patch: Partial<FeedbackState>) => void;
  onSubmitFeedback: (event: FormEvent<HTMLFormElement>) => void;
  submittingFeedback: boolean;
}

export function ProgressPage({
  impact,
  missions,
  forecast,
  route,
  feedback,
  onChangeFeedback,
  onSubmitFeedback,
  submittingFeedback,
}: ProgressPageProps) {
  const chartData = missions
    .slice()
    .reverse()
    .map((mission) => ({
      label: new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(mission.date)),
      collectedKg: mission.collected_kg,
      distanceKm: mission.distance_km,
    }));

  return (
    <section className="progress-layout">
      <div className="mini-metric-grid impact-grid">
        <article className="panel-card">
          <span>Kg per hour</span>
          <strong>{impact ? formatNumber(impact.kg_per_hour, 2) : "0.00"}</strong>
        </article>
        <article className="panel-card">
          <span>Hotspot precision</span>
          <strong>{impact ? formatPercent(impact.hotspot_precision) : "0%"}</strong>
        </article>
        <article className="panel-card">
          <span>Mission hit rate</span>
          <strong>{impact ? formatPercent(impact.mission_hit_rate) : "0%"}</strong>
        </article>
        <article className="panel-card">
          <span>False-search km</span>
          <strong>{impact ? formatNumber(impact.false_search_distance_km) : "0.0"}</strong>
        </article>
      </div>

      <div className="progress-grid">
        <article className="panel-card">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Impact ledger</p>
              <h2>Mission progress</h2>
              <p>Logged missions, collection performance, and the current route/forecast trust state.</p>
            </div>
          </div>
          <div className="chart-frame">
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="kgFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3ab7ff" stopOpacity={0.58} />
                    <stop offset="95%" stopColor="#3ab7ff" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(33, 63, 94, 0.14)" />
                <XAxis dataKey="label" stroke="#557088" />
                <YAxis stroke="#557088" />
                <Tooltip />
                <Area type="monotone" dataKey="collectedKg" stroke="#0d74bd" fill="url(#kgFill)" strokeWidth={3} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="stack-list">
            {missions.length ? (
              missions.slice(0, 8).map((mission) => (
                <article key={mission.mission_id} className="rank-card">
                  <div>
                    <strong>{mission.mission_id}</strong>
                    <p>
                      {formatTimestamp(mission.date)} · {mission.mode}
                    </p>
                  </div>
                  <span>{formatNumber(mission.collected_kg)} kg</span>
                </article>
              ))
            ) : (
              <p className="empty-copy">No missions logged yet. Save mission feedback after a route run to populate the ledger.</p>
            )}
          </div>
        </article>

        <article className="panel-card">
          <div className="panel-header">
            <div>
              <h3>Current forecast and route</h3>
              <p>Trust and mission context for the active operator run.</p>
            </div>
          </div>
          {forecast ? (
            <div className="stack-list">
              <article className="meta-card">
                <span>Forecast</span>
                <strong>{forecast.region.name}</strong>
                <p>
                  {sourceLabel(forecast.provenance)} · {formatFreshness(forecast.provenance.age_minutes)}
                </p>
              </article>
              <article className="meta-card">
                <span>Runtime</span>
                <strong>{modelLabel(forecast.provenance)}</strong>
                <p>{forecast.provenance.model_fallback_reason ?? "No model fallback recorded."}</p>
              </article>
              <article className="meta-card">
                <span>Route</span>
                <strong>{route ? route.recommended_mode : "No route yet"}</strong>
                <p>{route ? formatRange(route.expected_kg_min, route.expected_kg_max) : "Optimize a route from Mission Map first."}</p>
              </article>
            </div>
          ) : (
            <p className="empty-copy">No active forecast yet. Run a forecast from Mission Map to populate the trust panel and route context.</p>
          )}
        </article>

        <article className="panel-card">
          <div className="panel-header">
            <div>
              <h3>Mission feedback</h3>
              <p>Keep the existing SeaSweep feedback loop in the demo branch instead of splitting it into a separate records app.</p>
            </div>
          </div>
          {route ? (
            <form className="stack-form" onSubmit={onSubmitFeedback}>
              <label>
                Outcome
                <select
                  value={feedback.foundStatus}
                  onChange={(event) => onChangeFeedback({ foundStatus: event.target.value as FeedbackState["foundStatus"] })}
                >
                  <option value="found">Found debris</option>
                  <option value="not_found">No debris found</option>
                </select>
              </label>
              <div className="control-grid compact-grid">
                <label>
                  Estimated kg
                  <input
                    type="number"
                    min="0"
                    step="0.5"
                    value={feedback.estimatedKg}
                    onChange={(event) => onChangeFeedback({ estimatedKg: Number(event.target.value) })}
                  />
                </label>
                <label>
                  Collected kg
                  <input
                    type="number"
                    min="0"
                    step="0.5"
                    value={feedback.collectedKg}
                    onChange={(event) => onChangeFeedback({ collectedKg: Number(event.target.value) })}
                  />
                </label>
              </div>
              <label>
                Crew notes
                <textarea value={feedback.note} onChange={(event) => onChangeFeedback({ note: event.target.value })} rows={4} />
              </label>
              <label>
                Route deviation reason
                <input
                  type="text"
                  value={feedback.routeDeviationReason}
                  onChange={(event) => onChangeFeedback({ routeDeviationReason: event.target.value })}
                />
              </label>
              <button className="primary-button" type="submit" disabled={submittingFeedback}>
                {submittingFeedback ? "Saving..." : "Save mission feedback"}
              </button>
            </form>
          ) : (
            <p className="empty-copy">No route available yet. Optimize a route first so the feedback form can attach to a mission and update the impact ledger.</p>
          )}
        </article>
      </div>
    </section>
  );
}

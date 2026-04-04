import { useMemo } from "react";

import { useOceanStore } from "../../store/oceanStore";
import { buildTopMissionTargets } from "../../utils/missionInsights";
import { formatCoords, formatFlux } from "../../utils/formatters";
import InfoHint from "../common/InfoHint";

export default function PriorityTargets() {
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const anomalies = useOceanStore((state) => state.anomalies);
  const verifiedMap = Boolean(heatmapData?.metadata?.verified_map);
  const points = heatmapData?.features || [];

  const targets = useMemo(() => buildTopMissionTargets(points, anomalies, 1.5, verifiedMap), [anomalies, points, verifiedMap]);

  const degradation = targets.degradationTarget;
  const recovery = targets.topRecoveryTarget;
  const route = recovery?.routeTarget || null;

  return (
    <section className="sidebar-section">
      <div className="section-header">
        <div>
          <h2>Priority Targets</h2>
          <small>Biggest mission items visible right now.</small>
        </div>
      </div>
      <div className="list-stack">
        <div className="stat-card">
          <div className="metric-label">
            <span>CO2 degradation</span>
            <InfoHint
              label="CO2 degradation"
              description="Highest weakening cell on the current map. This shows where observed or modeled sink strength is deteriorating the most."
            />
          </div>
          {verifiedMap && degradation ? (
            <>
              <strong>{formatCoords(degradation.geometry.coordinates[1], degradation.geometry.coordinates[0])}</strong>
              <small>
                Weakening {(degradation.properties.weakening_score || 0).toFixed(2)} · Flux{" "}
                {formatFlux(degradation.properties.predicted_flux || degradation.properties.flux)}
              </small>
            </>
          ) : (
            <small>Waiting for the checkpoint-backed map to refresh.</small>
          )}
        </div>
        <div className="stat-card">
          <div className="metric-label">
            <span>Recovery target</span>
            <InfoHint
              label="Recovery target"
              description="Cluster of high-priority recovery cells derived from the current verified route-priority and weakening surface. This is a routing proxy until real plastic observation feeds are connected."
            />
          </div>
          {verifiedMap && recovery ? (
            <>
              <strong>{formatCoords(recovery.lat, recovery.lon)}</strong>
              <small>
                {recovery.clusterSize} cells · Route priority {recovery.routePriority.toFixed(2)} · Weakening{" "}
                {recovery.weakening.toFixed(2)}
              </small>
            </>
          ) : (
            <small>Needs verified routing cells before recovery targets can be shown honestly.</small>
          )}
        </div>
        <div className="stat-card">
          <div className="metric-label">
            <span>Closest port route</span>
            <InfoHint
              label="Closest port route"
              description="Nearest major port to the strongest recovery target. These ports are real locations; the route is a straight-line mission cue, not a finished navigation engine."
            />
          </div>
          {verifiedMap && recovery && route ? (
            <>
              <strong>
                {route.name} · {formatCoords(route.lat, route.lon)}
              </strong>
              <small>
                Linked from {formatCoords(recovery.lat, recovery.lon)} · Distance score {route.score.toFixed(2)}
              </small>
            </>
          ) : (
            <small>Port routing appears after a verified recovery target exists.</small>
          )}
        </div>
      </div>
    </section>
  );
}

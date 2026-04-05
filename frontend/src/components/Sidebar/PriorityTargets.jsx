import { useMemo } from "react";

import { useOceanStore } from "../../store/oceanStore";
import { buildTopMissionTargets } from "../../utils/missionInsights";
import { formatCoords, formatFlux } from "../../utils/formatters";
import InfoHint from "../common/InfoHint";

export default function PriorityTargets() {
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const trashData = useOceanStore((state) => state.trashData);
  const anomalies = useOceanStore((state) => state.anomalies);
  const verifiedMap = Boolean(heatmapData?.metadata?.verified_map);
  const points = heatmapData?.features || [];

  const targets = useMemo(() => buildTopMissionTargets(points, anomalies, 1.5, verifiedMap), [anomalies, points, verifiedMap]);

  const degradation = targets.degradationTarget;
  const topTrash = [...(trashData?.hotspots || [])].sort((a, b) => (b.intensity || 0) - (a.intensity || 0))[0] || null;
  const recovery = topTrash;
  const route = recovery?.nearest_port || null;

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
            degradation ? (
              <>
                <strong>{formatCoords(degradation.geometry.coordinates[1], degradation.geometry.coordinates[0])}</strong>
                <small>
                  Provisional weakening {(degradation.properties.weakening_score || 0).toFixed(2)} · Flux{" "}
                  {formatFlux(degradation.properties.predicted_flux || degradation.properties.flux)}
                </small>
              </>
            ) : <small>Waiting for the map to produce a priority degradation target.</small>
          )}
        </div>
        <div className="stat-card">
          <div className="metric-label">
            <span>{topTrash?.observed ? "Trash hotspot" : "Predicted trash"}</span>
            <InfoHint
              label={topTrash?.observed ? "Trash hotspot" : "Predicted trash"}
              description={
                topTrash?.observed
                  ? "Highest-intensity measured trash/debris hotspot from the configured external feed."
                  : "Highest-intensity ML-predicted trash accumulation zone, optionally cross-checked against an observation feed."
              }
            />
          </div>
          {recovery ? (
            <>
              <strong>{formatCoords(recovery.lat, recovery.lon)}</strong>
              <small>{`${recovery.observed ? "Observed" : "Predicted"} intensity ${(recovery.intensity || 0).toFixed(2)}`}</small>
            </>
          ) : (
            <small>No trash hotspot is available yet.</small>
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
          {recovery && route ? (
            <>
              <strong>
                {route.name} · {formatCoords(route.lat, route.lon)}
              </strong>
              <small>
                Linked from {formatCoords(recovery.lat, recovery.lon)} · Distance score {route.score?.toFixed(2) || "n/a"}
              </small>
            </>
          ) : (
            <small>
              {recovery?.metadata?.transport_path?.length > 1
                ? "Following the ML transport path while a port destination is still unavailable."
                : "Port routing appears once a port link is available."}
            </small>
          )}
        </div>
      </div>
    </section>
  );
}

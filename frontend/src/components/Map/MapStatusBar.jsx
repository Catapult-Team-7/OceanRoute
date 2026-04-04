import { useMemo } from "react";

import { useOceanStore } from "../../store/oceanStore";

export default function MapStatusBar() {
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const anomalies = useOceanStore((state) => state.anomalies);
  const selectedRegion = useOceanStore((state) => state.selectedRegion);

  const summary = useMemo(() => {
    const points = heatmapData?.features || [];
    const sinkCells = points.filter((point) => point.properties.flux < 0).length;
    const maxWeakening = Math.max(0, ...points.map((point) => point.properties.weakening_score || 0));
    const topRoutePriority = Math.max(0, ...points.map((point) => point.properties.route_priority || 0));
    return {
      sinkCells,
      maxWeakening,
      topRoutePriority,
      modelReady: heatmapData?.metadata?.trained_model_ready,
      mode: heatmapData?.metadata?.inference_mode || "observed_demo_baseline",
      verifiedMap: heatmapData?.metadata?.verified_map || false,
      mapSource: heatmapData?.metadata?.map_source || "unavailable",
      anomalyCount: anomalies.length,
    };
  }, [anomalies.length, heatmapData]);

  return (
    <div className="map-status-bar">
      <div className="map-status-chip">
        <strong>{summary.verifiedMap ? "Verified map" : "Map status"}</strong>
        <span>{summary.mode}</span>
      </div>
      <div className="map-status-chip">
        <strong>Region</strong>
        <span>{selectedRegion}</span>
      </div>
      <div className="map-status-chip">
        <strong>Source</strong>
        <span>{summary.mapSource}</span>
      </div>
      <div className="map-status-chip">
        <strong>ML</strong>
        <span>{summary.modelReady ? "trained" : "not trained"}</span>
      </div>
      <div className="map-status-chip">
        <strong>Alerts</strong>
        <span>{summary.verifiedMap ? `${summary.anomalyCount} active anomalies` : "hidden until verified"}</span>
      </div>
      <div className="map-status-chip">
        <strong>Overlays</strong>
        <span>
          {summary.verifiedMap
            ? `${summary.sinkCells} sinks · ${summary.maxWeakening.toFixed(2)} weakening · ${summary.topRoutePriority.toFixed(2)} routing`
            : "suppressed until real gridded data is wired"}
        </span>
      </div>
    </div>
  );
}

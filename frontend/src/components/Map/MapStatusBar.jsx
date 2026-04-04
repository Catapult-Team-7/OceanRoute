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
      anomalyCount: anomalies.length,
    };
  }, [anomalies.length, heatmapData]);

  return (
    <div className="map-status-bar">
      <div className="map-status-chip">
        <strong>{summary.modelReady ? "ML mode" : "Demo mode"}</strong>
        <span>{summary.mode}</span>
      </div>
      <div className="map-status-chip">
        <strong>Region</strong>
        <span>{selectedRegion}</span>
      </div>
      <div className="map-status-chip">
        <strong>Sink cells</strong>
        <span>{summary.sinkCells}</span>
      </div>
      <div className="map-status-chip">
        <strong>Weakening</strong>
        <span>{summary.maxWeakening.toFixed(2)} max score</span>
      </div>
      <div className="map-status-chip">
        <strong>Alerts</strong>
        <span>{summary.anomalyCount} active anomalies</span>
      </div>
      <div className="map-status-chip">
        <strong>Routing</strong>
        <span>{summary.topRoutePriority.toFixed(2)} top route priority</span>
      </div>
    </div>
  );
}

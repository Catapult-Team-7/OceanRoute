import { useMemo } from "react";

import { useOceanStore } from "../../store/oceanStore";
import InfoHint from "../common/InfoHint";

export default function MapStatusBar() {
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const mlRuntimeStatus = useOceanStore((state) => state.mlRuntimeStatus);
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
      modelReady: heatmapData?.metadata?.trained_model_ready ?? mlRuntimeStatus?.model_ready ?? false,
      mode: heatmapData?.metadata?.inference_mode || "real_grid_pending",
      verifiedMap: heatmapData?.metadata?.verified_map || false,
      mapSource: heatmapData?.metadata?.map_source || "unavailable",
      anomalyCount: anomalies.length,
    };
  }, [anomalies.length, heatmapData, mlRuntimeStatus]);

  return (
    <div className="map-status-bar">
      <div className="map-status-chip">
        <strong className="metric-label">
          {summary.verifiedMap ? "Verified map" : "Map status"}
          <InfoHint
            label="Map status"
            description="Tells you whether the visible geospatial layer comes from a verified checkpoint-backed grid or from a provisional real-data state while the published map catches up."
          />
        </strong>
        <span>{summary.verifiedMap ? summary.mode : `provisional · ${summary.mode}`}</span>
      </div>
      <div className="map-status-chip">
        <strong className="metric-label">
          Region
          <InfoHint label="Region" description="Current ocean basin filter applied to the map and sidebar summaries." />
        </strong>
        <span>{selectedRegion}</span>
      </div>
      <div className="map-status-chip">
        <strong className="metric-label">
          Source
          <InfoHint
            label="Source"
            description="Backend source for the map layer. This should ultimately identify the live gridded data product powering the visible ocean overlays."
          />
        </strong>
        <span>{summary.mapSource}</span>
      </div>
      <div className="map-status-chip">
        <strong className="metric-label">
          ML
          <InfoHint
            label="ML"
            description="Shows whether the training pipeline has produced a usable model checkpoint. It is separate from map verification."
          />
        </strong>
        <span>{summary.modelReady ? "trained" : "not trained"}</span>
      </div>
      <div className="map-status-chip">
        <strong className="metric-label">
          Alerts
          <InfoHint
            label="Alerts"
            description="Count of anomaly regions where observed or predicted sink behavior deviates from baseline expectations enough to surface an alert."
          />
        </strong>
        <span>{`${summary.anomalyCount} ${summary.anomalyCount === 1 ? "alert" : "alerts"} visible`}</span>
      </div>
      <div className="map-status-chip">
        <strong className="metric-label">
          Overlays
          <InfoHint
            label="Overlays"
            description="Quick summary of visible sink cells, strongest weakening score, and top routing priority from the current map response."
          />
        </strong>
        <span>
          {`${summary.sinkCells} sinks · ${summary.maxWeakening.toFixed(2)} weakening · ${summary.topRoutePriority.toFixed(2)} routing`}
        </span>
      </div>
    </div>
  );
}

import { useOceanStore } from "../../store/oceanStore";

export default function DataDiagnostics() {
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const trashData = useOceanStore((state) => state.trashData);
  const anomalies = useOceanStore((state) => state.anomalies);
  const mlRuntimeStatus = useOceanStore((state) => state.mlRuntimeStatus);

  const featureCount = heatmapData?.features?.length || 0;
  const trashCount = trashData?.hotspots?.length || 0;
  const anomalyCount = anomalies?.length || 0;
  const source = heatmapData?.metadata?.map_source || "unavailable";
  const summary = heatmapData?.metadata?.source_summary || "No map summary returned yet.";
  const inferenceMode = heatmapData?.metadata?.inference_mode || "unknown";
  const modelReady = heatmapData?.metadata?.trained_model_ready ?? mlRuntimeStatus?.model_ready ?? false;

  return (
    <section className="sidebar-section">
      <h2>Diagnostics</h2>
      <div className="list-stack">
        <article className="stat-card">
          <span className="metric-label">Heatmap cells</span>
          <strong>{featureCount}</strong>
          <small>{source}</small>
        </article>
        <article className="stat-card">
          <span className="metric-label">Trash hotspots</span>
          <strong>{trashCount}</strong>
          <small>{trashData?.source || "unavailable"}</small>
        </article>
        <article className="stat-card">
          <span className="metric-label">Anomalies</span>
          <strong>{anomalyCount}</strong>
          <small>{inferenceMode}</small>
        </article>
        <article className="stat-card">
          <span className="metric-label">Model ready</span>
          <strong>{modelReady ? "yes" : "no"}</strong>
          <small>{summary}</small>
        </article>
      </div>
    </section>
  );
}

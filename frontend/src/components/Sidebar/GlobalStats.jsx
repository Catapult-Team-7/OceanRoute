import { useOceanStore } from "../../store/oceanStore";
import { formatFlux } from "../../utils/formatters";

export default function GlobalStats() {
  const globalStats = useOceanStore((state) => state.globalStats);
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const verifiedMap = Boolean(heatmapData?.metadata?.verified_map);

  const cards = [
    {
      label: "Mean Flux",
      value: verifiedMap ? formatFlux(globalStats.meanFlux) : "Unavailable",
      sub: verifiedMap ? (globalStats.meanFlux < 0 ? "Net sink state" : "Net source state") : "Waiting for verified gridded data",
    },
    {
      label: "Sink Coverage",
      value: verifiedMap && globalStats.sinkCoverage != null ? `${globalStats.sinkCoverage.toFixed(1)}%` : "Unavailable",
      sub: verifiedMap ? "Ocean cells with negative flux" : "Suppressed until the map is source-backed",
    },
    {
      label: "Model Status",
      value: heatmapData?.metadata?.trained_model_ready ? "Trained" : "Baseline",
      sub: heatmapData?.metadata?.verified_map
        ? "Can be compared against verified map inputs"
        : "Training exists, but the spatial map is not verified yet",
    },
  ];

  return (
    <section className="sidebar-section">
      <h2>Global Statistics</h2>
      <div className="stats-grid">
        {cards.map((card) => (
          <article key={card.label} className="stat-card">
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <small>{card.sub}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

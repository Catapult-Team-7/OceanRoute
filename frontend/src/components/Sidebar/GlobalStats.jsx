import { useOceanStore } from "../../store/oceanStore";
import { formatFlux } from "../../utils/formatters";
import InfoHint from "../common/InfoHint";

export default function GlobalStats() {
  const globalStats = useOceanStore((state) => state.globalStats);
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const verifiedMap = Boolean(heatmapData?.metadata?.verified_map);

  const cards = [
    {
      label: "Mean Flux",
      info: "Average air-sea CO2 flux across all map cells. Negative values mean net absorption by the ocean; positive values mean net release.",
      value: verifiedMap ? formatFlux(globalStats.meanFlux) : "Unavailable",
      sub: verifiedMap ? (globalStats.meanFlux < 0 ? "Net sink state" : "Net source state") : "Waiting for verified gridded data",
    },
    {
      label: "Sink Coverage",
      info: "Share of visible ocean cells where predicted flux is below zero. This is calculated as negative-flux cells divided by total mapped cells.",
      value: verifiedMap && globalStats.sinkCoverage != null ? `${globalStats.sinkCoverage.toFixed(1)}%` : "Unavailable",
      sub: verifiedMap ? "Ocean cells with negative flux" : "Suppressed until the map is source-backed",
    },
    {
      label: "Model Status",
      info: "Shows whether the ML training pipeline has produced a checkpoint. It does not by itself guarantee the public map is verified.",
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
            <span className="metric-label">
              {card.label}
              <InfoHint label={card.label} description={card.info} />
            </span>
            <strong>{card.value}</strong>
            <small>{card.sub}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

import { useOceanStore } from "../../store/oceanStore";
import { formatFlux } from "../../utils/formatters";

export default function GlobalStats() {
  const globalStats = useOceanStore((state) => state.globalStats);

  const cards = [
    {
      label: "Mean Flux",
      value: formatFlux(globalStats.meanFlux),
      sub: globalStats.meanFlux < 0 ? "Net sink state" : "Net source state",
    },
    {
      label: "Sink Coverage",
      value: globalStats.sinkCoverage == null ? "—" : `${globalStats.sinkCoverage.toFixed(1)}%`,
      sub: "Ocean cells with negative flux",
    },
    {
      label: "Demo Metric",
      value: "10.2 Gt CO2",
      sub: "Monitored equivalent this year",
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

import AlertBadge from "../common/AlertBadge";
import { useOceanStore } from "../../store/oceanStore";
import { formatCoords } from "../../utils/formatters";

export default function AnomalyList() {
  const anomalies = useOceanStore((state) => state.anomalies);
  const setSelectedPoint = useOceanStore((state) => state.setSelectedPoint);
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const verifiedMap = Boolean(heatmapData?.metadata?.verified_map);

  if (!verifiedMap) {
    return (
      <section className="sidebar-section">
        <div className="section-header">
          <h2>Anomaly Alerts</h2>
          <span>0</span>
        </div>
        <p className="empty-state">Anomaly alerts stay hidden until the map is backed by verified gridded data.</p>
      </section>
    );
  }

  return (
    <section className="sidebar-section">
      <div className="section-header">
        <h2>Anomaly Alerts</h2>
        <span>{anomalies.length}</span>
      </div>
      <div className="list-stack">
        {anomalies.map((anomaly) => (
          <button
            key={anomaly.id}
            type="button"
            className="anomaly-card"
            onClick={() => setSelectedPoint({ lat: anomaly.lat, lon: anomaly.lon })}
          >
            <div className="section-header">
              <strong>{anomaly.region_name}</strong>
              <AlertBadge severity={anomaly.severity} />
            </div>
            <span>{formatCoords(anomaly.lat, anomaly.lon)}</span>
            <small>{anomaly.deviation_pct}% deviation from baseline</small>
          </button>
        ))}
      </div>
    </section>
  );
}

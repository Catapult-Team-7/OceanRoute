import AlertBadge from "../common/AlertBadge";
import InfoHint from "../common/InfoHint";
import { useOceanStore } from "../../store/oceanStore";
import { formatCoords } from "../../utils/formatters";

export default function AnomalyList() {
  const anomalies = useOceanStore((state) => state.anomalies);
  const setSelectedPoint = useOceanStore((state) => state.setSelectedPoint);
  const heatmapData = useOceanStore((state) => state.heatmapData);
  const verifiedMap = Boolean(heatmapData?.metadata?.verified_map);

  return (
    <section className="sidebar-section">
      <div className="section-header">
        <h2 className="metric-label">
          Anomaly Alerts
          <InfoHint
            label="Anomaly Alerts"
            description="Flags regions where sink behavior is weakening or deviating from expected baseline conditions enough to warrant attention."
          />
        </h2>
        <span>{anomalies.length}</span>
      </div>
      {!verifiedMap ? (
        <p className="subtle">These alerts are provisional until the verified gridded map is published.</p>
      ) : null}
      <div className="list-stack">
        {anomalies.length ? anomalies.map((anomaly) => (
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
            <small className="metric-label">
              {anomaly.deviation_pct}% deviation from baseline
              <InfoHint
                label="Deviation from baseline"
                description="Difference between the current anomaly region and its expected baseline signal, expressed as a percentage."
              />
            </small>
          </button>
        )) : <p className="empty-state">No anomaly alerts are active for the current map month.</p>}
      </div>
    </section>
  );
}

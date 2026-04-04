import { Suspense, lazy } from "react";

import { useForecast } from "../../hooks/useForecast";
import { useOceanStore } from "../../store/oceanStore";
import { formatCoords, formatFlux } from "../../utils/formatters";

const FluxChart = lazy(() => import("../Timeline/FluxChart"));

export default function PointInspector() {
  useForecast();

  const selectedPoint = useOceanStore((state) => state.selectedPoint);
  const forecast = useOceanStore((state) => state.forecast);
  const history = useOceanStore((state) => state.history);

  if (!selectedPoint) {
    return (
      <section className="sidebar-section">
        <h2>Point Inspector</h2>
        <p className="empty-state">Click the map or an anomaly card to inspect a point forecast.</p>
      </section>
    );
  }

  return (
    <section className="sidebar-section">
      <h2>Point Inspector</h2>
      <p className="subtle">{formatCoords(selectedPoint.lat, selectedPoint.lon)}</p>
      <div className="inspector-metric">
        <strong>{formatFlux(forecast?.current_flux)}</strong>
        <small>{forecast?.current_flux < 0 ? "Absorbing CO2" : "Releasing CO2"}</small>
      </div>
      <div className="chart-block">
        <span>12-month context</span>
        <Suspense fallback={<div className="chart-skeleton" />}>
          <FluxChart data={history} />
        </Suspense>
      </div>
      <div className="forecast-list">
        {(forecast?.forecast || []).map((item) => (
          <div key={item.hours_ahead} className="forecast-row">
            <span>+{item.hours_ahead}h</span>
            <strong>{item.flux.toFixed(2)}</strong>
            <small>
              {item.confidence_low.toFixed(2)} to {item.confidence_high.toFixed(2)}
            </small>
          </div>
        ))}
      </div>
    </section>
  );
}

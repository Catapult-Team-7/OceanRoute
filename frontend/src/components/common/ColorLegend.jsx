import { FLUX_SCALE_LABELS } from "../../utils/colorScale";

export default function ColorLegend() {
  return (
    <div className="legend">
      <div className="legend-title">Carbon Sink Intensity</div>
      <div className="legend-gradient" />
      <div className="legend-labels">
        {FLUX_SCALE_LABELS.map((item) => (
          <span key={item.label}>{item.label}</span>
        ))}
      </div>
    </div>
  );
}

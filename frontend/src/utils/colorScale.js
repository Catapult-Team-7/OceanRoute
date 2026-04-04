export function fluxToColor(flux) {
  const normalized = Math.max(0, Math.min(1, (flux + 4) / 8));
  if (normalized < 0.25) return [16, 73, 186 + Math.round(normalized * 120)];
  if (normalized < 0.5) return [20 + Math.round(normalized * 140), 180, 230];
  if (normalized < 0.75) return [224, 196 - Math.round(normalized * 80), 70];
  return [226, 78 - Math.round((normalized - 0.75) * 180), 52];
}

export const FLUX_SCALE_LABELS = [
  { value: -4, label: "Strong sink", color: "#0B3D91" },
  { value: -2, label: "Sink", color: "#1597BB" },
  { value: 0, label: "Neutral", color: "#F3CA40" },
  { value: 2, label: "Source", color: "#F08A24" },
  { value: 4, label: "Strong source", color: "#C53030" },
];

export function formatFlux(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(2)} mol/m²/yr`;
}

export function formatCoords(lat, lon) {
  if (lat == null || lon == null) return "—";
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(2)}°${ns}, ${Math.abs(lon).toFixed(2)}°${ew}`;
}

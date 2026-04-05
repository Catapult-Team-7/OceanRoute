const MAJOR_PORTS = [
  { name: "Los Angeles", lat: 33.7405, lon: -118.273 },
  { name: "San Francisco", lat: 37.7749, lon: -122.4194 },
  { name: "Vancouver", lat: 49.2827, lon: -123.1207 },
  { name: "Honolulu", lat: 21.3069, lon: -157.8583 },
  { name: "Yokohama", lat: 35.4437, lon: 139.638 },
  { name: "Singapore", lat: 1.2903, lon: 103.8519 },
  { name: "Cape Town", lat: -33.9249, lon: 18.4241 },
  { name: "Santos", lat: -23.9608, lon: -46.3336 },
  { name: "Rotterdam", lat: 51.9244, lon: 4.4777 },
  { name: "Durban", lat: -29.8587, lon: 31.0218 },
];

function normalizeLon(lon) {
  return ((lon + 180) % 360) - 180;
}

function distanceScore(latA, lonA, latB, lonB) {
  const lonDelta = normalizeLon(lonA - lonB);
  return Math.hypot((latA - latB) / 8, lonDelta / 10);
}

export function nearestPort(lat, lon) {
  return MAJOR_PORTS.reduce((best, port) => {
    const score = distanceScore(lat, lon, port.lat, port.lon);
    if (!best || score < best.score) {
      return { ...port, score };
    }
    return best;
  }, null);
}

function clusterWindowForZoom(zoom) {
  if (zoom < 1.4) return 14;
  if (zoom < 2.1) return 8;
  if (zoom < 3) return 5;
  return 3;
}

function coordinateOf(feature) {
  const [lon, lat] = feature.geometry.coordinates;
  return { lat, lon };
}

export function buildRecoveryTargets(points, zoom = 1.5, verifiedMap = false) {
  const rankedPoints = [...points]
    .filter((point) => point?.properties)
    .sort(
      (a, b) =>
        ((b.properties.route_priority || 0) * 1.2 +
          (b.properties.weakening_score || 0) +
          (b.properties.anomaly_score || 0) * 0.9) -
        ((a.properties.route_priority || 0) * 1.2 +
          (a.properties.weakening_score || 0) +
          (a.properties.anomaly_score || 0) * 0.9)
    );
  const candidateCells = rankedPoints.filter(
    (point, index) =>
      index < 120 ||
      (point.properties.route_priority || 0) >= (verifiedMap ? 0.35 : 0.15) ||
      (point.properties.weakening_score || 0) >= (verifiedMap ? 0.18 : 0.08) ||
      (point.properties.anomaly_score || 0) >= (verifiedMap ? 0.25 : 0.12)
  );
  if (!candidateCells.length) return [];

  const clusterStep = clusterWindowForZoom(zoom);
  const groups = new Map();
  for (const point of candidateCells) {
    const { lat, lon } = coordinateOf(point);
    const weight =
      (point.properties.route_priority || 0) +
      (point.properties.weakening_score || 0) +
      Math.max(0.1, point.properties.anomaly_score || 0);
    const latKey = Math.round(lat / clusterStep) * clusterStep;
    const lonKey = Math.round(normalizeLon(lon) / clusterStep) * clusterStep;
    const key = `${latKey}:${lonKey}`;
    const current = groups.get(key) || {
      points: [],
      weightSum: 0,
      latSum: 0,
      lonSum: 0,
      maxWeakening: 0,
      maxRoutePriority: 0,
      maxAnomalyScore: 0,
      meanFluxAccumulator: 0,
    };
    current.points.push(point);
    current.weightSum += weight;
    current.latSum += lat * weight;
    current.lonSum += lon * weight;
    current.maxWeakening = Math.max(current.maxWeakening, point.properties.weakening_score || 0);
    current.maxRoutePriority = Math.max(current.maxRoutePriority, point.properties.route_priority || 0);
    current.maxAnomalyScore = Math.max(current.maxAnomalyScore, point.properties.anomaly_score || 0);
    current.meanFluxAccumulator += point.properties.predicted_flux || point.properties.flux || 0;
    groups.set(key, current);
  }

  return Array.from(groups.values())
    .map((group, index) => {
      const lat = group.latSum / Math.max(group.weightSum, 0.001);
      const lon = normalizeLon(group.lonSum / Math.max(group.weightSum, 0.001));
      const port = nearestPort(lat, lon);
      const clusterSize = group.points.length;
      return {
        id: `recovery-target-${index}`,
        lat,
        lon,
        clusterSize,
        label: clusterSize > 1 ? `${verifiedMap ? "Recovery" : "Provisional recovery"} cluster (${clusterSize})` : verifiedMap ? "Recovery target" : "Provisional recovery target",
        routeTarget: port,
        weakening: group.maxWeakening,
        routePriority: group.maxRoutePriority,
        anomalyScore: group.maxAnomalyScore,
        meanFlux: group.meanFluxAccumulator / Math.max(clusterSize, 1),
        provisional: !verifiedMap,
      };
    })
    .sort((a, b) => b.routePriority + b.weakening - (a.routePriority + a.weakening))
    .slice(0, 18);
}

export function buildDisplayTrashTargets(observedHotspots = [], fallbackTargets = [], zoom = 1.5) {
  const sourceHotspots = observedHotspots.length ? observedHotspots : fallbackTargets;
  if (!sourceHotspots.length) {
    return [];
  }

  const clusterStep = clusterWindowForZoom(zoom);
  const groups = new Map();
  for (const item of sourceHotspots) {
    const latKey = Math.round(item.lat / clusterStep) * clusterStep;
    const lonKey = Math.round(normalizeLon(item.lon) / clusterStep) * clusterStep;
    const key = `${latKey}:${lonKey}`;
    const weight = item.intensity || 1;
    const current = groups.get(key) || {
      items: [],
      weightSum: 0,
      latSum: 0,
      lonSum: 0,
      intensity: 0,
      nearestPort: null,
      routeTarget: null,
      routePriority: 0,
      weakening: 0,
      anomalyScore: 0,
    };
    current.items.push(item);
    current.weightSum += weight;
    current.latSum += item.lat * weight;
    current.lonSum += item.lon * weight;
    current.intensity = Math.max(current.intensity, item.intensity || item.routePriority || 0);
    current.nearestPort = current.nearestPort || item.nearest_port || null;
    current.routeTarget = current.routeTarget || item.routeTarget || null;
    current.routePriority = Math.max(current.routePriority, item.routePriority || item.intensity || 0);
    current.weakening = Math.max(current.weakening, item.weakening || 0);
    current.anomalyScore = Math.max(current.anomalyScore, item.anomalyScore || 0);
    groups.set(key, current);
  }

  return Array.from(groups.values())
    .map((group, index) => ({
      id: `trash-cluster-${index}`,
      label:
        group.items.length > 1
          ? `${group.items.some((item) => item.observed) ? "Trash" : "Predicted trash"} cluster (${group.items.length})`
          : group.items[0].label,
      lat: group.latSum / Math.max(group.weightSum, 0.001),
      lon: normalizeLon(group.lonSum / Math.max(group.weightSum, 0.001)),
      clusterSize: group.items.length,
      intensity: group.intensity,
      routePriority: group.routePriority,
      weakening: group.weakening,
      anomalyScore: group.anomalyScore,
      routeTarget: group.routeTarget || (group.nearestPort
        ? {
            name: group.nearestPort.name,
            lat: group.nearestPort.lat,
            lon: group.nearestPort.lon,
            country: group.nearestPort.country,
          }
        : null),
      observed: group.items.some((item) => item.observed),
      source: group.items[0].source,
      metadata: group.items[0].metadata || {},
    }))
    .sort((a, b) => (b.routePriority || b.intensity || 0) - (a.routePriority || a.intensity || 0));
}

export function buildTopMissionTargets(points, anomalies, zoom = 1.5, verifiedMap = false) {
  const degradationTarget = [...points]
    .filter((point) => typeof point?.properties?.weakening_score === "number")
    .sort(
      (a, b) =>
        (b.properties.weakening_score || 0) +
          (b.properties.anomaly_score || 0) -
        ((a.properties.weakening_score || 0) + (a.properties.anomaly_score || 0))
    )[0];

  const recoveryTargets = buildRecoveryTargets(points, zoom, verifiedMap);
  const topRecoveryTarget = recoveryTargets[0] || null;
  const topAnomaly = [...anomalies].sort((a, b) => (b.anomaly_score || 0) - (a.anomaly_score || 0))[0] || null;

  return {
    degradationTarget,
    topRecoveryTarget,
    topAnomaly,
    recoveryTargets,
  };
}

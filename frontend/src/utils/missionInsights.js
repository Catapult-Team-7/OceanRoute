function normalizeLon(lon) {
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

function distanceScore(latA, lonA, latB, lonB) {
  const lonDelta = normalizeLon(lonA - lonB);
  return Math.hypot((latA - latB) / 8, lonDelta / 10);
}

function distanceDeg(latA, lonA, latB, lonB) {
  const lonDelta = normalizeLon(lonA - lonB);
  return Math.hypot(latA - latB, lonDelta);
}

function clusterDistanceForZoom(zoom) {
  if (zoom < 1.4) return 10;
  if (zoom < 2.1) return 6;
  if (zoom < 3) return 3.5;
  return 2;
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
      const clusterSize = group.points.length;
      return {
        id: `recovery-target-${index}`,
        lat,
        lon,
        clusterSize,
        label: clusterSize > 1 ? `${verifiedMap ? "Recovery" : "Provisional recovery"} cluster (${clusterSize})` : verifiedMap ? "Recovery target" : "Provisional recovery target",
        routeTarget: null,
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

  if (observedHotspots.length) {
    const clusterDistance = clusterDistanceForZoom(zoom);
    const clusters = [];
    for (const item of observedHotspots
      .filter((item) => Math.abs(item.lat) <= 62)
      .sort(
        (a, b) =>
          (b.metadata?.source_count || 1) * (b.intensity || 0) -
          (a.metadata?.source_count || 1) * (a.intensity || 0)
      )) {
      const weight = Math.max(0.15, (item.metadata?.source_count || 1) * (item.intensity || 1));
      const existing = clusters.find(
        (cluster) => distanceDeg(item.lat, item.lon, cluster.lat, cluster.lon) <= clusterDistance
      );
      if (existing) {
        const previousWeight = existing.weight;
        const nextWeight = previousWeight + weight;
        existing.lat = (existing.lat * previousWeight + item.lat * weight) / nextWeight;
        existing.lon = normalizeLon((existing.lon * previousWeight + item.lon * weight) / nextWeight);
        existing.weight = nextWeight;
        existing.clusterSize += item.metadata?.source_count || 1;
        existing.intensity = Math.max(existing.intensity, item.intensity || 0);
        existing.routePriority = Math.max(existing.routePriority, item.metadata?.route_priority || item.intensity || 0);
        existing.weakening = Math.max(existing.weakening, item.metadata?.weakening_score || item.weakening || 0);
        existing.anomalyScore = Math.max(existing.anomalyScore, item.metadata?.anomaly_score || item.anomalyScore || 0);
        if (!existing.routeTarget && item.nearest_port) {
          existing.routeTarget = {
            name: item.nearest_port.name,
            lat: item.nearest_port.lat,
            lon: item.nearest_port.lon,
            country: item.nearest_port.country,
          };
        }
      } else {
        clusters.push({
          id: item.id || `trash-hotspot-${clusters.length}`,
          label: item.label || "Predicted trash convergence zone",
          lat: item.lat,
          lon: normalizeLon(item.lon),
          clusterSize: item.metadata?.source_count || 1,
          intensity: item.intensity || 0,
          routePriority: item.metadata?.route_priority || item.intensity || 0,
          weakening: item.metadata?.weakening_score || item.weakening || 0,
          anomalyScore: item.metadata?.anomaly_score || item.anomalyScore || 0,
          routeTarget: item.nearest_port
            ? {
                name: item.nearest_port.name,
                lat: item.nearest_port.lat,
                lon: item.nearest_port.lon,
                country: item.nearest_port.country,
              }
            : item.routeTarget || null,
          observed: Boolean(item.observed),
          source: item.source,
          metadata: item.metadata || {},
          weight,
        });
      }
    }

    return clusters
      .map((cluster, index) => ({
        id: cluster.id || `trash-hotspot-${index}`,
        label: cluster.clusterSize > 1 ? `Trash cluster (${cluster.clusterSize})` : cluster.label,
        lat: cluster.lat,
        lon: normalizeLon(cluster.lon),
        clusterSize: cluster.clusterSize,
        intensity: cluster.intensity,
        routePriority: cluster.routePriority,
        weakening: cluster.weakening,
        anomalyScore: cluster.anomalyScore,
        routeTarget: cluster.routeTarget,
        observed: cluster.observed,
        source: cluster.source,
        metadata: cluster.metadata,
      }))
      .sort((a, b) => (b.routePriority || b.intensity || 0) - (a.routePriority || a.intensity || 0));
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

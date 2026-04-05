import { Suspense, lazy, useMemo, useState } from "react";

import { GeoJsonLayer, PathLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";

import { useHeatmapData } from "../../hooks/useHeatmapData";
import { useTrashData } from "../../hooks/useTrashData";
import { useOceanStore } from "../../store/oceanStore";
import { buildDisplayTrashTargets, buildRecoveryTargets, buildTopMissionTargets } from "../../utils/missionInsights";

const MapLibreSurface = lazy(() => import("./MapLibreSurface"));
const INITIAL_VIEW_STATE = { longitude: 0, latitude: 12, zoom: 1.15, pitch: 0, bearing: 0 };

function normalizeLon(lon) {
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

function distanceDeg(pointA, pointB) {
  const lonDelta = normalizeLon(pointA.lon - pointB.lon);
  return Math.hypot(pointA.lat - pointB.lat, lonDelta);
}

function selectSpacedFeatures(points, minDistanceDeg, limit, predicate = null) {
  const selected = [];
  for (const point of points) {
    if (predicate && !predicate(point)) continue;
    const candidate = {
      point,
      lat: point.geometry.coordinates[1],
      lon: point.geometry.coordinates[0],
    };
    if (selected.every((item) => distanceDeg(candidate, item) >= minDistanceDeg)) {
      selected.push(candidate);
    }
    if (selected.length >= limit) {
      break;
    }
  }
  return selected.map((item) => item.point);
}

function hotspotStrength(point) {
  const predicted = point.properties?.predicted_flux || 0;
  const observed = point.properties?.observed_flux ?? predicted;
  const displaySignal = point.properties?.display_signal || 0;
  const weakening = point.properties?.weakening_score || 0;
  const anomaly = point.properties?.anomaly_score || 0;
  return Math.abs(predicted - observed) * 1.6 + displaySignal * 2.1 + weakening * 3.0 + anomaly * 1.5;
}

function co2HotspotScore(point) {
  const predicted = point.properties?.predicted_flux || 0;
  const observed = point.properties?.observed_flux ?? predicted;
  const displayFlux = point.properties?.display_flux ?? predicted;
  const displaySignal = point.properties?.display_signal || 0;
  const weakening = point.properties?.weakening_score || 0;
  const anomaly = point.properties?.anomaly_score || 0;
  const routePriority = point.properties?.route_priority || 0;
  return (
    Math.abs(predicted - observed) * 0.9 +
    Math.abs(displayFlux) * 0.9 +
    displaySignal * 1.6 +
    weakening * 3.6 +
    anomaly * 2.2 +
    routePriority * 1.1
  );
}

function routeStrength(target) {
  return (
    (target.routePriority || target.metadata?.route_priority || 0) * 1.3 +
    (target.intensity || 0) * 0.7 +
    (target.weakening || target.metadata?.weakening_score || 0) * 0.5
  );
}

function mapSignalScore(point) {
  const properties = point?.properties || {};
  const predicted = properties.predicted_flux || 0;
  const observed = properties.observed_flux ?? predicted;
  return (
    (properties.display_signal || 0) * 2.1 +
    (properties.weakening_score || 0) * 4.2 +
    (properties.anomaly_score || 0) * 3.4 +
    (properties.route_priority || 0) * 1.5 +
    Math.abs((properties.display_flux ?? predicted)) * 0.8 +
    Math.abs(predicted - observed) * 0.85
  );
}

function pointDistanceToSeed(point, seed) {
  return distanceDeg(
    { lat: point.geometry.coordinates[1], lon: point.geometry.coordinates[0] },
    seed
  );
}

function nearestSupportCoordinate(supportFeatures, lon, lat, maxDistanceDeg = 8) {
  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const feature of supportFeatures) {
    const [candidateLon, candidateLat] = feature.geometry.coordinates;
    const candidate = { lon: candidateLon, lat: candidateLat };
    const distance = distanceDeg(candidate, { lon, lat });
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  }
  return bestDistance <= maxDistanceDeg ? best : null;
}

function splitRouteAcrossDateline(path) {
  if (!Array.isArray(path) || path.length < 2) {
    return [];
  }
  const segments = [];
  let current = [path[0]];
  for (let index = 1; index < path.length; index += 1) {
    const previous = path[index - 1];
    const next = path[index];
    const lonDelta = Math.abs(previous[0] - next[0]);
    if (lonDelta > 180) {
      if (current.length > 1) {
        segments.push(current);
      }
      current = [next];
      continue;
    }
    current.push(next);
  }
  if (current.length > 1) {
    segments.push(current);
  }
  return segments;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function severityBand(score) {
  if (score >= 3.2) return "critical";
  if (score >= 2.2) return "high";
  if (score >= 1.4) return "elevated";
  return "watch";
}

function severityWeight(score) {
  return clamp(score / 3.6, 0.18, 1);
}

function formatArea(areaKm2) {
  if (areaKm2 >= 1_000_000) return `${(areaKm2 / 1_000_000).toFixed(2)}M km²`;
  if (areaKm2 >= 10_000) return `${Math.round(areaKm2 / 1000)}k km²`;
  return `${Math.round(areaKm2)} km²`;
}

function vectorOrientationDegrees(eastward = 0, northward = 0) {
  if (!eastward && !northward) {
    return 0;
  }
  return (Math.atan2(northward, eastward) * 180) / Math.PI;
}

function constrainLonToCenter(lon, centerLon) {
  let adjusted = lon;
  while (adjusted - centerLon > 180) adjusted -= 360;
  while (adjusted - centerLon < -180) adjusted += 360;
  if (centerLon >= 160 && adjusted < 0) {
    adjusted = 180;
  } else if (centerLon <= -160 && adjusted > 0) {
    adjusted = -180;
  }
  return clamp(adjusted, -180, 180);
}

function blobPolygon(
  lon,
  lat,
  radiusKm,
  {
    orientationDeg = 0,
    elongation = 0.28,
    roughness = 0.16,
    seed = 0,
    steps = 56,
  } = {}
) {
  const safeLat = clamp(lat, -84, 84);
  const latScale = 110.574;
  const lonScale = Math.max(111.32 * Math.cos((safeLat * Math.PI) / 180), 18);
  const coordinates = [];
  const orientation = (orientationDeg * Math.PI) / 180;
  for (let index = 0; index <= steps; index += 1) {
    const angle = (index / steps) * Math.PI * 2;
    const directionalStretch = 1 + elongation * Math.cos(angle - orientation);
    const wobble =
      1 +
      roughness * 0.45 * Math.sin(angle * 2 + seed) +
      roughness * 0.28 * Math.cos(angle * 3 - seed * 0.7) +
      roughness * 0.16 * Math.sin(angle * 5 + seed * 1.7);
    const localRadius = Math.max(radiusKm * 0.45, radiusKm * directionalStretch * wobble);
    const ringLat = clamp(safeLat + (Math.sin(angle) * localRadius) / latScale, -84, 84);
    const ringLon = constrainLonToCenter(lon + (Math.cos(angle) * localRadius) / lonScale, lon);
    coordinates.push([ringLon, ringLat]);
  }
  return {
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [coordinates],
    },
  };
}

function oceanBlobPolygon(lon, lat, radiusKm, supportFeatures, options = {}) {
  const sectorCount = options.sectorCount || 24;
  const safeLat = clamp(lat, -84, 84);
  const latScale = 110.574;
  const lonScale = Math.max(111.32 * Math.cos((safeLat * Math.PI) / 180), 18);
  const maxLatDelta = (radiusKm * 1.15) / latScale;
  const maxLonDelta = (radiusKm * 1.15) / lonScale;
  const bins = Array.from({ length: sectorCount }, () => []);
  const fallback = blobPolygon(lon, lat, radiusKm, options);

  for (const feature of supportFeatures) {
    const [candidateLon, candidateLat] = feature.geometry.coordinates;
    const lonDelta = normalizeLon(candidateLon - lon);
    const latDelta = candidateLat - lat;
    if (Math.abs(latDelta) > maxLatDelta || Math.abs(lonDelta) > maxLonDelta) {
      continue;
    }
    const dxKm = lonDelta * lonScale;
    const dyKm = latDelta * latScale;
    const distanceKm = Math.hypot(dxKm, dyKm);
    if (distanceKm > radiusKm * 1.2 || distanceKm < radiusKm * 0.14) {
      continue;
    }
    const angle = (Math.atan2(dyKm, dxKm) + Math.PI * 2) % (Math.PI * 2);
    const binIndex = Math.min(sectorCount - 1, Math.floor((angle / (Math.PI * 2)) * sectorCount));
    bins[binIndex].push({ angle, distanceKm });
  }

  const supportedBins = bins.filter((bin) => bin.length);
  if (supportedBins.length < Math.floor(sectorCount * 0.45)) {
    return fallback;
  }

  const smoothedDistances = bins.map((bin, index) => {
    const own = bin.length ? Math.max(...bin.map((item) => item.distanceKm)) : null;
    const previous = bins[(index - 1 + sectorCount) % sectorCount];
    const next = bins[(index + 1) % sectorCount];
    const prevDistance = previous.length ? Math.max(...previous.map((item) => item.distanceKm)) : null;
    const nextDistance = next.length ? Math.max(...next.map((item) => item.distanceKm)) : null;
    const samples = [prevDistance, own, nextDistance].filter((value) => value != null);
    if (!samples.length) {
      return radiusKm * 0.58;
    }
    return samples.reduce((sum, value) => sum + value, 0) / samples.length;
  });

  const coordinates = [];
  for (let index = 0; index <= sectorCount; index += 1) {
    const angle = ((index % sectorCount) / sectorCount) * Math.PI * 2;
    const localRadius = clamp(smoothedDistances[index % sectorCount], radiusKm * 0.46, radiusKm * 0.96);
    const ringLat = clamp(safeLat + (Math.sin(angle) * localRadius) / latScale, -84, 84);
    const ringLon = constrainLonToCenter(lon + (Math.cos(angle) * localRadius) / lonScale, lon);
    coordinates.push([ringLon, ringLat]);
  }
  return {
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [coordinates],
    },
  };
}

function zoneColor(kind, score) {
  const level = severityBand(score);
  if (kind === "trash_zone") {
    if (level === "critical") return [255, 108, 39];
    if (level === "high") return [255, 152, 61];
    if (level === "elevated") return [255, 198, 79];
    return [255, 227, 133];
  }
  if (level === "critical") return [255, 82, 123];
  if (level === "high") return [255, 112, 156];
  if (level === "elevated") return [255, 150, 186];
  return [255, 192, 216];
}

export default function OceanMap() {
  useHeatmapData();
  useTrashData();

  const heatmapData = useOceanStore((state) => state.heatmapData);
  const trashData = useOceanStore((state) => state.trashData);
  const anomalies = useOceanStore((state) => state.anomalies);
  const selectedRegion = useOceanStore((state) => state.selectedRegion);
  const basemapStyle = useOceanStore((state) => state.basemapStyle);
  const setSelectedPoint = useOceanStore((state) => state.setSelectedPoint);
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [showBrief, setShowBrief] = useState(true);
  const verifiedMap = Boolean(heatmapData?.metadata?.verified_map);
  const sourceSummary = heatmapData?.metadata?.source_summary || "";
  const gridResolution = selectedRegion === "global" ? "2deg" : "1deg";

  const points = heatmapData?.features || [];
  const recoveryTargets = useMemo(
    () => buildRecoveryTargets(points, viewState.zoom, verifiedMap),
    [points, verifiedMap, viewState.zoom]
  );
  const trashTargets = useMemo(
    () => buildDisplayTrashTargets(trashData?.hotspots || [], recoveryTargets, viewState.zoom),
    [trashData?.hotspots, recoveryTargets, viewState.zoom]
  );
  const visibleAnomalies = useMemo(
    () =>
      [...anomalies]
        .filter((anomaly) => Math.abs(anomaly.lat) <= 55)
        .sort((a, b) => (b.anomaly_score || 0) - (a.anomaly_score || 0))
        .slice(0, selectedRegion === "global" ? 8 : 12),
    [anomalies, selectedRegion]
  );
  const visibleTrashTargets = useMemo(
    () =>
      [...trashTargets]
        .sort((a, b) => routeStrength(b) - routeStrength(a))
        .slice(0, selectedRegion === "global" ? 30 : 36),
    [selectedRegion, trashTargets]
  );
  const anchoredTrashTargets = useMemo(
    () =>
      visibleTrashTargets.map((target) => {
        const anchored = nearestSupportCoordinate(points, target.lon, target.lat, selectedRegion === "global" ? 8 : 5);
        return anchored
          ? {
              ...target,
              lon: anchored.lon,
              lat: anchored.lat,
            }
          : target;
      }),
    [points, selectedRegion, visibleTrashTargets]
  );
  const rawRouteTargets = useMemo(
    () =>
      [...(trashData?.hotspots || [])]
        .filter((target) => target.nearest_port)
        .sort(
          (a, b) =>
            ((b.metadata?.route_priority || 0) + (b.intensity || 0)) -
            ((a.metadata?.route_priority || 0) + (a.intensity || 0))
        )
        .slice(0, selectedRegion === "global" ? 22 : 28),
    [selectedRegion, trashData?.hotspots]
  );
  const prioritySummary = useMemo(
    () => buildTopMissionTargets(points, visibleAnomalies, viewState.zoom, verifiedMap),
    [points, verifiedMap, viewState.zoom, visibleAnomalies]
  );

  const routedTrashTargets = useMemo(
    () => {
      const seenTransportOrigins = new Set();
      const selected = [];
      for (const item of [...rawRouteTargets]
        .filter((target) => target.routeTarget || target.nearest_port)
        .sort((a, b) => routeStrength(b) - routeStrength(a))) {
        if (!item.routeTarget) {
          const path = item.metadata?.transport_path || [];
          const end = path[path.length - 1] || [item.lon, item.lat];
          const transportKey = `${Math.round(item.lat)}:${Math.round(item.lon)}:${Math.round(end[1])}:${Math.round(end[0])}`;
          if (seenTransportOrigins.has(transportKey)) {
            continue;
          }
          seenTransportOrigins.add(transportKey);
        }
        const anchored = nearestSupportCoordinate(points, item.lon, item.lat, selectedRegion === "global" ? 8 : 5);
        selected.push(
          anchored
            ? {
                ...item,
                lon: anchored.lon,
                lat: anchored.lat,
              }
            : item
        );
        if (selected.length >= (selectedRegion === "global" ? 18 : 20)) {
          break;
        }
      }
      return selected;
    },
    [points, rawRouteTargets, selectedRegion]
  );

  const routeSegments = useMemo(
    () =>
      routedTrashTargets
        .flatMap((item) => {
          const transportPath =
            Array.isArray(item.metadata?.transport_path) && item.metadata.transport_path.length > 1
              ? item.metadata.transport_path
              : null;
          const routeTarget = item.routeTarget || item.nearest_port || null;
          if (!routeTarget) {
            return [];
          }
          const path = routeTarget
            ? [
                ...(transportPath || [[item.lon, item.lat]]),
                [routeTarget.lon, routeTarget.lat],
              ]
            : transportPath || [[item.lon, item.lat]];
          const targetPosition = transportPath
            ? transportPath[transportPath.length - 1]
            : routeTarget
              ? [routeTarget.lon, routeTarget.lat]
              : [item.lon, item.lat];
          return splitRouteAcrossDateline(path).map((segmentPath, segmentIndex) => ({
            id: `${item.id}-route-${segmentIndex}`,
            path: segmentPath,
            source: [item.lon, item.lat],
            target: targetPosition,
            density: item.routePriority || item.intensity || 0,
            label: routeTarget
              ? `${item.label} to ${routeTarget.name}`
              : `${item.label} current transport`,
            weakening: item.weakening || item.metadata?.weakening_score || 0,
            routeMode: "port",
          }));
        })
        .filter((item) => item.path.length > 1),
    [routedTrashTargets]
  );

  const hotspotNodes = useMemo(
    () => {
      const ranked = [...points]
        .filter((point) => point?.properties)
        .filter((point) => Math.abs(point.geometry.coordinates[1]) <= 62)
        .filter(
          (point) =>
            (point.properties.display_signal || 0) >= 0.38 ||
            (point.properties.weakening_score || 0) >= 0.14 ||
            (point.properties.anomaly_score || 0) >= 0.12 ||
            (point.properties.route_priority || 0) >= 0.38
        )
        .sort((a, b) => co2HotspotScore(b) - co2HotspotScore(a));
      return selectSpacedFeatures(
        ranked,
        selectedRegion === "global" ? 8 : 5,
        selectedRegion === "global" ? 28 : 34,
        (point) =>
          co2HotspotScore(point) >= (selectedRegion === "global" ? 1.0 : 0.8)
      );
    },
    [points, selectedRegion]
  );

  const weakeningZones = useMemo(
    () => {
      const ranked = [...points]
        .filter((point) => Math.abs(point.geometry.coordinates[1]) <= 60)
        .filter((point) => (point.properties.weakening_score || 0) > 0.1)
        .sort((a, b) => (b.properties.weakening_score || 0) - (a.properties.weakening_score || 0));
      return selectSpacedFeatures(
        ranked,
        selectedRegion === "global" ? 9 : 5,
        selectedRegion === "global" ? 26 : 28
      );
    },
    [points, selectedRegion]
  );

  const routeLabels = useMemo(
    () =>
      routedTrashTargets.slice(0, viewState.zoom > 2.2 ? 8 : 4).map((item) => ({
        id: `${item.id}-label`,
        position: [item.lon, item.lat],
        label: item.label,
      })),
    [routedTrashTargets, viewState.zoom]
  );

  const fieldHighlights = useMemo(() => {
    const seedLocations = [
      ...hotspotNodes.map((point) => ({
        lat: point.geometry.coordinates[1],
        lon: point.geometry.coordinates[0],
      })),
      ...weakeningZones.slice(0, 8).map((point) => ({
        lat: point.geometry.coordinates[1],
        lon: point.geometry.coordinates[0],
      })),
      ...visibleAnomalies.map((anomaly) => ({ lat: anomaly.lat, lon: anomaly.lon })),
      ...anchoredTrashTargets.map((target) => ({ lat: target.lat, lon: target.lon })),
    ];

    const strongThreshold = selectedRegion === "global" ? 1.65 : 1.05;
    const contextualThreshold = selectedRegion === "global" ? 0.72 : 0.5;
    const contextualDistance = selectedRegion === "global" ? 8.5 : 5;

    return [...points]
      .filter((point) => Math.abs(point.geometry.coordinates[1]) <= 60)
      .filter((point) => {
        const score = mapSignalScore(point);
        if (score >= strongThreshold) {
          return true;
        }
        return (
          score >= contextualThreshold &&
          seedLocations.some((seed) => pointDistanceToSeed(point, seed) <= contextualDistance)
        );
      })
      .sort((a, b) => mapSignalScore(b) - mapSignalScore(a))
      .slice(0, selectedRegion === "global" ? 900 : 1200);
  }, [anchoredTrashTargets, hotspotNodes, points, selectedRegion, visibleAnomalies, weakeningZones]);

  const co2Zones = useMemo(
    () =>
      hotspotNodes.map((point, index) => {
        const properties = point.properties || {};
        const lon = point.geometry.coordinates[0];
        const lat = point.geometry.coordinates[1];
        const score = co2HotspotScore(point);
        const radiusKm =
          (selectedRegion === "global" ? 180 : 120) +
          (properties.display_signal || 0) * 150 +
          (properties.weakening_score || 0) * 240 +
          (properties.anomaly_score || 0) * 120;
        const areaKm2 = Math.PI * radiusKm * radiusKm;
        const color = zoneColor("co2_zone", score);
        const orientationDeg = vectorOrientationDegrees(properties.current_u || 0, properties.current_v || 0);
        const seed = ((lon + 180) * 0.013) + ((lat + 90) * 0.021);
        return {
          ...oceanBlobPolygon(lon, lat, radiusKm, points, {
            orientationDeg,
            elongation: 0.16 + Math.min(0.14, Math.abs(properties.current_u || 0) + Math.abs(properties.current_v || 0)),
            roughness: 0.05 + Math.min(0.05, (properties.display_signal || 0) * 0.04),
            seed,
            sectorCount: 28,
          }),
          id: `co2-zone-${index}`,
          properties: {
            kind: "co2_zone",
            label: `${properties.predicted_flux < 0 ? "Carbon sink stress" : "CO2 outgassing"} zone`,
            lat,
            lon,
            severity: severityBand(score),
            severity_score: score,
            area_km2: areaKm2,
            radius_km: radiusKm,
            predicted_flux: properties.predicted_flux || 0,
            observed_flux: properties.observed_flux || properties.predicted_flux || 0,
            display_signal: properties.display_signal || 0,
            weakening_score: properties.weakening_score || 0,
            anomaly_score: properties.anomaly_score || 0,
            route_priority: properties.route_priority || 0,
            fill_color: color,
          },
        };
      }),
    [hotspotNodes, points, selectedRegion]
  );

  const trashZones = useMemo(
    () =>
      anchoredTrashTargets.map((target, index) => {
        const score = routeStrength(target);
        const radiusKm =
          (selectedRegion === "global" ? 140 : 100) +
          (target.clusterSize || 1) * 18 +
          (target.intensity || 0) * 70 +
          (target.routePriority || 0) * 34;
        const areaKm2 = Math.PI * radiusKm * radiusKm;
        const color = zoneColor("trash_zone", score);
        const routeName = target.routeTarget?.name || target.nearest_port?.name || "No verified port route available";
        const transportPath = target.metadata?.transport_path || [];
        const lastPoint = transportPath[transportPath.length - 1] || [target.lon, target.lat];
        const firstPoint = transportPath[0] || [target.lon, target.lat];
        const orientationDeg = vectorOrientationDegrees(lastPoint[0] - firstPoint[0], lastPoint[1] - firstPoint[1]);
        const seed = ((target.lon + 180) * 0.017) + ((target.lat + 90) * 0.029);
        return {
          ...oceanBlobPolygon(target.lon, target.lat, radiusKm, points, {
            orientationDeg,
            elongation: 0.18 + Math.min(0.14, (target.routePriority || 0) * 0.06),
            roughness: 0.06 + Math.min(0.05, (target.intensity || 0) * 0.03),
            seed,
            sectorCount: 28,
          }),
          id: `trash-zone-${index}`,
          properties: {
            kind: "trash_zone",
            label: target.clusterSize > 1 ? `Trash accumulation cluster x${target.clusterSize}` : "Trash accumulation zone",
            lat: target.lat,
            lon: target.lon,
            severity: severityBand(score),
            severity_score: score,
            area_km2: areaKm2,
            radius_km: radiusKm,
            intensity: target.intensity || 0,
            route_priority: target.routePriority || 0,
            weakening_score: target.weakening || target.metadata?.weakening_score || 0,
            anomaly_score: target.anomalyScore || target.metadata?.anomaly_score || 0,
            cluster_size: target.clusterSize || 1,
            route_name: routeName,
            route_mode: target.routeTarget || target.nearest_port ? "port" : "transport",
            fill_color: color,
          },
        };
      }),
    [anchoredTrashTargets, points, selectedRegion]
  );

  const hoverTargets = useMemo(
    () => [
      ...co2Zones.map((feature) => ({
        ...feature.properties,
        position: [feature.properties.lon, feature.properties.lat],
        hoverRadius: Math.max(70000, (feature.properties.radius_km || 120) * 1050),
      })),
      ...trashZones.map((feature) => ({
        ...feature.properties,
        position: [feature.properties.lon, feature.properties.lat],
        hoverRadius: Math.max(70000, (feature.properties.radius_km || 120) * 1050),
      })),
    ],
    [co2Zones, trashZones]
  );

  const zoneLabels = useMemo(() => {
    const co2Labels = co2Zones.slice(0, selectedRegion === "global" ? 12 : 16).map((feature) => ({
      id: `${feature.id}-label`,
      position: [feature.properties.lon, feature.properties.lat],
      text: `CO2 ${feature.properties.severity}`,
      color: [255, 232, 244, 230],
    }));
    const trashLabels = trashZones.slice(0, selectedRegion === "global" ? 14 : 18).map((feature) => ({
      id: `${feature.id}-label`,
      position: [feature.properties.lon, feature.properties.lat],
      text: `Trash ${feature.properties.severity}`,
      color: [255, 247, 216, 230],
    }));
    return [...co2Labels, ...trashLabels];
  }, [co2Zones, selectedRegion, trashZones]);

  const tooltipText = ({ object }) => {
    if (!object) return null;
    const zone = object.properties || object;
    if (zone.kind === "trash_zone") {
      return {
        text: `${zone.label}\nSeverity: ${zone.severity}\nEstimated spread: ${formatArea(zone.area_km2)}\nTransport intensity: ${zone.intensity.toFixed(2)}\nRoute: ${zone.route_name}\nWhy surfaced: route priority ${zone.route_priority.toFixed(2)}, weakening ${zone.weakening_score.toFixed(2)}, anomaly ${zone.anomaly_score.toFixed(2)}`,
      };
    }
    if (zone.kind === "co2_zone") {
      return {
        text: `${zone.label}\nSeverity: ${zone.severity}\nArea of effect: ${formatArea(zone.area_km2)}\nPredicted flux: ${zone.predicted_flux.toFixed(2)}\nObserved reference: ${zone.observed_flux.toFixed(2)}\nWhy surfaced: signal ${zone.display_signal.toFixed(2)}, weakening ${zone.weakening_score.toFixed(2)}, anomaly ${zone.anomaly_score.toFixed(2)}`,
      };
    }
    if (object.routeTarget?.name) {
      return {
        text: `${object.label}\nObserved trash intensity: ${(
          object.intensity || object.routePriority || 0
        ).toFixed(2)}\nNearest port: ${object.routeTarget.name}`,
      };
    }
    if (object.source && object.target) {
      return {
        text: `${object.label}\nPriority corridor score: ${object.density.toFixed(2)}\nSink weakening nearby: ${object.weakening.toFixed(2)}`,
      };
    }
    if (typeof object.anomaly_score === "number") {
      return {
        text: `${object.region_name}\nAnomaly score: ${object.anomaly_score}\nWeakening score: ${(object.weakening_score || 0).toFixed(2)}`,
      };
    }
    if (typeof object.predicted_flux === "number") {
      return {
        text: `Predicted flux: ${object.predicted_flux.toFixed(2)}\nObserved flux: ${object.observed_flux.toFixed(2)}\nWeakening: ${object.weakening_score.toFixed(2)}`,
      };
    }
    return null;
  };

  const layers = useMemo(
    () => {
      const activeLayers = [];
      activeLayers.push(
      new PathLayer({
        id: "recovery-routes",
        data: routeSegments,
        getPath: (d) => d.path,
        getColor: (d) =>
          d.weakening > 0.35
            ? [255, 120, 98, verifiedMap ? 225 : 150]
            : [255, 209, 102, verifiedMap ? 214 : 138],
        getWidth: (d) => 9000 + d.density * 15000,
        widthMinPixels: verifiedMap ? 3 : 2,
        rounded: true,
        pickable: true,
      }),
      new GeoJsonLayer({
        id: "co2-zones-hitbox",
        data: co2Zones,
        filled: true,
        stroked: false,
        getFillColor: [0, 0, 0, 0],
        pickable: true,
      }),
      new GeoJsonLayer({
        id: "trash-zones-hitbox",
        data: trashZones,
        filled: true,
        stroked: false,
        getFillColor: [0, 0, 0, 0],
        pickable: true,
      }),
      new ScatterplotLayer({
        id: "zone-hover-targets",
        data: hoverTargets,
        getPosition: (d) => d.position,
        getRadius: (d) => d.hoverRadius,
        getFillColor: [0, 0, 0, 0],
        radiusMinPixels: 14,
        stroked: false,
        pickable: true,
      }),
      new ScatterplotLayer({
        id: "anomalies",
        data: visibleAnomalies,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (d) => 90000 + d.anomaly_score * 150000,
        radiusMinPixels: selectedRegion === "global" ? 10 : 12,
        getFillColor: [255, 90, 95, 110],
        getLineColor: [255, 220, 140, 210],
        lineWidthMinPixels: 1.6,
        stroked: true,
        pickable: true,
      }));
      if (anchoredTrashTargets.length) {
        activeLayers.push(
      new TextLayer({
        id: "zone-labels",
        data: zoneLabels,
        getPosition: (d) => d.position,
        getText: (d) => d.text,
        getColor: (d) => d.color,
        getSize: 12.5,
        sizeUnits: "pixels",
        getPixelOffset: [0, -16],
        getTextAnchor: "middle",
        getAlignmentBaseline: "bottom",
        pickable: false,
      }));
      }
      if (!verifiedMap && routeSegments.length) {
        activeLayers.push(
          new PathLayer({
            id: "fallback-recovery-routes",
            data: routeSegments,
            getPath: (d) => d.path,
            getColor: [255, 209, 102, 180],
            getWidth: (d) => 8000 + d.density * 12000,
            widthMinPixels: 2,
            rounded: true,
            pickable: true,
          })
        );
      }
      return activeLayers;
    },
    [anchoredTrashTargets.length, co2Zones, hoverTargets, routeSegments, selectedRegion, trashZones, verifiedMap, visibleAnomalies, zoneLabels]
  );

  return (
    <div className="ocean-map">
      {showBrief ? (
        <div className="map-brief">
          <div>
            <span className="map-brief-label">Mission Map</span>
            <strong>CO2 stress zones, trash accumulation areas, and recovery corridors on a satellite basemap</strong>
            {verifiedMap && prioritySummary.degradationTarget ? (
              <small className="map-brief-subcopy">
                Hover any zone to inspect severity, area of effect, and routing context. Top route target:{" "}
                {routedTrashTargets[0]?.routeTarget?.name || "no verified port route available"}
              </small>
            ) : (
              <small className="map-brief-subcopy">
                Provisional mission zones stay visible while the verified grid catches up. Severity, footprint, and transport routes update as checkpoint-backed products refresh.
              </small>
            )}
          </div>
          <div className="map-brief-actions">
            <span className="map-brief-meta">{selectedRegion} view</span>
            <button type="button" className="map-brief-close" onClick={() => setShowBrief(false)}>
              Hide
            </button>
          </div>
        </div>
      ) : (
        <button type="button" className="map-brief-restore" onClick={() => setShowBrief(true)}>
          Show map brief
        </button>
      )}
      <DeckGL
        controller={{ dragRotate: false, touchRotate: false }}
        viewState={viewState}
        layers={layers}
        onViewStateChange={({ viewState: nextViewState }) => setViewState(nextViewState)}
        onClick={({ coordinate, object }) => {
          if (typeof object?.lat === "number" && typeof object?.lon === "number") {
            setSelectedPoint({ lat: object.lat, lon: normalizeLon(object.lon) });
          } else if (coordinate) {
            setSelectedPoint({ lat: coordinate[1], lon: normalizeLon(coordinate[0]) });
          }
        }}
        getTooltip={tooltipText}
      >
        <Suspense fallback={null}>
          <MapLibreSurface
            basemapStyle={basemapStyle}
            heatmapFeatures={fieldHighlights}
            co2Zones={co2Zones}
            trashZones={trashZones}
            co2Hotspots={[]}
            trashTargets={[]}
            routeSegments={routeSegments}
            anomalies={[]}
            resolution={gridResolution}
          />
        </Suspense>
      </DeckGL>
      <div className="map-legend">
        <div className="map-legend-header">
          <strong>Mission Layers</strong>
          <span>Hover zones for severity, footprint, and routing</span>
        </div>
        <div className="map-legend-row">
          <span className="legend-swatch legend-swatch-co2" />
          <span>CO2 hotspot zone</span>
          <small>magenta fill · severity grows with opacity and outline</small>
        </div>
        <div className="map-legend-row">
          <span className="legend-swatch legend-swatch-trash" />
          <span>Trash accumulation zone</span>
          <small>gold fill · size reflects spread and transport intensity</small>
        </div>
        <div className="map-legend-row">
          <span className="legend-swatch legend-swatch-route" />
          <span>Recovery route</span>
          <small>shown only when a real port is resolved from the live port source</small>
        </div>
      </div>
      {!verifiedMap ? (
        <div className="map-empty-state">
          <strong>Verified CO2 layers are still catching up.</strong>
          <span>
            {anchoredTrashTargets.length
              ? "Trash and routing overlays are shown from the model-first transport prediction while the CO2 surface is being verified. "
              : ""}
            {sourceSummary || "The current CO2 layer is provisional. It uses only real data paths and stays limited until the checkpoint-backed grid is ready."}
          </span>
        </div>
      ) : null}
    </div>
  );
}

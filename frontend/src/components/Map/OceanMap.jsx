import { Suspense, lazy, useMemo, useState } from "react";

import { PathLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
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
  const rawRouteTargets = useMemo(
    () =>
      [...(trashData?.hotspots || [])]
        .filter(
          (target) =>
            target.nearest_port ||
            (Array.isArray(target.metadata?.transport_path) && target.metadata.transport_path.length > 1)
        )
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
        .filter(
          (target) =>
            target.routeTarget ||
            target.nearest_port ||
            (Array.isArray(target.metadata?.transport_path) && target.metadata.transport_path.length > 1)
        )
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
        selected.push(item);
        if (selected.length >= (selectedRegion === "global" ? 18 : 20)) {
          break;
        }
      }
      return selected;
    },
    [rawRouteTargets, selectedRegion]
  );

  const routeSegments = useMemo(
    () =>
      routedTrashTargets
        .map((item) => {
          const transportPath =
            Array.isArray(item.metadata?.transport_path) && item.metadata.transport_path.length > 1
              ? item.metadata.transport_path
              : null;
          const routeTarget = item.routeTarget || item.nearest_port || null;
          const path = routeTarget
            ? [
                ...(transportPath || [[item.lon, item.lat]]),
                [routeTarget.lon, routeTarget.lat],
              ]
            : transportPath || [[item.lon, item.lat]];
          return {
            id: `${item.id}-route`,
            path,
            source: [item.lon, item.lat],
            target: transportPath
              ? transportPath[transportPath.length - 1]
              : routeTarget
                ? [routeTarget.lon, routeTarget.lat]
                : [item.lon, item.lat],
            density: item.routePriority || item.intensity || 0,
            label: routeTarget
              ? `${item.label} to ${routeTarget.name}`
              : `${item.label} current transport`,
            weakening: item.weakening || item.metadata?.weakening_score || 0,
            routeMode: routeTarget ? "port" : "transport",
          };
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
      ...visibleTrashTargets.map((target) => ({ lat: target.lat, lon: target.lon })),
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
  }, [hotspotNodes, points, selectedRegion, visibleAnomalies, visibleTrashTargets, weakeningZones]);

  const tooltipText = ({ object }) => {
    if (!object) return null;
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
      new ScatterplotLayer({
        id: "sink-nodes",
        data: hotspotNodes,
        getPosition: (d) => d.geometry.coordinates,
        getRadius: (d) => 24000 + Math.abs(d.properties.predicted_flux) * 12000 + (d.properties.weakening_score || 0) * 42000,
        radiusMinPixels: selectedRegion === "global" ? 6 : 8,
        getFillColor: (d) =>
          d.properties.predicted_flux < 0 ? [64, 219, 168, 130] : [255, 135, 102, 130],
        getLineColor: [214, 244, 255, 190],
        lineWidthMinPixels: 1.6,
        stroked: true,
        pickable: true,
      }),
      new ScatterplotLayer({
        id: "weakening-zones",
        data: weakeningZones,
        getPosition: (d) => d.geometry.coordinates,
        getRadius: (d) => 22000 + d.properties.weakening_score * 52000,
        radiusMinPixels: selectedRegion === "global" ? 8 : 10,
        getFillColor: [255, 114, 94, 54],
        getLineColor: [255, 184, 116, 170],
        lineWidthMinPixels: 1.2,
        stroked: true,
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
      if (visibleTrashTargets.length) {
        activeLayers.push(
      new ScatterplotLayer({
        id: "trash-targets",
        data: visibleTrashTargets,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (d) => 60000 + (d.clusterSize || 1) * 15000 + (d.routePriority || d.intensity || 0) * 40000,
        radiusMinPixels: selectedRegion === "global" ? 12 : 14,
        getFillColor: (d) =>
          d.observed
            ? [255, 174, 66, 90 + Math.round(Math.min(d.intensity || 0, 1.4) * 80)]
            : [255, 196, 61, 80 + Math.round(Math.min(d.routePriority || 0, 1.4) * 72)],
        getLineColor: (d) => (d.observed ? [255, 232, 188, 200] : [255, 230, 160, 180]),
        lineWidthMinPixels: 4,
        stroked: true,
        pickable: true,
      }),
      new TextLayer({
        id: "recovery-target-labels",
        data: routeLabels,
        getPosition: (d) => d.position,
        getText: (d) => d.label,
        getColor: [246, 249, 252, 210],
        getSize: 12,
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
    [hotspotNodes, routeLabels, routeSegments, selectedRegion, verifiedMap, viewState.zoom, visibleAnomalies, visibleTrashTargets, weakeningZones]
  );

  return (
    <div className="ocean-map">
      {showBrief ? (
        <div className="map-brief">
          <div>
            <span className="map-brief-label">Mission Map</span>
            <strong>CO2 weakening, recovery targets, and port corridors on a satellite basemap</strong>
            {verifiedMap && prioritySummary.degradationTarget ? (
              <small className="map-brief-subcopy">
                Top weakening cell: {prioritySummary.degradationTarget.properties.weakening_score.toFixed(2)} · Top route
                target: {visibleTrashTargets[0]?.routeTarget?.name || "awaiting routing target"}
              </small>
            ) : (
              <small className="map-brief-subcopy">
                Provisional map overlays stay visible while the verified grid catches up. Checkpoint-backed products refresh automatically as training and ingestion progress.
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
            co2Hotspots={hotspotNodes}
            trashTargets={visibleTrashTargets}
            routeSegments={routeSegments}
            anomalies={visibleAnomalies}
            resolution={gridResolution}
          />
        </Suspense>
      </DeckGL>
      {!verifiedMap ? (
        <div className="map-empty-state">
          <strong>Verified CO2 layers are still catching up.</strong>
          <span>
            {visibleTrashTargets.length
              ? "Trash and routing overlays are shown from the model-first transport prediction while the CO2 surface is being verified. "
              : ""}
            {sourceSummary || "The current CO2 layer is provisional. It uses only real data paths and stays limited until the checkpoint-backed grid is ready."}
          </span>
        </div>
      ) : null}
    </div>
  );
}

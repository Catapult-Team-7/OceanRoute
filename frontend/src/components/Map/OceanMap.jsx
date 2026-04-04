import { Suspense, lazy, useMemo, useState } from "react";

import { PathLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";

import { useHeatmapData } from "../../hooks/useHeatmapData";
import { useOceanStore } from "../../store/oceanStore";
import { fluxToColor } from "../../utils/colorScale";

const MapLibreSurface = lazy(() => import("./MapLibreSurface"));
const INITIAL_VIEW_STATE = { longitude: 0, latitude: 12, zoom: 1.15, pitch: 0, bearing: 0 };

const BASE_PLASTIC_HOTSPOTS = [
  {
    id: "north-pacific-garbage-patch",
    label: "North Pacific Patch",
    lat: 33.2,
    lon: -145.4,
    density: 0.94,
    tonnage: "79K tons",
    routeTarget: { lat: 37.77, lon: -122.42, port: "San Francisco" },
  },
  {
    id: "indian-ocean-gyre",
    label: "Indian Ocean Gyre",
    lat: -18.4,
    lon: 82.1,
    density: 0.72,
    tonnage: "31K tons",
    routeTarget: { lat: 1.29, lon: 103.85, port: "Singapore" },
  },
  {
    id: "south-atlantic-drift",
    label: "South Atlantic Drift",
    lat: -27.6,
    lon: -14.8,
    density: 0.66,
    tonnage: "18K tons",
    routeTarget: { lat: -33.92, lon: 18.42, port: "Cape Town" },
  },
];

function nearestInsight(points, lat, lon) {
  if (!points.length) return null;
  let best = points[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const point of points) {
    const [pointLon, pointLat] = point.geometry.coordinates;
    const distance = Math.hypot((pointLat - lat) / 10, (pointLon - lon) / 14);
    if (distance < bestDistance) {
      best = point;
      bestDistance = distance;
    }
  }
  return best?.properties || null;
}

export default function OceanMap() {
  useHeatmapData();

  const heatmapData = useOceanStore((state) => state.heatmapData);
  const anomalies = useOceanStore((state) => state.anomalies);
  const selectedRegion = useOceanStore((state) => state.selectedRegion);
  const setSelectedPoint = useOceanStore((state) => state.setSelectedPoint);
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [showBrief, setShowBrief] = useState(true);
  const verifiedMap = Boolean(heatmapData?.metadata?.verified_map);
  const sourceSummary = heatmapData?.metadata?.source_summary || "";

  const points = heatmapData?.features || [];
  const plasticHotspots = useMemo(() => {
    const regionMatchers = {
      global: () => true,
      pacific: (item) => item.lon >= 110 || item.lon <= -70,
      atlantic: (item) => item.lon >= -80 && item.lon <= 20,
      indian: (item) => item.lon >= 20 && item.lon <= 120,
    };
    const matcher = regionMatchers[selectedRegion] || regionMatchers.global;
    return BASE_PLASTIC_HOTSPOTS.filter(matcher).map((item) => {
      const insight = nearestInsight(points, item.lat, item.lon);
      const weakening = insight?.weakening_score || 0;
      const routePriority = insight?.route_priority || 0;
      return {
        ...item,
        weakening,
        routePriority,
        density: Math.min(1.3, item.density + routePriority * 0.06),
      };
    });
  }, [points, selectedRegion]);

  const routeSegments = useMemo(
    () =>
      plasticHotspots.map((item) => ({
        id: `${item.id}-route`,
        path: [
          [item.lon, item.lat],
          [item.routeTarget.lon, item.routeTarget.lat],
        ],
        source: [item.lon, item.lat],
        target: [item.routeTarget.lon, item.routeTarget.lat],
        density: item.routePriority || item.density,
        label: `${item.label} to ${item.routeTarget.port}`,
        weakening: item.weakening,
      })),
    [plasticHotspots]
  );

  const sinkNodes = useMemo(
    () => points.filter((point, index) => point.properties.predicted_flux < 0 && index % 5 === 0).slice(0, 320),
    [points]
  );

  const weakeningZones = useMemo(
    () =>
      points
        .filter((point, index) => point.properties.weakening_score > 0.28 && index % 3 === 0)
        .slice(0, 180),
    [points]
  );

  const displayPoints = useMemo(() => {
    const stride = selectedRegion === "global" && viewState.zoom < 1.8 ? 2 : 1;
    return points.filter((_, index) => index % stride === 0);
  }, [points, selectedRegion, viewState.zoom]);

  const routeLabels = useMemo(
    () =>
      plasticHotspots.map((item) => ({
        id: `${item.id}-label`,
        position: [item.lon, item.lat],
        label: item.label,
      })),
    [plasticHotspots]
  );

  const tooltipText = ({ object }) => {
    if (!object) return null;
    if (object.routeTarget?.port) {
      return {
        text: `${object.label}\nPlastic load: ${object.tonnage}\nWeakening score: ${object.weakening.toFixed(2)}\nRecommended port: ${object.routeTarget.port}`,
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
    () =>
      verifiedMap
        ? [
      new ScatterplotLayer({
        id: "flux-cells",
        data: displayPoints,
        getPosition: (d) => d.geometry.coordinates,
        radiusUnits: "meters",
        getRadius: (d) => 65000 + Math.min(Math.abs(d.properties.predicted_flux), 4) * 18000,
        getFillColor: (d) => [...fluxToColor(d.properties.predicted_flux), 112],
        getLineColor: (d) => [...fluxToColor(d.properties.predicted_flux), 190],
        lineWidthMinPixels: viewState.zoom > 2 ? 1 : 0,
        stroked: viewState.zoom > 2,
        pickable: true,
      }),
      new PathLayer({
        id: "recovery-routes",
        data: routeSegments,
        getPath: (d) => d.path,
        getColor: (d) => (d.weakening > 0.35 ? [255, 120, 98, 225] : [255, 209, 102, 214]),
        getWidth: (d) => 9000 + d.density * 15000,
        widthMinPixels: 3,
        rounded: true,
        pickable: true,
      }),
      new ScatterplotLayer({
        id: "sink-nodes",
        data: sinkNodes,
        getPosition: (d) => d.geometry.coordinates,
        getRadius: (d) => 10000 + Math.abs(d.properties.predicted_flux) * 6000,
        getFillColor: (d) =>
          d.properties.weakening_score > 0.3 ? [125, 231, 245, 78] : [64, 219, 168, 62],
        getLineColor: [214, 244, 255, 72],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
      }),
      new ScatterplotLayer({
        id: "weakening-zones",
        data: weakeningZones,
        getPosition: (d) => d.geometry.coordinates,
        getRadius: (d) => 22000 + d.properties.weakening_score * 52000,
        getFillColor: [255, 114, 94, 28],
        getLineColor: [255, 184, 116, 120],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
      }),
      new ScatterplotLayer({
        id: "anomalies",
        data: anomalies,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (d) => 90000 + d.anomaly_score * 150000,
        getFillColor: [255, 90, 95, 78],
        getLineColor: [255, 220, 140, 148],
        lineWidthMinPixels: 1,
        stroked: true,
        pickable: true,
      }),
      new ScatterplotLayer({
        id: "plastic-hotspots",
        data: plasticHotspots,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (d) => 90000 + d.density * 110000,
        getFillColor: (d) => [255, 196, 61, 34 + Math.round(d.density * 52)],
        getLineColor: [255, 230, 160, 180],
        lineWidthMinPixels: 3,
        stroked: true,
        pickable: true,
      }),
      new TextLayer({
        id: "plastic-hotspot-labels",
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
      }),
    ]
        : [],
    [anomalies, displayPoints, plasticHotspots, routeLabels, routeSegments, sinkNodes, verifiedMap, viewState.zoom, weakeningZones]
  );

  return (
    <div className="ocean-map">
      {showBrief ? (
        <div className="map-brief">
          <div>
            <span className="map-brief-label">Mission Map</span>
            <strong>Carbon sinks, weakening zones, and recovery corridors</strong>
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
            setSelectedPoint({ lat: object.lat, lon: object.lon });
          } else if (coordinate) {
            setSelectedPoint({ lat: coordinate[1], lon: coordinate[0] });
          }
        }}
        getTooltip={tooltipText}
      >
        <Suspense fallback={null}>
          <MapLibreSurface />
        </Suspense>
      </DeckGL>
      {!verifiedMap ? (
        <div className="map-empty-state">
          <strong>No verified ocean layers are available yet.</strong>
          <span>{sourceSummary || "The current backend map is still demo-backed, so overlays are intentionally hidden."}</span>
        </div>
      ) : null}
    </div>
  );
}

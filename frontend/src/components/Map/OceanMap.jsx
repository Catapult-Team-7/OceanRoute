import { Suspense, lazy, useMemo, useState } from "react";

import { PathLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";

import { useHeatmapData } from "../../hooks/useHeatmapData";
import { useTrashData } from "../../hooks/useTrashData";
import { useOceanStore } from "../../store/oceanStore";
import { fluxToColor } from "../../utils/colorScale";
import { buildDisplayTrashTargets, buildRecoveryTargets, buildTopMissionTargets } from "../../utils/missionInsights";

const MapLibreSurface = lazy(() => import("./MapLibreSurface"));
const INITIAL_VIEW_STATE = { longitude: 0, latitude: 12, zoom: 1.15, pitch: 0, bearing: 0 };

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

  const points = heatmapData?.features || [];
  const recoveryTargets = useMemo(
    () => buildRecoveryTargets(points, viewState.zoom, verifiedMap),
    [points, verifiedMap, viewState.zoom]
  );
  const trashTargets = useMemo(
    () => buildDisplayTrashTargets(trashData?.hotspots || [], recoveryTargets, viewState.zoom),
    [trashData?.hotspots, recoveryTargets, viewState.zoom]
  );
  const prioritySummary = useMemo(
    () => buildTopMissionTargets(points, anomalies, viewState.zoom, verifiedMap),
    [anomalies, points, verifiedMap, viewState.zoom]
  );

  const routeSegments = useMemo(
    () =>
      trashTargets
        .filter((item) => item.routeTarget)
        .map((item) => ({
        id: `${item.id}-route`,
        path: [
          [item.lon, item.lat],
          [item.routeTarget.lon, item.routeTarget.lat],
        ],
        source: [item.lon, item.lat],
        target: [item.routeTarget.lon, item.routeTarget.lat],
        density: item.routePriority || item.intensity || 0,
        label: `${item.label} to ${item.routeTarget.name}`,
        weakening: item.weakening || 0,
      })),
    [trashTargets]
  );

  const hotspotNodes = useMemo(
    () =>
      [...points]
        .filter((point) => point?.properties)
        .sort(
          (a, b) =>
            Math.abs((b.properties.predicted_flux || 0) - (b.properties.observed_flux || 0)) +
              (b.properties.weakening_score || 0) -
            (Math.abs((a.properties.predicted_flux || 0) - (a.properties.observed_flux || 0)) +
              (a.properties.weakening_score || 0))
        )
        .slice(0, 320),
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
      trashTargets.slice(0, viewState.zoom > 2.2 ? 10 : 5).map((item) => ({
        id: `${item.id}-label`,
        position: [item.lon, item.lat],
        label: item.label,
      })),
    [trashTargets, viewState.zoom]
  );

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
      new ScatterplotLayer({
        id: "flux-cells",
        data: displayPoints,
        getPosition: (d) => d.geometry.coordinates,
        radiusUnits: "meters",
        getRadius: (d) => 65000 + Math.min(Math.abs(d.properties.predicted_flux), 4) * 18000,
        getFillColor: (d) => [...fluxToColor(d.properties.predicted_flux), verifiedMap ? 112 : 58],
        getLineColor: (d) => [...fluxToColor(d.properties.predicted_flux), verifiedMap ? 190 : 112],
        lineWidthMinPixels: viewState.zoom > 2 ? 1 : 0.5,
        stroked: true,
        pickable: true,
      }),
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
        getRadius: (d) => 10000 + Math.abs(d.properties.predicted_flux) * 6000,
        getFillColor: (d) =>
          d.properties.predicted_flux < 0 ? [64, 219, 168, 72] : [255, 135, 102, 72],
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
      }));
      if (trashTargets.length) {
        activeLayers.push(
      new ScatterplotLayer({
        id: "trash-targets",
        data: trashTargets,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (d) => 60000 + (d.clusterSize || 1) * 15000 + (d.routePriority || d.intensity || 0) * 40000,
        getFillColor: (d) =>
          d.observed
            ? [255, 174, 66, 54 + Math.round(Math.min(d.intensity || 0, 1.4) * 80)]
            : [255, 196, 61, 42 + Math.round(Math.min(d.routePriority || 0, 1.4) * 72)],
        getLineColor: (d) => (d.observed ? [255, 232, 188, 200] : [255, 230, 160, 180]),
        lineWidthMinPixels: 3,
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
    [anomalies, displayPoints, hotspotNodes, routeLabels, routeSegments, trashTargets, verifiedMap, viewState.zoom, weakeningZones]
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
                target: {trashTargets[0]?.routeTarget?.name || "awaiting routing target"}
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
            setSelectedPoint({ lat: object.lat, lon: object.lon });
          } else if (coordinate) {
            setSelectedPoint({ lat: coordinate[1], lon: coordinate[0] });
          }
        }}
        getTooltip={tooltipText}
      >
        <Suspense fallback={null}>
          <MapLibreSurface basemapStyle={basemapStyle} />
        </Suspense>
      </DeckGL>
      {!verifiedMap ? (
        <div className="map-empty-state">
          <strong>Verified CO2 layers are still catching up.</strong>
          <span>
            {trashTargets.length
              ? "Trash and routing overlays are shown only from measured feeds while the CO2 surface is being verified. "
              : ""}
            {sourceSummary || "The current CO2 layer is provisional. It uses only real data paths and stays limited until the checkpoint-backed grid is ready."}
          </span>
        </div>
      ) : null}
    </div>
  );
}

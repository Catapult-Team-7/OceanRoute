import { useEffect, useMemo, useState } from "react";

import { PathStyleExtension } from "@deck.gl/extensions";
import { PathLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";
import maplibregl from "maplibre-gl";
import Map from "react-map-gl/maplibre";

import { buildRouteCoordinates, confidenceColor, formatRange, regionViewport, resolveRouteSteps } from "../lib/mission-utils";
import type { ForecastSnapshot, ForecastStep, RegionInfo, RouteFormState, RoutePlan } from "../types";

interface MissionMapCanvasProps {
  forecast: ForecastSnapshot | null;
  route: RoutePlan | null;
  selectedRegion: RegionInfo | null;
  routeForm: RouteFormState;
}

interface MapViewState {
  latitude: number;
  longitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

const MAP_STYLE = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
} as const;

function hotspotRadius(step: ForecastStep): number {
  return 1800 + Math.max(step.expected_kg_max, 0.6) * 2800;
}

export function MissionMapCanvas({ forecast, route, selectedRegion, routeForm }: MissionMapCanvasProps) {
  const [viewState, setViewState] = useState<MapViewState>(() => {
    const base = regionViewport(selectedRegion);
    return { ...base, pitch: 24, bearing: 0 };
  });

  useEffect(() => {
    const base = regionViewport(selectedRegion ?? forecast?.region ?? null);
    setViewState((current) => ({
      ...current,
      latitude: base.latitude,
      longitude: base.longitude,
      zoom: base.zoom,
      pitch: current.pitch,
      bearing: 0,
    }));
  }, [forecast?.region, selectedRegion]);

  const routeSteps = useMemo(() => resolveRouteSteps(route, forecast), [forecast, route]);
  const routePath = useMemo(() => buildRouteCoordinates(route, forecast, routeForm), [forecast, route, routeForm]);
  const pathExtensions = useMemo(() => [new PathStyleExtension({ dash: true })], []);

  const hotspotLayers = useMemo(() => {
    const layers: object[] = [];
    if (forecast?.steps.length) {
      layers.push(
        new ScatterplotLayer<ForecastStep>({
          id: "forecast-hotspots",
          data: forecast.steps,
          getPosition: (step) => [step.lon, step.lat],
          getRadius: (step) => hotspotRadius(step),
          radiusUnits: "meters",
          radiusMinPixels: 7,
          radiusMaxPixels: 28,
          getFillColor: (step) => confidenceColor(step),
          getLineColor: [241, 247, 255, 235],
          lineWidthMinPixels: 1.4,
          stroked: true,
          pickable: true,
        }),
      );
      layers.push(
        new TextLayer({
          id: "hotspot-labels",
          data: forecast.top_hotspots.slice(0, 8),
          getPosition: (step: { lon: number; lat: number }) => [step.lon, step.lat],
          getText: (step: { cell_id: string }) => step.cell_id.replaceAll("_", " "),
          getColor: [243, 246, 255, 230],
          getSize: 13,
          sizeUnits: "pixels",
          getTextAnchor: "middle",
          getAlignmentBaseline: "bottom",
          getPixelOffset: [0, -14],
        }),
      );
    }
    if (routePath.length > 1) {
      layers.push(
        new PathLayer({
          id: "route-path",
          data: [{ path: routePath }],
          getPath: (item: { path: [number, number][] }) => item.path,
          getColor: route?.recommended_mode === "recon" ? [245, 179, 63, 240] : [38, 145, 255, 245],
          getWidth: route?.recommended_mode === "recon" ? 420 : 520,
          widthUnits: "meters",
          widthMinPixels: route?.recommended_mode === "recon" ? 3 : 4,
          rounded: true,
          dashJustified: true,
          getDashArray: route?.recommended_mode === "recon" ? [5, 4] : [0, 0],
          extensions: pathExtensions,
        }),
      );
      layers.push(
        new ScatterplotLayer({
          id: "route-stops",
          data: routeSteps,
          getPosition: (step: ForecastStep) => [step.lon, step.lat],
          getRadius: 2300,
          radiusUnits: "meters",
          radiusMinPixels: 9,
          getFillColor: route?.recommended_mode === "recon" ? [255, 228, 171, 230] : [213, 234, 255, 235],
          getLineColor: route?.recommended_mode === "recon" ? [181, 123, 28, 245] : [20, 92, 169, 245],
          lineWidthMinPixels: 2,
          stroked: true,
          pickable: true,
        }),
      );
    }
    layers.push(
      new ScatterplotLayer({
        id: "depot",
        data: [{ lat: routeForm.depotLat, lon: routeForm.depotLon }],
        getPosition: (point: { lat: number; lon: number }) => [point.lon, point.lat],
        getRadius: 2800,
        radiusUnits: "meters",
        radiusMinPixels: 11,
        getFillColor: [247, 250, 255, 255],
        getLineColor: [9, 48, 82, 255],
        lineWidthMinPixels: 2.6,
        stroked: true,
      }),
    );
    return layers;
  }, [forecast?.steps, forecast?.top_hotspots, pathExtensions, route?.recommended_mode, routeForm.depotLat, routeForm.depotLon, routePath, routeSteps]);

  return (
    <div className="mission-map-canvas">
      <div className="mission-map-frame">
        <DeckGL
          controller={{ dragRotate: false, touchRotate: false }}
          layers={hotspotLayers}
          viewState={viewState}
          style={{ position: "absolute", inset: 0 }}
          onViewStateChange={({ viewState: nextViewState }) => {
            setViewState({
              latitude: nextViewState.latitude,
              longitude: nextViewState.longitude,
              zoom: nextViewState.zoom,
              pitch: nextViewState.pitch ?? 0,
              bearing: nextViewState.bearing ?? 0,
            });
          }}
          getTooltip={({ object }) => {
            if (!object) {
              return null;
            }
            const item = object as Partial<ForecastStep> & { lat?: number; lon?: number };
            if (item.cell_id && typeof item.expected_kg_min === "number" && typeof item.expected_kg_max === "number") {
              return {
                text: `${item.cell_id.replaceAll("_", " ")}\n${formatRange(item.expected_kg_min, item.expected_kg_max)}\nConfidence ${Math.round((item.confidence ?? 0) * 100)}%`,
              };
            }
            if (typeof item.lat === "number" && typeof item.lon === "number") {
              return { text: "Mission depot" };
            }
            return null;
          }}
        >
          <Map
            reuseMaps
            mapLib={maplibregl}
            mapStyle={MAP_STYLE}
            attributionControl={false}
            dragRotate={false}
            pitchWithRotate={false}
            style={{ width: "100%", height: "100%" }}
          />
        </DeckGL>
      </div>
    </div>
  );
}

import { useMemo } from "react";

import { Layer, Map as MapLibreMap, Source } from "react-map-gl/maplibre";

import { MAP_STYLES } from "../../utils/constants";
import { fluxToColor } from "../../utils/colorScale";

function pointFeature(id, lon, lat, properties = {}) {
  return {
    type: "Feature",
    id,
    geometry: {
      type: "Point",
      coordinates: [lon, lat],
    },
    properties,
  };
}

function cellPolygonFeature(id, lon, lat, properties = {}, latStep = 2, lonStep = 2) {
  const halfLat = latStep / 2;
  const halfLon = lonStep / 2;
  const west = Math.max(-180, lon - halfLon);
  const east = Math.min(180, lon + halfLon);
  const south = Math.max(-89.5, lat - halfLat);
  const north = Math.min(89.5, lat + halfLat);
  return {
    type: "Feature",
    id,
    geometry: {
      type: "Polygon",
      coordinates: [[
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
      ]],
    },
    properties,
  };
}

export default function MapLibreSurface({
  basemapStyle = "satellite",
  heatmapFeatures = [],
  co2Hotspots = [],
  trashTargets = [],
  routeSegments = [],
  anomalies = [],
  resolution = "2deg",
}) {
  const heatmapStep = resolution === "2deg" ? 2 : 1;

  const heatmapGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: heatmapFeatures.slice(0, 18000).map((feature, index) => {
        const [lon, lat] = feature.geometry.coordinates;
        const flux = feature.properties?.predicted_flux ?? feature.properties?.flux ?? 0;
        const [r, g, b] = fluxToColor(flux);
        const weakening = feature.properties?.weakening_score || 0;
        const routePriority = feature.properties?.route_priority || 0;
        const intensity = Math.min(0.5, 0.08 + Math.abs(flux) * 0.06 + weakening * 0.18 + routePriority * 0.14);
        return cellPolygonFeature(
          `heat-cell-${index}`,
          lon,
          lat,
          {
            flux,
            fill: `rgba(${r}, ${g}, ${b}, ${intensity.toFixed(3)})`,
            stroke: `rgba(${r}, ${g}, ${b}, ${(Math.min(0.72, intensity + 0.18)).toFixed(3)})`,
          },
          heatmapStep,
          heatmapStep
        );
      }),
    }),
    [heatmapFeatures, heatmapStep]
  );

  const routeGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: routeSegments.map((segment) => ({
        type: "Feature",
        id: segment.id,
        geometry: {
          type: "LineString",
          coordinates: segment.path,
        },
        properties: {
          density: segment.density || 0,
          weakening: segment.weakening || 0,
        },
      })),
    }),
    [routeSegments]
  );

  const routeEndpointGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: routeSegments.flatMap((segment) => [
        pointFeature(`route-origin-${segment.id}`, segment.source[0], segment.source[1], {
          point_type: "trash_origin",
          label: segment.label,
          density: segment.density || 0,
        }),
        pointFeature(`route-target-${segment.id}`, segment.target[0], segment.target[1], {
          point_type: "port_target",
          label: segment.label,
          density: segment.density || 0,
        }),
      ]),
    }),
    [routeSegments]
  );

  const hotspotGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: co2Hotspots.slice(0, 120).map((point, index) =>
        pointFeature(
          `co2-hotspot-${index}`,
          point.geometry.coordinates[0],
          point.geometry.coordinates[1],
          {
            predicted_flux: point.properties?.predicted_flux || 0,
            weakening_score: point.properties?.weakening_score || 0,
          }
        )
      ),
    }),
    [co2Hotspots]
  );

  const anomalyGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: anomalies.slice(0, 24).map((anomaly) =>
        pointFeature(`anomaly-${anomaly.id}`, anomaly.lon, anomaly.lat, {
          anomaly_score: anomaly.anomaly_score || 0,
        })
      ),
    }),
    [anomalies]
  );

  const trashGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: trashTargets.slice(0, 40).map((target) =>
        pointFeature(`trash-${target.id}`, target.lon, target.lat, {
          intensity: target.intensity || 0,
          route_priority: target.routePriority || 0,
          observed: target.observed ? 1 : 0,
          label: target.label || "Trash hotspot",
          route_target_name: target.routeTarget?.name || "Nearest port",
        })
      ),
    }),
    [trashTargets]
  );

  const trashLabelGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: trashTargets.slice(0, 16).map((target) =>
        pointFeature(`trash-label-${target.id}`, target.lon, target.lat, {
          label: target.clusterSize > 1 ? `Trash x${target.clusterSize}` : "Trash",
        })
      ),
    }),
    [trashTargets]
  );

  const hotspotLabelGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: co2Hotspots.slice(0, 14).map((point, index) =>
        pointFeature(
          `co2-label-${index}`,
          point.geometry.coordinates[0],
          point.geometry.coordinates[1],
          {
            label:
              (point.properties?.predicted_flux || 0) < 0
                ? "CO2 sink"
                : "CO2 source",
          }
        )
      ),
    }),
    [co2Hotspots]
  );

  return (
    <MapLibreMap mapStyle={MAP_STYLES[basemapStyle] || MAP_STYLES.satellite} reuseMaps>
      {heatmapFeatures.length ? (
        <Source id="co2-field-source" type="geojson" data={heatmapGeojson}>
          <Layer
            id="co2-field-fill"
            type="fill"
            paint={{
              "fill-color": ["get", "fill"],
              "fill-opacity": 1,
            }}
          />
          <Layer
            id="co2-field-outline"
            type="line"
            paint={{
              "line-color": ["get", "stroke"],
              "line-opacity": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                0.04,
                2,
                0.08,
                4,
                0.14,
              ],
              "line-width": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                0.15,
                2,
                0.3,
                4,
                0.55,
              ],
            }}
          />
        </Source>
      ) : null}

      {routeSegments.length ? (
        <Source id="mission-routes-source" type="geojson" data={routeGeojson}>
          <Layer
            id="mission-routes-line"
            type="line"
            paint={{
              "line-color": "#ffd166",
              "line-opacity": 0.96,
              "line-width": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                3,
                2,
                4.5,
                4,
                6.5,
              ],
              "line-blur": 0.35,
            }}
            layout={{
              "line-cap": "round",
              "line-join": "round",
            }}
          />
          <Layer
            id="mission-routes-glow"
            type="line"
            paint={{
              "line-color": "#fff2b5",
              "line-opacity": 0.45,
              "line-width": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                5,
                2,
                8,
                4,
                11,
              ],
              "line-blur": 1.2,
            }}
            layout={{
              "line-cap": "round",
              "line-join": "round",
            }}
          />
        </Source>
      ) : null}

      {routeSegments.length ? (
        <Source id="route-endpoints-source" type="geojson" data={routeEndpointGeojson}>
          <Layer
            id="route-endpoints-circle"
            type="circle"
            paint={{
              "circle-color": [
                "case",
                ["==", ["get", "point_type"], "port_target"],
                "#71d5ff",
                "#ffc43d",
              ],
              "circle-opacity": 0.96,
              "circle-stroke-color": "#f8fbff",
              "circle-stroke-width": 1.8,
              "circle-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                4,
                2,
                6,
                4,
                8,
              ],
            }}
          />
        </Source>
      ) : null}

      {co2Hotspots.length ? (
        <Source id="co2-hotspots-source" type="geojson" data={hotspotGeojson}>
          <Layer
            id="co2-hotspots-circle"
            type="circle"
            paint={{
              "circle-color": [
                "case",
                [">", ["get", "predicted_flux"], 0],
                "#ff855f",
                "#40dba8",
              ],
              "circle-opacity": 0.82,
              "circle-stroke-color": "#e8f7ff",
              "circle-stroke-width": 1.4,
              "circle-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                4,
                2,
                7,
                4,
                11,
              ],
            }}
          />
        </Source>
      ) : null}

      {co2Hotspots.length ? (
        <Source id="co2-hotspot-labels-source" type="geojson" data={hotspotLabelGeojson}>
          <Layer
            id="co2-hotspot-labels"
            type="symbol"
            layout={{
              "text-field": ["get", "label"],
              "text-size": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                9,
                3,
                11,
              ],
              "text-font": ["Open Sans Bold"],
              "text-offset": [0, 1.1],
              "text-anchor": "top",
              "text-allow-overlap": false,
            }}
            paint={{
              "text-color": "#f6fbff",
              "text-halo-color": "#10263b",
              "text-halo-width": 1.2,
            }}
          />
        </Source>
      ) : null}

      {anomalies.length ? (
        <Source id="anomalies-source" type="geojson" data={anomalyGeojson}>
          <Layer
            id="anomalies-circle"
            type="circle"
            paint={{
              "circle-color": "#ff5a5f",
              "circle-opacity": 0.88,
              "circle-stroke-color": "#ffdc8c",
              "circle-stroke-width": 1.5,
              "circle-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                6,
                2,
                10,
                4,
                14,
              ],
            }}
          />
        </Source>
      ) : null}

      {trashTargets.length ? (
        <Source id="trash-targets-source" type="geojson" data={trashGeojson}>
          <Layer
            id="trash-targets-circle"
            type="circle"
            paint={{
              "circle-color": [
                "case",
                [">", ["get", "observed"], 0],
                "#ffae42",
                "#ffc43d",
              ],
              "circle-opacity": 0.88,
              "circle-stroke-color": "#fff1c9",
              "circle-stroke-width": 1.8,
              "circle-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                8,
                2,
                14,
                4,
                18,
              ],
            }}
          />
        </Source>
      ) : null}

      {trashTargets.length ? (
        <Source id="trash-targets-labels-source" type="geojson" data={trashLabelGeojson}>
          <Layer
            id="trash-targets-labels"
            type="symbol"
            layout={{
              "text-field": ["get", "label"],
              "text-size": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                10,
                3,
                12,
              ],
              "text-font": ["Open Sans Bold"],
              "text-offset": [0, 1.15],
              "text-anchor": "top",
              "text-allow-overlap": true,
            }}
            paint={{
              "text-color": "#fff7e0",
              "text-halo-color": "#3f2c0c",
              "text-halo-width": 1.25,
            }}
          />
        </Source>
      ) : null}
    </MapLibreMap>
  );
}

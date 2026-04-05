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
  co2Zones = [],
  trashZones = [],
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
        const flux = feature.properties?.display_flux ?? feature.properties?.predicted_flux ?? feature.properties?.flux ?? 0;
        const [r, g, b] = fluxToColor(flux);
        const weakening = feature.properties?.weakening_score || 0;
        const routePriority = feature.properties?.route_priority || 0;
        const anomaly = feature.properties?.anomaly_score || 0;
        const signal = feature.properties?.display_signal ?? (weakening * 4.2 + anomaly * 3.4 + routePriority * 1.5);
        const intensity = Math.min(0.24, 0.02 + signal * 0.08 + Math.min(Math.abs(flux), 2.0) * 0.012);
        return cellPolygonFeature(
          `heat-cell-${index}`,
          lon,
          lat,
          {
            flux,
            signal,
            fill: `rgba(${r}, ${g}, ${b}, ${intensity.toFixed(3)})`,
            stroke: `rgba(${r}, ${g}, ${b}, ${(Math.min(0.4, intensity + 0.08)).toFixed(3)})`,
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
          route_mode: segment.routeMode || "port",
        },
      })),
    }),
    [routeSegments]
  );

  const co2ZoneGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: co2Zones.map((feature) => ({
        ...feature,
        properties: {
          ...feature.properties,
          fill_color:
            feature.properties.severity === "critical"
              ? "rgba(255, 59, 118, 0.32)"
              : feature.properties.severity === "high"
                ? "rgba(255, 105, 168, 0.24)"
                : feature.properties.severity === "elevated"
                  ? "rgba(255, 150, 200, 0.18)"
                  : "rgba(255, 196, 224, 0.13)",
          line_color:
            feature.properties.severity === "critical"
              ? "#ff4f93"
              : feature.properties.severity === "high"
                ? "#ff78b2"
                : feature.properties.severity === "elevated"
                  ? "#ff9fd0"
                  : "#ffc8e2",
        },
      })),
    }),
    [co2Zones]
  );

  const trashZoneGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: trashZones.map((feature) => ({
        ...feature,
        properties: {
          ...feature.properties,
          fill_color:
            feature.properties.severity === "critical"
              ? "rgba(255, 106, 48, 0.34)"
              : feature.properties.severity === "high"
                ? "rgba(255, 158, 55, 0.28)"
                : feature.properties.severity === "elevated"
                  ? "rgba(255, 201, 79, 0.21)"
                  : "rgba(255, 226, 130, 0.16)",
          line_color:
            feature.properties.severity === "critical"
              ? "#ff7a2f"
              : feature.properties.severity === "high"
                ? "#ffae44"
                : feature.properties.severity === "elevated"
                  ? "#ffd15e"
                  : "#ffe59c",
        },
      })),
    }),
    [trashZones]
  );

  const routeEndpointGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: routeSegments.flatMap((segment) => {
        const features = [pointFeature(`route-origin-${segment.id}`, segment.source[0], segment.source[1], {
          point_type: "trash_origin",
          label: segment.label,
          density: segment.density || 0,
        })];
        if (segment.routeMode === "transport") {
          features.push(
            pointFeature(`route-target-${segment.id}`, segment.target[0], segment.target[1], {
              point_type: "transport_target",
              label: segment.label,
              density: segment.density || 0,
            })
          );
        }
        return features;
      }),
    }),
    [routeSegments]
  );

  const hotspotGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: co2Hotspots.slice(0, 180).map((point, index) =>
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
      features: anomalies.slice(0, 40).map((anomaly) =>
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
      features: trashTargets.slice(0, 80).map((target) =>
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
      features: trashTargets.slice(0, 24).map((target) =>
        pointFeature(`trash-label-${target.id}`, target.lon, target.lat, {
          label: target.clusterSize > 1 ? `Trash x${target.clusterSize}` : "Trash hotspot",
        })
      ),
    }),
    [trashTargets]
  );

  const hotspotLabelGeojson = useMemo(
    () => ({
      type: "FeatureCollection",
      features: co2Hotspots.slice(0, 20).map((point, index) =>
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
    <MapLibreMap mapStyle={MAP_STYLES[basemapStyle] || MAP_STYLES.satellite} reuseMaps renderWorldCopies={false}>
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

      {co2Zones.length ? (
        <Source id="co2-zones-source" type="geojson" data={co2ZoneGeojson}>
          <Layer
            id="co2-zones-fill"
            type="fill"
            paint={{
              "fill-color": ["get", "fill_color"],
              "fill-opacity": 1,
            }}
          />
          <Layer
            id="co2-zones-outline"
            type="line"
            paint={{
              "line-color": ["get", "line_color"],
              "line-opacity": 0.95,
              "line-width": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                1.2,
                2,
                2.2,
                4,
                3.2,
              ],
            }}
          />
        </Source>
      ) : null}

      {trashZones.length ? (
        <Source id="trash-zones-source" type="geojson" data={trashZoneGeojson}>
          <Layer
            id="trash-zones-fill"
            type="fill"
            paint={{
              "fill-color": ["get", "fill_color"],
              "fill-opacity": 1,
            }}
          />
          <Layer
            id="trash-zones-outline"
            type="line"
            paint={{
              "line-color": ["get", "line_color"],
              "line-opacity": 0.98,
              "line-width": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                1.4,
                2,
                2.6,
                4,
                3.6,
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
              "line-color": [
                "case",
                ["==", ["get", "route_mode"], "transport"],
                "#ff9f43",
                "#ffd166",
              ],
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
              "line-color": [
                "case",
                ["==", ["get", "route_mode"], "transport"],
                "#ffe5b8",
                "#fff2b5",
              ],
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
                ["==", ["get", "point_type"], "transport_target"],
                "#ff9f43",
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
                2.5,
                2,
                4.5,
                4,
                6,
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
                "#ff6d4d",
                "#34e0c0",
              ],
              "circle-opacity": 0.92,
              "circle-stroke-color": "#dff7ff",
              "circle-stroke-width": 1.8,
              "circle-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                5,
                2,
                8,
                4,
                12,
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
                "#ff9800",
                "#ffd54a",
              ],
              "circle-opacity": 0.96,
              "circle-stroke-color": "#fff6cc",
              "circle-stroke-width": 2.8,
              "circle-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                8,
                2,
                12,
                4,
                16,
              ],
            }}
          />
          <Layer
            id="trash-targets-inner"
            type="circle"
            paint={{
              "circle-color": "#7a5300",
              "circle-opacity": 0.55,
              "circle-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                2.5,
                2,
                3.5,
                4,
                5,
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

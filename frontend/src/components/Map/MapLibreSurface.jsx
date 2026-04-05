import { useMemo } from "react";

import { Layer, Map as MapLibreMap, Source } from "react-map-gl/maplibre";

import { MAP_STYLES } from "../../utils/constants";

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

export default function MapLibreSurface({
  basemapStyle = "satellite",
  co2Hotspots = [],
  trashTargets = [],
  routeSegments = [],
  anomalies = [],
}) {
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
        })
      ),
    }),
    [trashTargets]
  );

  return (
    <MapLibreMap mapStyle={MAP_STYLES[basemapStyle] || MAP_STYLES.satellite} reuseMaps>
      {routeSegments.length ? (
        <Source id="mission-routes-source" type="geojson" data={routeGeojson}>
          <Layer
            id="mission-routes-line"
            type="line"
            paint={{
              "line-color": "#ffd166",
              "line-opacity": 0.9,
              "line-width": [
                "interpolate",
                ["linear"],
                ["zoom"],
                0,
                2,
                2,
                3,
                4,
                5,
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
                3,
                2,
                5,
                4,
                8,
              ],
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
    </MapLibreMap>
  );
}

function monthValue(dateString) {
  const month = Number((dateString || "").split("-")[1] || 1);
  return Number.isFinite(month) ? month : 1;
}

function regionMatches(region, lon) {
  if (region === "pacific") return lon >= 110 || lon <= -70;
  if (region === "atlantic") return lon >= -80 && lon <= 20;
  if (region === "indian") return lon >= 20 && lon <= 120;
  return true;
}

export function buildFallbackHeatmap(date = "", region = "global") {
  const month = monthValue(date);
  const features = [];
  const step = region === "global" ? 12 : 8;

  for (let lat = -64; lat <= 64; lat += step) {
    for (let lon = -180; lon <= 180; lon += step) {
      if (!regionMatches(region, lon)) continue;

      const seasonal = Math.sin((month / 12) * Math.PI * 2);
      const gyre = Math.cos((lon / 35) * (Math.PI / 2)) * 0.9;
      const latBand = -2.2 * Math.cos((lat / 85) * (Math.PI / 2)) + 0.85;
      const observed = latBand + gyre + seasonal * 0.22;
      const weakening =
        Math.max(0, 0.45 - Math.abs(lat - 28) / 80) * (lon < -120 && lon > -165 ? 1 : 0.22) +
        Math.max(0, 0.35 - Math.abs(lat + 22) / 70) * (lon > -25 && lon < 10 ? 1 : 0.18) +
        Math.max(0, 0.42 - Math.abs(lat - 15) / 60) * (lon > 55 && lon < 78 ? 1 : 0.14);
      const predicted = observed - weakening;
      const routePriority = Math.min(5, weakening * 3.4 + (Math.abs(lat) < 38 ? 0.9 : 0.35));

      features.push({
        type: "Feature",
        geometry: { type: "Point", coordinates: [lon, lat] },
        properties: {
          flux: Number(predicted.toFixed(3)),
          observed_flux: Number(observed.toFixed(3)),
          predicted_flux: Number(predicted.toFixed(3)),
          sst: Number((25 - Math.abs(lat) * 0.28 + Math.sin(lon / 45) * 1.1).toFixed(2)),
          anomaly_score: Number(Math.min(1, weakening * 1.7).toFixed(3)),
          weakening_score: Number(Math.min(1, weakening).toFixed(3)),
          route_priority: Number(routePriority.toFixed(3)),
          is_anomaly: weakening > 0.32,
        },
      });
    }
  }

  const meanFlux = features.reduce((sum, item) => sum + item.properties.flux, 0) / Math.max(features.length, 1);
  const sinkAreaPct =
    (features.filter((item) => item.properties.flux < 0).length / Math.max(features.length, 1)) * 100;

  return {
    type: "FeatureCollection",
    metadata: {
      date: date || new Date().toISOString().slice(0, 7),
      units: "mol CO2/m²/yr",
      mean_flux: Number(meanFlux.toFixed(3)),
      sink_area_pct: Number(sinkAreaPct.toFixed(1)),
      inference_mode: "frontend_demo_fallback",
      trained_model_ready: false,
    },
    features,
  };
}

export function buildFallbackAnomalies(date = "", region = "global") {
  const candidates = [
    {
      id: "north-pacific-weakening",
      lat: 33.5,
      lon: -147,
      region_name: "North Pacific Weakening Front",
      anomaly_score: 0.89,
      deviation_pct: -34.2,
      severity: "high",
    },
    {
      id: "south-atlantic-plastics",
      lat: -24,
      lon: -9,
      region_name: "South Atlantic Drift",
      anomaly_score: 0.74,
      deviation_pct: -20.6,
      severity: "medium",
    },
    {
      id: "arabian-sea-routing",
      lat: 16.5,
      lon: 66.2,
      region_name: "Arabian Sea Route Stress",
      anomaly_score: 0.91,
      deviation_pct: -28.8,
      severity: "critical",
    },
  ];

  return candidates
    .filter((item) => region === "global" || regionMatches(region, item.lon))
    .map((item, index) => ({
      ...item,
      detected_at: `${date || new Date().toISOString().slice(0, 7)}-${String(10 + index).padStart(2, "0")}T06:00:00Z`,
    }));
}

export const HACKATHON_API_DEFAULTS = [
  {
    id: "era5",
    name: "ERA5",
    purpose: "Wind forcing for gas transfer velocity and routing conditions.",
    status: "connected",
    enabled: true,
    fields: ["10m wind", "surface pressure", "wave-relevant forcing"],
    url: "https://cds.climate.copernicus.eu",
    env_var: "ERA5_API_URL",
    notes: "~/OceanPulseData/ERA",
  },
  {
    id: "socat",
    name: "SOCAT",
    purpose: "Ship-based pCO2 observations for target construction and validation.",
    status: "connected",
    enabled: true,
    fields: ["time", "latitude", "longitude", "temp", "sal", "surface ocean fCO2"],
    url: "https://data.pmel.noaa.gov/socat/erddap/tabledap/socat_v2025_decimated.csv",
    env_var: "SOCAT_DATA_URL",
    notes: "ERDDAP live decimated dataset",
  },
  {
    id: "noaa_gml_co2",
    name: "NOAA GML CO2",
    purpose: "Atmospheric pCO2 baseline feature.",
    status: "connected",
    enabled: true,
    fields: ["monthly atmospheric CO2"],
    url: "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
    env_var: "NOAA_GML_CO2_URL",
    notes: "Hackathon default enabled",
  },
  {
    id: "noaa_hycom",
    name: "NOAA HYCOM",
    purpose: "Ocean current vectors for advection-aware routing features.",
    status: "planned",
    enabled: false,
    fields: ["water_u", "water_v", "surface currents"],
    url: "https://tds.hycom.org",
    env_var: "NOAA_HYCOM_URL",
    notes: "",
  },
  {
    id: "copernicus_marine",
    name: "Copernicus Marine",
    purpose: "Salinity and marine state variables for flux estimation.",
    status: "testing",
    enabled: true,
    fields: ["salinity", "surface currents", "biogeochemical grids"],
    url: "https://marine.copernicus.eu",
    env_var: "COPERNICUS_MARINE_URL",
    notes:
      "currents_dataset_id=cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m;" +
      "salinity_dataset_id=cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m;" +
      "temperature_dataset_id=cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m;" +
      "min_longitude=-160;max_longitude=-120;min_latitude=15;max_latitude=40;" +
      "min_depth=0;max_depth=1;" +
      "path=Training_Data/Copernicus",
  },
  {
    id: "nasa_ocean_color",
    name: "NASA MODIS / Ocean Color",
    purpose: "Chlorophyll-a and optical indicators tied to biological uptake.",
    status: "planned",
    enabled: false,
    fields: ["chlorophyll_a", "ocean color"],
    url: "https://oceancolor.gsfc.nasa.gov",
    env_var: "NASA_OCEANCOLOR_URL",
    notes: "",
  },
  {
    id: "global_fishing_watch",
    name: "Global Fishing Watch",
    purpose: "AIS vessel presence, port visits, and vessel identity for route feasibility and supervision.",
    status: "planned",
    enabled: false,
    fields: ["vessel presence", "port visits", "identity", "AIS gaps"],
    url: "https://globalfishingwatch.org/our-apis/documentation",
    env_var: "GLOBAL_FISHING_WATCH_URL",
    notes: "",
  },
  {
    id: "world_port_index",
    name: "NGA World Port Index",
    purpose: "Real port locations and metadata for route targets and nearest-port logic.",
    status: "planned",
    enabled: false,
    fields: ["port name", "country", "coordinates", "harbor metadata"],
    url: "https://vcps.nga.mil/nauticalpubs-feature/rest/services/WPI/World_Port_Index_Viewer/FeatureServer",
    env_var: "WORLD_PORT_INDEX_URL",
    notes: "",
  },
  {
    id: "emodnet_litter",
    name: "EMODnet Chemistry / Litter",
    purpose: "Marine litter observations for replacing modeled recovery targets with measured debris context.",
    status: "planned",
    enabled: false,
    fields: ["marine litter observations", "survey metadata"],
    url: "https://emodnet.ec.europa.eu/en/chemistry",
    env_var: "EMODNET_LITTER_URL",
    notes: "",
  },
  {
    id: "oceanscan",
    name: "OceanScan",
    purpose: "Additional marine debris and ocean monitoring products for trash accumulation evidence.",
    status: "planned",
    enabled: false,
    fields: ["debris observations", "ocean monitoring products"],
    url: "https://www.oceanscan.org",
    env_var: "OCEANSCAN_URL",
    notes: "",
  },
];

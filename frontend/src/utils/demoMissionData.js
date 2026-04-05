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
  return {
    type: "FeatureCollection",
    metadata: {
      date: date || new Date().toISOString().slice(0, 7),
      units: "mol CO2/m²/yr",
      mean_flux: null,
      sink_area_pct: null,
      inference_mode: "real_monthly_convlstm",
      verified_map: false,
      map_source: "real_grid_unavailable",
      source_summary: "No verified CO2 ocean layers are available until real gridded ingestion succeeds.",
      trained_model_ready: null,
    },
    features: [],
  };
}

export function buildFallbackAnomalies(date = "", region = "global") {
  return [];
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
      "min_longitude=-180;max_longitude=180;min_latitude=-80;max_latitude=80;" +
      "min_depth=0;max_depth=1;" +
      "path=Training_Data/Copernicus",
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

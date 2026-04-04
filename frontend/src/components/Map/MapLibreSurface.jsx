import { Map as MapLibreMap } from "react-map-gl/maplibre";

import { MAP_STYLES } from "../../utils/constants";

export default function MapLibreSurface({ basemapStyle = "satellite" }) {
  return <MapLibreMap mapStyle={MAP_STYLES[basemapStyle] || MAP_STYLES.satellite} reuseMaps />;
}

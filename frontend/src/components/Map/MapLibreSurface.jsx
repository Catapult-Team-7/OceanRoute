import { Map as MapLibreMap } from "react-map-gl/maplibre";

import { PUBLIC_MAP_STYLE } from "../../utils/constants";

export default function MapLibreSurface() {
  return <MapLibreMap mapStyle={PUBLIC_MAP_STYLE} reuseMaps />;
}

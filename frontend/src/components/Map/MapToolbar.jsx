import ColorLegend from "../common/ColorLegend";
import MapControls from "./MapControls";

export default function MapToolbar() {
  return (
    <div className="map-toolbar-panels">
      <MapControls />
      <ColorLegend />
    </div>
  );
}

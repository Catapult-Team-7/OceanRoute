import { useOceanStore } from "../../store/oceanStore";

const REGIONS = ["global", "pacific", "atlantic", "indian"];
const BASEMAPS = [
  { id: "satellite", label: "Satellite" },
  { id: "ocean", label: "Ocean" },
];

export default function MapControls() {
  const selectedRegion = useOceanStore((state) => state.selectedRegion);
  const basemapStyle = useOceanStore((state) => state.basemapStyle);
  const setSelectedRegion = useOceanStore((state) => state.setSelectedRegion);
  const setBasemapStyle = useOceanStore((state) => state.setBasemapStyle);
  const requestRefresh = useOceanStore((state) => state.requestRefresh);

  return (
    <div className="map-controls">
      <div className="map-controls-title">Region</div>
      {REGIONS.map((region) => (
        <button
          key={region}
          type="button"
          className={selectedRegion === region ? "active" : ""}
          onClick={() => setSelectedRegion(region)}
        >
          {region}
        </button>
      ))}
      <div className="map-controls-title">Basemap</div>
      {BASEMAPS.map((item) => (
        <button
          key={item.id}
          type="button"
          className={basemapStyle === item.id ? "active" : ""}
          onClick={() => setBasemapStyle(item.id)}
        >
          {item.label}
        </button>
      ))}
      <div className="map-controls-title">Data</div>
      <button type="button" onClick={() => requestRefresh()}>
        Refresh
      </button>
    </div>
  );
}

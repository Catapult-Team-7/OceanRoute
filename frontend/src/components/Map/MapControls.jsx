import { useOceanStore } from "../../store/oceanStore";

const REGIONS = ["global", "pacific", "atlantic", "indian"];

export default function MapControls() {
  const selectedRegion = useOceanStore((state) => state.selectedRegion);
  const setSelectedRegion = useOceanStore((state) => state.setSelectedRegion);

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
    </div>
  );
}

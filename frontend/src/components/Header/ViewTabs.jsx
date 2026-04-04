import { useOceanStore } from "../../store/oceanStore";

const VIEWS = [
  { id: "mission", label: "Mission Map" },
  { id: "ml", label: "ML Lab" },
];

export default function ViewTabs() {
  const currentView = useOceanStore((state) => state.currentView);
  const setCurrentView = useOceanStore((state) => state.setCurrentView);

  return (
    <div className="view-tabs">
      {VIEWS.map((view) => (
        <button
          key={view.id}
          type="button"
          className={currentView === view.id ? "active" : ""}
          onClick={() => setCurrentView(view.id)}
        >
          {view.label}
        </button>
      ))}
    </div>
  );
}

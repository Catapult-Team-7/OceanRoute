import type { DashboardView } from "../types";

interface AppHeaderProps {
  view: DashboardView;
  onChangeView: (view: DashboardView) => void;
  regionName: string;
  forecastStatus: string;
  sourceStatus: string;
}

const TABS: Array<{ id: DashboardView; label: string; hint: string }> = [
  { id: "mission", label: "Mission Map", hint: "Forecasts, routes, and trust" },
  { id: "ml", label: "ML Lab", hint: "Datasets, models, and promotion" },
  { id: "progress", label: "Progress", hint: "Impact ledger and mission log" },
];

export function AppHeader({ view, onChangeView, regionName, forecastStatus, sourceStatus }: AppHeaderProps) {
  return (
    <header className="demo-header">
      <div className="demo-brand">
        <span className="demo-kicker">SeaSweep Demo</span>
        <h1>Mission-grade coastal cleanup ops</h1>
        <p>Deck-powered regional mission map, ML model controls, and impact tracking on the live SeaSweep backend.</p>
      </div>
      <div className="demo-header-meta">
        <div className="demo-meta-card">
          <span>Area</span>
          <strong>{regionName}</strong>
        </div>
        <div className="demo-meta-card">
          <span>Forecast</span>
          <strong>{forecastStatus}</strong>
        </div>
        <div className="demo-meta-card">
          <span>Source</span>
          <strong>{sourceStatus}</strong>
        </div>
      </div>
      <nav className="demo-tabs" aria-label="Demo views">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`demo-tab ${view === tab.id ? "is-active" : ""}`}
            onClick={() => onChangeView(tab.id)}
            aria-pressed={view === tab.id}
          >
            <strong>{tab.label}</strong>
            <span>{tab.hint}</span>
          </button>
        ))}
      </nav>
    </header>
  );
}

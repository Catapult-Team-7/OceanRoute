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
  { id: "ml", label: "ML Lab", hint: "Models and datasets" },
  { id: "progress", label: "Mission Progress", hint: "Impact and mission log" },
];

export function AppHeader({ view, onChangeView, regionName, forecastStatus, sourceStatus }: AppHeaderProps) {
  return (
    <header className="demo-header">
      <div className="demo-brand">
        <span className="demo-kicker">SeaSweep</span>
        <h1>Cleanup Operations</h1>
        <p>Forecasts, routes, model oversight, and mission results for active cleanup work.</p>
      </div>
      <div className="header-block">
        <div className="header-block-label">Current Status</div>
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
      </div>
      <div className="header-block">
        <div className="header-block-label">Views</div>
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
      </div>
    </header>
  );
}

import { Suspense, lazy } from "react";

import Header from "./components/Header/Header";
import ViewTabs from "./components/Header/ViewTabs";
import MapStatusBar from "./components/Map/MapStatusBar";
import MapToolbar from "./components/Map/MapToolbar";
import Sidebar from "./components/Sidebar/Sidebar";
import TimeSlider from "./components/Timeline/TimeSlider";
import LoadingOverlay from "./components/common/LoadingOverlay";
import { useAnomalies } from "./hooks/useAnomalies";
import { useWebSocket } from "./hooks/useWebSocket";
import { useOceanStore } from "./store/oceanStore";

const OceanMap = lazy(() => import("./components/Map/OceanMap"));
const MLLab = lazy(() => import("./components/ML/MLLab"));
const ProgressDashboard = lazy(() => import("./components/ML/ProgressDashboard"));

export default function App() {
  useWebSocket();
  useAnomalies();

  const isLoading = useOceanStore((state) => state.isLoading);
  const currentView = useOceanStore((state) => state.currentView);

  return (
    <div className="app-shell">
      <Header />
      <ViewTabs />
      {currentView === "mission" ? (
        <>
          <MapStatusBar />
          <main className="main-layout">
            <section className="map-column">
              <div className="map-panel">
                <Suspense fallback={null}>
                  <OceanMap />
                </Suspense>
                {isLoading ? <LoadingOverlay /> : null}
              </div>
              <div className="map-toolbar">
                <MapToolbar />
                <TimeSlider />
              </div>
            </section>
            <Sidebar />
          </main>
        </>
      ) : currentView === "ml" ? (
        <main className="ml-page">
          <Suspense fallback={null}>
            <MLLab />
          </Suspense>
        </main>
      ) : (
        <main className="ml-page">
          <Suspense fallback={null}>
            <ProgressDashboard />
          </Suspense>
        </main>
      )}
    </div>
  );
}

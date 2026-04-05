import { create } from "zustand";

function monthString(date) {
  return date.toISOString().slice(0, 7);
}

export const useOceanStore = create((set) => ({
  currentView: "mission",
  selectedDate: monthString(new Date()),
  selectedPoint: null,
  selectedRegion: "global",
  basemapStyle: "satellite",
  heatmapData: null,
  trashData: null,
  anomalies: [],
  forecast: null,
  history: [],
  globalStats: { meanFlux: null, sinkCoverage: null, strongestSink: null },
  mlRuntimeStatus: null,
  isLoading: false,
  refreshNonce: 0,
  wsConnected: false,
  setCurrentView: (currentView) => set({ currentView }),
  setSelectedDate: (selectedDate) => set({ selectedDate }),
  setSelectedPoint: (selectedPoint) => set({ selectedPoint }),
  setSelectedRegion: (selectedRegion) => set({ selectedRegion }),
  setBasemapStyle: (basemapStyle) => set({ basemapStyle }),
  setTrashData: (trashData) => set({ trashData }),
  setHeatmapData: (heatmapData) =>
    set({
      heatmapData,
      globalStats: {
        meanFlux: heatmapData?.metadata?.mean_flux ?? null,
        sinkCoverage: heatmapData?.metadata?.sink_area_pct ?? null,
      },
    }),
  setAnomalies: (anomalies) => set({ anomalies }),
  setForecast: (forecast) => set({ forecast }),
  setHistory: (history) => set({ history }),
  setMlRuntimeStatus: (mlRuntimeStatus) => set({ mlRuntimeStatus }),
  setWsConnected: (wsConnected) => set({ wsConnected }),
  setLoading: (isLoading) => set({ isLoading }),
  requestRefresh: () => set((state) => ({ refreshNonce: state.refreshNonce + 1 })),
}));

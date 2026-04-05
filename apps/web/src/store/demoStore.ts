import { create } from "zustand";

import type { DashboardView, DebrisFilter, ForecastFilters, RouteFormState } from "../types";

interface DemoStore {
  view: DashboardView;
  selectedRegionId: string | null;
  filters: ForecastFilters;
  routeForm: RouteFormState;
  setView: (view: DashboardView) => void;
  setSelectedRegionId: (regionId: string | null) => void;
  updateFilters: (patch: Partial<ForecastFilters>) => void;
  updateRouteForm: (patch: Partial<RouteFormState>) => void;
  setRouteDepot: (lat: number, lon: number) => void;
}

const DEFAULT_FILTERS: ForecastFilters = {
  horizonHour: 24,
  debrisClass: "all",
  minConfidence: 0.25,
};

const DEFAULT_ROUTE_FORM: RouteFormState = {
  missionHours: 4,
  vesselSpeedKmh: 18,
  fuelBurnLph: 12,
  depotLat: 37.8066,
  depotLon: -122.4659,
};

export const useDemoStore = create<DemoStore>((set) => ({
  view: "mission",
  selectedRegionId: null,
  filters: DEFAULT_FILTERS,
  routeForm: DEFAULT_ROUTE_FORM,
  setView: (view) => set({ view }),
  setSelectedRegionId: (selectedRegionId) => set({ selectedRegionId }),
  updateFilters: (patch) =>
    set((state) => ({
      filters: {
        ...state.filters,
        ...patch,
        debrisClass: (patch.debrisClass as DebrisFilter | undefined) ?? state.filters.debrisClass,
      },
    })),
  updateRouteForm: (patch) =>
    set((state) => ({
      routeForm: {
        ...state.routeForm,
        ...patch,
      },
    })),
  setRouteDepot: (depotLat, depotLon) =>
    set((state) => ({
      routeForm: {
        ...state.routeForm,
        depotLat,
        depotLon,
      },
    })),
}));

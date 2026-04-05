import { useEffect } from "react";

import { API_BASE } from "../utils/constants";
import { useOceanStore } from "../store/oceanStore";
import { buildFallbackHeatmap } from "../utils/demoMissionData";
import { fetchJson } from "../utils/fetchJson";

export function useHeatmapData() {
  const currentView = useOceanStore((state) => state.currentView);
  const selectedDate = useOceanStore((state) => state.selectedDate);
  const selectedRegion = useOceanStore((state) => state.selectedRegion);
  const refreshNonce = useOceanStore((state) => state.refreshNonce);
  const setHeatmapData = useOceanStore((state) => state.setHeatmapData);
  const setLoading = useOceanStore((state) => state.setLoading);

  useEffect(() => {
    if (currentView !== "mission") return undefined;

    let cancelled = false;
    const controller = new AbortController();

    if (!useOceanStore.getState().heatmapData) {
      setHeatmapData(buildFallbackHeatmap(selectedDate, selectedRegion));
    }

    async function load(showSpinner = true) {
      if (showSpinner) setLoading(true);
      try {
        const params = new URLSearchParams({
          date: selectedDate,
          region: selectedRegion,
          resolution: selectedRegion === "global" ? "2deg" : "1deg",
        });
        const data = await fetchJson(`${API_BASE}/api/heatmap?${params.toString()}`, {
          signal: controller.signal,
          timeoutMs: 45000,
        });
        if (!cancelled) setHeatmapData(data);
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          console.error("Failed to fetch heatmap", error);
        }
      } finally {
        if (!cancelled && showSpinner) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [currentView, refreshNonce, selectedDate, selectedRegion, setHeatmapData, setLoading]);
}

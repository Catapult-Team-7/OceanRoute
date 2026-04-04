import { useEffect } from "react";

import { API_BASE } from "../utils/constants";
import { useOceanStore } from "../store/oceanStore";
import { fetchJson } from "../utils/fetchJson";

export function useHeatmapData() {
  const selectedDate = useOceanStore((state) => state.selectedDate);
  const selectedRegion = useOceanStore((state) => state.selectedRegion);
  const setHeatmapData = useOceanStore((state) => state.setHeatmapData);
  const setLoading = useOceanStore((state) => state.setLoading);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function load() {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          date: selectedDate,
          region: selectedRegion,
          resolution: selectedRegion === "global" ? "2deg" : "1deg",
        });
        const data = await fetchJson(`${API_BASE}/api/heatmap?${params.toString()}`, {
          signal: controller.signal,
          timeoutMs: 9000,
        });
        if (!cancelled) setHeatmapData(data);
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          console.error("Failed to fetch heatmap", error);
          setHeatmapData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selectedDate, selectedRegion, setHeatmapData, setLoading]);
}

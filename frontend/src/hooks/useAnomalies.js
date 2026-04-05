import { useEffect } from "react";

import { API_BASE } from "../utils/constants";
import { useOceanStore } from "../store/oceanStore";
import { fetchJson } from "../utils/fetchJson";

export function useAnomalies() {
  const currentView = useOceanStore((state) => state.currentView);
  const selectedDate = useOceanStore((state) => state.selectedDate);
  const selectedRegion = useOceanStore((state) => state.selectedRegion);
  const refreshNonce = useOceanStore((state) => state.refreshNonce);
  const setAnomalies = useOceanStore((state) => state.setAnomalies);

  useEffect(() => {
    if (currentView !== "mission") return undefined;

    let cancelled = false;
    const controller = new AbortController();

    async function load() {
      try {
        const params = new URLSearchParams({ date: selectedDate, threshold: "0.35", limit: "12" });
        const data = await fetchJson(`${API_BASE}/api/anomalies?${params.toString()}`, {
          signal: controller.signal,
          timeoutMs: 15000,
        });
        if (!cancelled) {
          setAnomalies(
            (data.anomalies || []).filter((anomaly) => Math.abs(anomaly.lat) <= 55)
          );
        }
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          console.error("Failed to fetch anomalies", error);
          setAnomalies([]);
        }
      }
    }
    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [currentView, refreshNonce, selectedDate, selectedRegion, setAnomalies]);
}

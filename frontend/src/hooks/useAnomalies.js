import { useEffect } from "react";

import { API_BASE } from "../utils/constants";
import { useOceanStore } from "../store/oceanStore";
import { fetchJson } from "../utils/fetchJson";

export function useAnomalies() {
  const selectedDate = useOceanStore((state) => state.selectedDate);
  const setAnomalies = useOceanStore((state) => state.setAnomalies);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    async function load() {
      try {
        const params = new URLSearchParams({ date: selectedDate, threshold: "0.7", limit: "10" });
        const data = await fetchJson(`${API_BASE}/api/anomalies?${params.toString()}`, {
          signal: controller.signal,
          timeoutMs: 7000,
        });
        if (!cancelled) setAnomalies(data.anomalies || []);
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          console.error("Failed to fetch anomalies", error);
          setAnomalies([]);
        }
      }
    }
    load();
    const pollTimer = window.setInterval(load, 20000);
    return () => {
      cancelled = true;
      window.clearInterval(pollTimer);
      controller.abort();
    };
  }, [selectedDate, setAnomalies]);
}

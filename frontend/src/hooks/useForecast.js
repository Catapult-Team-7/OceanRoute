import { useEffect } from "react";

import { API_BASE } from "../utils/constants";
import { useOceanStore } from "../store/oceanStore";
import { fetchJson } from "../utils/fetchJson";

export function useForecast() {
  const selectedPoint = useOceanStore((state) => state.selectedPoint);
  const setForecast = useOceanStore((state) => state.setForecast);
  const setHistory = useOceanStore((state) => state.setHistory);

  useEffect(() => {
    if (!selectedPoint) return;
    let cancelled = false;
    const forecastController = new AbortController();
    const historyController = new AbortController();

    async function load() {
      try {
        const forecastParams = new URLSearchParams({
          lat: String(selectedPoint.lat),
          lon: String(selectedPoint.lon),
          horizon: "72",
        });
        const historyParams = new URLSearchParams({
          lat: String(selectedPoint.lat),
          lon: String(selectedPoint.lon),
          months: "12",
        });
        const forecast = await fetchJson(`${API_BASE}/api/forecast?${forecastParams.toString()}`, {
          signal: forecastController.signal,
          timeoutMs: 30000,
        });
        const history = await fetchJson(`${API_BASE}/api/history?${historyParams.toString()}`, {
          signal: historyController.signal,
          timeoutMs: 30000,
        });
        if (!cancelled) {
          setForecast(forecast);
          setHistory(history.history || []);
        }
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          console.error("Failed to fetch forecast", error);
          setForecast(null);
          setHistory([]);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
      forecastController.abort();
      historyController.abort();
    };
  }, [selectedPoint, setForecast, setHistory]);
}

import { useEffect } from "react";

import { API_BASE } from "../utils/constants";
import { useOceanStore } from "../store/oceanStore";
import { fetchJson } from "../utils/fetchJson";

export function useMlStatus() {
  const currentView = useOceanStore((state) => state.currentView);
  const refreshNonce = useOceanStore((state) => state.refreshNonce);
  const setMlRuntimeStatus = useOceanStore((state) => state.setMlRuntimeStatus);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    let pollTimer = null;

    async function load() {
      try {
        const data = await fetchJson(`${API_BASE}/api/ml/status`, {
          signal: controller.signal,
          timeoutMs: 12000,
        });
        if (!cancelled) setMlRuntimeStatus(data);
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          console.error("Failed to fetch ML status", error);
        }
      } finally {
        if (!cancelled && currentView === "mission") {
          pollTimer = window.setTimeout(load, 20000);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
      if (pollTimer) window.clearTimeout(pollTimer);
      controller.abort();
    };
  }, [currentView, refreshNonce, setMlRuntimeStatus]);
}

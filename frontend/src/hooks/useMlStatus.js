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

    async function load() {
      try {
        const data = await fetchJson(`${API_BASE}/api/ml/status`, {
          signal: controller.signal,
          timeoutMs: 7000,
        });
        if (!cancelled) setMlRuntimeStatus(data);
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          console.error("Failed to fetch ML status", error);
        }
      }
    }

    load();
    const shouldPoll = currentView === "ml" || currentView === "progress";
    const pollTimer = shouldPoll ? window.setInterval(load, 10000) : null;
    return () => {
      cancelled = true;
      if (pollTimer) window.clearInterval(pollTimer);
      controller.abort();
    };
  }, [currentView, refreshNonce, setMlRuntimeStatus]);
}

import { useEffect } from "react";

import { API_BASE } from "../utils/constants";
import { useOceanStore } from "../store/oceanStore";
import { fetchJson } from "../utils/fetchJson";

export function useTrashData() {
  const currentView = useOceanStore((state) => state.currentView);
  const selectedRegion = useOceanStore((state) => state.selectedRegion);
  const refreshNonce = useOceanStore((state) => state.refreshNonce);
  const setTrashData = useOceanStore((state) => state.setTrashData);

  useEffect(() => {
    if (currentView !== "mission") return undefined;

    let cancelled = false;
    const controller = new AbortController();

    async function load() {
      try {
        const params = new URLSearchParams({
          region: selectedRegion,
          limit: "40",
        });
        const data = await fetchJson(`${API_BASE}/api/trash?${params.toString()}`, {
          signal: controller.signal,
          timeoutMs: 9000,
        });
        if (!cancelled) setTrashData(data);
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") {
          console.error("Failed to fetch trash data", error);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [currentView, refreshNonce, selectedRegion, setTrashData]);
}

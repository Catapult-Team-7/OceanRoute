export async function fetchJson(url, options = {}) {
  const { timeoutMs = 8000, ...rest } = options;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  const startedAt = performance.now();

  if (rest.signal) {
    rest.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    if (import.meta.env.DEV) {
      console.debug("[fetch:start]", url);
    }
    const response = await fetch(url, {
      ...rest,
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }
    const data = await response.json();
    if (import.meta.env.DEV) {
      console.debug("[fetch:ok]", {
        url,
        ms: Math.round(performance.now() - startedAt),
        keys: data && typeof data === "object" ? Object.keys(data) : [],
      });
    }
    return data;
  } catch (error) {
    if (import.meta.env.DEV) {
      console.error("[fetch:error]", {
        url,
        ms: Math.round(performance.now() - startedAt),
        message: error instanceof Error ? error.message : String(error),
      });
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

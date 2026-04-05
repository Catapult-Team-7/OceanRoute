const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export { API_BASE };

export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  const contentType = response.headers.get("content-type") ?? "";
  const isJsonResponse = contentType.includes("application/json");

  if (!response.ok) {
    if (isJsonResponse) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      throw new Error(payload?.detail ?? `Request failed with status ${response.status}`);
    }
    const text = (await response.text().catch(() => "")).trim();
    if (text.startsWith("<")) {
      throw new Error(`Expected JSON from ${path} but received HTML. Check that the frontend proxy is connected to the backend API.`);
    }
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  if (!isJsonResponse) {
    const text = (await response.text().catch(() => "")).trim();
    if (text.startsWith("<")) {
      throw new Error(`Expected JSON from ${path} but received HTML. Check that the frontend proxy is connected to the backend API.`);
    }
    throw new Error(`Expected JSON from ${path} but received ${contentType || "a non-JSON response"}.`);
  }

  return response.json() as Promise<T>;
}

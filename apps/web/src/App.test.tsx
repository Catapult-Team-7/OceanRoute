import type { ReactNode } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { App } from "./App";

vi.mock("react-leaflet", () => ({
  MapContainer: ({ children }: { children: ReactNode }) => <div data-testid="map">{children}</div>,
  TileLayer: () => null,
  CircleMarker: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  Popup: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  Polyline: () => <div data-testid="route-line" />,
}));

const baseForecast = {
  run_id: "run-1",
  generated_at: "2026-04-04T12:00:00Z",
  horizon_hours: 24,
  pilot_region: "sf_bay_estuary",
  region: {
    id: "sf_bay_estuary",
    name: "San Francisco Bay Estuary",
    description: "Harbor-estuary pilot",
    bbox: { lat_min: 37.45, lat_max: 38.25, lon_min: -123.05, lon_max: -121.75 },
    default_depot_lat: 37.8066,
    default_depot_lon: -122.4659,
    grid_dx_km: 1.2,
    grid_dy_km: 1.2,
    tags: ["pilot", "harbor", "estuary"],
  },
  source_mode_requested: "auto",
  source_mode_used: "sample",
  is_fallback: true,
  is_stale: false,
  age_minutes: 22,
  stale_after_minutes: 180,
  source_notes: ["Using deterministic SF Bay NOAA-style fixture data."],
  provenance: {
    generated_at: "2026-04-04T12:00:00Z",
    horizon_hours: 24,
    source_mode_requested: "auto",
    source_mode_used: "sample",
    is_fallback: true,
    is_stale: false,
    age_minutes: 22,
    stale_after_minutes: 180,
    source_notes: ["Using deterministic SF Bay NOAA-style fixture data."],
    baseline_engine: "custom_particle",
    baseline_artifact_uri: "C:/tmp/baseline.json",
    model_id: "model-1",
    model_architecture: "linear_residual",
    model_dataset_version: "v2",
    inference_service_version: "v1",
    prediction_artifact_uri: "C:/tmp/prediction.json",
  },
  debris_classes: [
    { code: "low", label: "Low-windage fragments", description: "", windage_factor: 0.05 },
    { code: "high", label: "High-windage floatables", description: "", windage_factor: 0.14 },
  ],
  steps: [
    {
      valid_at: "2026-04-05T12:00:00Z",
      horizon_hour: 24,
      cell_id: "north_bay",
      lat: 38.025,
      lon: -122.395,
      debris_class: "low",
      probability: 0.82,
      expected_kg_min: 4.2,
      expected_kg_max: 7.8,
      uncertainty: 0.21,
      confidence: 0.79,
      beaching_risk: 0.4,
      baseline_density: 1.2,
      ensemble_spread: 0.11,
      beaching_fraction: 0.08,
      stokes_drift_u: 0.02,
      stokes_drift_v: -0.01,
      windage_fraction: 0.01,
      restricted: false,
    },
    {
      valid_at: "2026-04-05T12:00:00Z",
      horizon_hour: 24,
      cell_id: "port_exit",
      lat: 37.79,
      lon: -122.31,
      debris_class: "high",
      probability: 0.66,
      expected_kg_min: 6.5,
      expected_kg_max: 11.2,
      uncertainty: 0.45,
      confidence: 0.55,
      beaching_risk: 0.52,
      baseline_density: 1.9,
      ensemble_spread: 0.22,
      beaching_fraction: 0.13,
      stokes_drift_u: 0.01,
      stokes_drift_v: -0.02,
      windage_fraction: 0.025,
      restricted: false,
    },
  ],
  top_hotspots: [
    {
      valid_at: "2026-04-05T12:00:00Z",
      horizon_hour: 24,
      cell_id: "north_bay",
      lat: 38.025,
      lon: -122.395,
      debris_class: "low",
      probability: 0.82,
      expected_kg_min: 4.2,
      expected_kg_max: 7.8,
      confidence: 0.79,
    },
    {
      valid_at: "2026-04-05T12:00:00Z",
      horizon_hour: 24,
      cell_id: "port_exit",
      lat: 37.79,
      lon: -122.31,
      debris_class: "high",
      probability: 0.66,
      expected_kg_min: 6.5,
      expected_kg_max: 11.2,
      confidence: 0.55,
    },
  ],
  summary: { step_count: 2, top_expected_kg_max: 11.2, mean_confidence: 0.67, source_mode_used: "sample" },
};

const lowOnlyForecast = {
  ...baseForecast,
  steps: [baseForecast.steps[0]],
  top_hotspots: [baseForecast.top_hotspots[0]],
  summary: { step_count: 1, top_expected_kg_max: 7.8, mean_confidence: 0.79, source_mode_used: "sample" },
};

const routeResponse = {
  mission_id: "mission-1",
  created_at: "2026-04-04T12:05:00Z",
  region_id: "sf_bay_estuary",
  vessel_id: "sf-bay-pilot-vessel",
  recommended_mode: "collection",
  target_horizon_hour: 24,
  ordered_cell_ids: ["north_bay:low"],
  expected_kg_min: 4.2,
  expected_kg_max: 7.8,
  expected_distance_km: 8.6,
  expected_duration_min: 92,
  estimated_fuel_liters: 18.4,
  objective_score: 4.4,
  uncertainty_risk: 0.21,
  alternates: ["port_exit:high"],
  forecast_run_id: "run-1",
  forecast_provenance: {
    generated_at: "2026-04-04T12:00:00Z",
    horizon_hours: 24,
    source_mode_requested: "auto",
    source_mode_used: "sample",
    is_fallback: true,
    is_stale: false,
    age_minutes: 22,
    stale_after_minutes: 180,
    source_notes: ["Using deterministic SF Bay NOAA-style fixture data."],
    baseline_engine: "custom_particle",
    model_id: "model-1",
    model_architecture: "linear_residual",
    model_dataset_version: "v2",
  },
  forecast_is_stale: false,
  legs: [
    { from_cell: "depot", to_cell: "north_bay:low", distance_km: 4.3, travel_minutes: 18.5 },
    { from_cell: "north_bay:low", to_cell: "depot", distance_km: 4.3, travel_minutes: 18.5 },
  ],
  metadata: { solver: "ortools" },
};

function mockResponse(payload: unknown) {
  return Promise.resolve({
    ok: true,
    json: async () => payload,
    headers: new Headers({ "Content-Type": "application/json" }),
  });
}

function installFetchMock() {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/api/health")) {
      return mockResponse({
        status: "ok",
        pilot_region: "sf_bay_estuary",
        default_horizon_hours: 24,
        scheduler_enabled: false,
        database_url: "sqlite",
        ingest_mode: "auto",
        forecast_stale_after_minutes: 180,
        default_depot_lat: 37.8066,
        default_depot_lon: -122.4659,
      });
    }
    if (url.includes("/api/impact/dashboard")) {
      return mockResponse({
        total_missions: 2,
        total_collected_kg: 36,
        total_distance_km: 14,
        total_hours: 5,
        kg_per_vessel_km: 2.57,
        kg_per_hour: 7.2,
        hotspot_precision: 0.75,
        false_search_distance_km: 1.4,
        mission_hit_rate: 1,
        latest_updated_at: "2026-04-04T12:00:00Z",
      });
    }
    if (url.includes("/api/forecast/latest")) {
      return mockResponse(url.includes("class=low") ? lowOnlyForecast : baseForecast);
    }
    if (url.includes("/api/forecast/run")) {
      return mockResponse({ run_id: "run-2" });
    }
    if (url.includes("/api/route/optimize")) {
      return mockResponse(routeResponse);
    }
    if (url.includes("/api/observations/upload")) {
      return mockResponse({ found_status: "found" });
    }
    if (url.includes("/api/cleanup/log")) {
      return mockResponse({ mission_id: "mission-1" });
    }
    return mockResponse({});
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("filters hotspot rankings by debris class", async () => {
    installFetchMock();
    render(<App />);

    expect(await screen.findByText(/Baseline custom particle/i)).not.toBeNull();
    expect(screen.getByText(/Model linear_residual v2/i)).not.toBeNull();
    expect((await screen.findAllByText(/north bay/i)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(/port exit/i)).length).toBeGreaterThan(0);

    fireEvent.change(screen.getByLabelText(/Debris class filter/i), { target: { value: "low" } });

    await waitFor(() => {
      expect(screen.queryByText(/port exit/i)).toBeNull();
    });
    expect(screen.getAllByText(/north bay/i).length).toBeGreaterThan(0);
  });

  it("runs route planning and renders the ordered cells", async () => {
    const fetchMock = installFetchMock();
    render(<App />);

    await screen.findAllByText(/north bay/i);
    fireEvent.click(await screen.findByRole("button", { name: /Optimize route/i }));

    await screen.findByText("north_bay:low");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/route/optimize"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("submits mission feedback through the observation and cleanup endpoints", async () => {
    const fetchMock = installFetchMock();
    render(<App />);

    await screen.findAllByText(/north bay/i);
    fireEvent.click(await screen.findByRole("button", { name: /Optimize route/i }));
    await screen.findByText("north_bay:low");

    fireEvent.change(screen.getByLabelText(/Collected kilograms/i), { target: { value: "9" } });
    fireEvent.change(screen.getByLabelText(/Crew notes/i), { target: { value: "Collected a foam cluster near the marker." } });
    fireEvent.submit(screen.getByRole("button", { name: /Save mission feedback/i }).closest("form") as HTMLFormElement);

    await screen.findByText(/Mission feedback saved/i);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/observations/upload"),
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/cleanup/log"),
      expect.objectContaining({ method: "POST" }),
    );
  });
});

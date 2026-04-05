import type { ReactNode } from "react";

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

vi.mock("@deck.gl/react", () => ({
  default: ({ children }: { children: ReactNode }) => <div data-testid="deck-map">{children}</div>,
}));

vi.mock("@deck.gl/layers", () => ({
  ScatterplotLayer: class ScatterplotLayer {},
  PathLayer: class PathLayer {},
  TextLayer: class TextLayer {},
}));

vi.mock("@deck.gl/extensions", () => ({
  PathStyleExtension: class PathStyleExtension {},
}));

vi.mock("react-map-gl/maplibre", () => ({
  default: () => <div data-testid="map-surface" />,
}));

vi.mock("maplibre-gl", () => ({
  default: {},
}));

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  AreaChart: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  CartesianGrid: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Area: () => null,
}));

const forecastResponse = {
  run_id: "run-1",
  generated_at: "2026-04-05T06:00:00Z",
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
  source_mode_used: "hybrid",
  is_fallback: false,
  is_stale: false,
  age_minutes: 14,
  stale_after_minutes: 180,
  source_notes: ["h1-h18 live, h19-h24 sample fill"],
  debris_classes: [
    { code: "low", label: "Low-windage", description: "", windage_factor: 0.01 },
    { code: "high", label: "High-windage", description: "", windage_factor: 0.025 },
  ],
  steps: [
    {
      valid_at: "2026-04-06T06:00:00Z",
      horizon_hour: 24,
      cell_id: "san_leandro",
      lat: 37.72,
      lon: -122.18,
      debris_class: "high",
      probability: 0.82,
      expected_kg_min: 3,
      expected_kg_max: 11.5,
      uncertainty: 0.22,
      confidence: 0.79,
      beaching_risk: 0.38,
      baseline_density: 1.2,
      ensemble_spread: 0.11,
      beaching_fraction: 0.08,
      stokes_drift_u: 0.02,
      stokes_drift_v: -0.01,
      windage_fraction: 0.01,
      restricted: false,
    },
  ],
  top_hotspots: [
    {
      valid_at: "2026-04-06T06:00:00Z",
      horizon_hour: 24,
      cell_id: "san_leandro",
      lat: 37.72,
      lon: -122.18,
      debris_class: "high",
      probability: 0.82,
      expected_kg_min: 3,
      expected_kg_max: 11.5,
      confidence: 0.79,
    },
  ],
  provenance: {
    generated_at: "2026-04-05T06:00:00Z",
    horizon_hours: 24,
    source_mode_requested: "auto",
    source_mode_used: "hybrid",
    is_fallback: false,
    is_stale: false,
    age_minutes: 14,
    stale_after_minutes: 180,
    source_notes: ["h1-h18 live, h19-h24 sample fill"],
    baseline_engine: "custom_particle",
    model_id: "model-1",
    model_architecture: "convlstm",
    model_dataset_version: "v2",
    model_stage: "champion",
    training_scope: "shared",
    used_candidate_override: false,
    used_inference_fallback: false,
    model_fallback_reason: null,
  },
  summary: { mean_confidence: 0.79 },
};

const routeResponse = {
  mission_id: "mission-1",
  created_at: "2026-04-05T06:10:00Z",
  region_id: "sf_bay_estuary",
  vessel_id: "vessel-1",
  recommended_mode: "collection",
  target_horizon_hour: 24,
  ordered_cell_ids: ["san_leandro:high"],
  expected_kg_min: 3,
  expected_kg_max: 11.5,
  expected_distance_km: 8.3,
  expected_duration_min: 92,
  estimated_fuel_liters: 18.4,
  objective_score: 4.2,
  uncertainty_risk: 0.22,
  alternates: ["south_bay:high"],
  legs: [],
  forecast_run_id: "run-1",
  forecast_provenance: forecastResponse.provenance,
  forecast_is_stale: false,
  metadata: { reason: "Yield clears threshold." },
};

function mockResponse(payload: unknown) {
  return Promise.resolve({
    ok: true,
    headers: new Headers({ "Content-Type": "application/json" }),
    json: async () => payload,
  });
}

function installFetchMock() {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/health")) {
      return mockResponse({
        status: "ok",
        pilot_region: "sf_bay_estuary",
        default_horizon_hours: 24,
        scheduler_enabled: false,
        database_url: "postgresql",
        ingest_mode: "auto",
        forecast_stale_after_minutes: 180,
        default_depot_lat: 37.8066,
        default_depot_lon: -122.4659,
      });
    }
    if (url.includes("/api/regions")) {
      return mockResponse([forecastResponse.region]);
    }
    if (url.includes("/api/impact/dashboard")) {
      return mockResponse({
        total_missions: 3,
        total_collected_kg: 48,
        total_distance_km: 16,
        total_hours: 5.5,
        kg_per_vessel_km: 3,
        kg_per_hour: 8.7,
        hotspot_precision: 1,
        false_search_distance_km: 1.1,
        mission_hit_rate: 1,
        latest_updated_at: "2026-04-05T06:10:00Z",
      });
    }
    if (url.includes("/api/missions")) {
      return mockResponse([
        {
          mission_id: "mission-1",
          date: "2026-04-05T06:10:00Z",
          collected_kg: 7.5,
          distance_km: 8.3,
          hours: 1.5,
          mode: "collection",
          notes: "Collected shoreline cluster.",
        },
      ]);
    }
    if (url.includes("/api/ml/datasets")) {
      return mockResponse([
        {
          dataset_id: "dataset-1",
          region_id: "__shared__",
          created_at: "2026-04-05T05:00:00Z",
          dataset_version: "v2",
          label_type: "expected_kg",
          sample_count: 466,
          feature_count: 12,
          feature_names: ["current_u"],
          artifact_path: "C:/tmp/dataset",
          split_counts: { train: 326, val: 70, test: 70 },
          schema_versions: {},
          region_ids: ["sf_bay_estuary", "puget_sound", "long_island_sound"],
          horizons: [24, 48, 72],
          input_channels: ["current_u"],
          target_channels: ["expected_kg"],
          tensor_shapes: {},
          metadata: {},
        },
      ]);
    }
    if (url.includes("/api/ml/models")) {
      if (url.endsWith("/evaluate")) {
        return mockResponse({
          model_id: "model-1",
          region_id: "__shared__",
          stage: "champion",
          artifact_path: "C:/tmp/model",
          metrics: { precision_at_10: 1, route_uplift_pct: 131.4 },
          evaluation_path: "C:/tmp/eval",
        });
      }
      return mockResponse([
        {
          model_id: "model-1",
          region_id: "__shared__",
          created_at: "2026-04-05T05:29:13.689986Z",
          architecture: "convlstm",
          status: "completed",
          stage: "champion",
          is_active: true,
          training_scope: "shared",
          artifact_path: "C:/tmp/model",
          dataset_id: "dataset-1",
          dataset_version: "v2",
          trained_regions: ["sf_bay_estuary", "puget_sound", "long_island_sound"],
          compatible_regions: ["sf_bay_estuary", "puget_sound", "long_island_sound"],
          horizons: [24, 48, 72],
          input_channels: ["current_u"],
          output_heads: ["hotspot_probability"],
          metrics: { precision_at_10: 1, route_uplift_pct: 131.4 },
        },
      ]);
    }
    if (url.includes("/api/forecast/latest")) {
      return mockResponse(forecastResponse);
    }
    if (url.includes("/api/forecast/run")) {
      return mockResponse({ run_id: "run-2" });
    }
    if (url.includes("/api/route/optimize")) {
      return mockResponse(routeResponse);
    }
    if (url.includes("/api/observations/upload")) {
      return mockResponse({ mission_id: "mission-1" });
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

  it("renders the demo shell and loads the mission map view", async () => {
    installFetchMock();
    render(<App />);

    expect(await screen.findByText(/Cleanup Operations/i)).not.toBeNull();
    expect((await screen.findAllByText(/San Francisco Bay Estuary/i)).length).toBeGreaterThan(0);
    expect(await screen.findByText(/h1-h18 live, h19-h24 sample fill/i)).not.toBeNull();
    expect(screen.getByTestId("deck-map")).not.toBeNull();
  });

  it("sends operator-tuned weights when optimizing a route", async () => {
    const fetchMock = installFetchMock();
    render(<App />);

    await screen.findByText(/san leandro/i);
    fireEvent.click(await screen.findByRole("button", { name: /Optimize route/i }));

    await screen.findByText(/Collection route created with 1 stop/i);
    const optimizeCall = fetchMock.mock.calls.find(([url]) => String(url).includes("/api/route/optimize"));
    expect(optimizeCall?.[1]).toEqual(
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining("\"distance_weight\":0.03"),
      }),
    );
  });

  it("switches to the ML Lab and evaluates the champion model", async () => {
    installFetchMock();
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /ML Lab/i }));
    expect(await screen.findByText(/Train and manage models/i)).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Evaluate/i }));
    await waitFor(() => {
      expect(screen.getByText(/route_uplift_pct/i)).not.toBeNull();
    });
    expect(screen.queryByText(/precision_at_10/i)).toBeNull();
    expect(screen.queryByText(/recall_at_10/i)).toBeNull();
    expect(screen.getByText(/Loaded evaluation metrics for model/i)).not.toBeNull();
  });
});

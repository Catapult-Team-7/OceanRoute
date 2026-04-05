import type {
  ForecastProvenance,
  ForecastSnapshot,
  ForecastStep,
  RegionInfo,
  RouteFormState,
  RoutePlan,
} from "../types";

export function formatRange(min: number, max: number): string {
  return `${min.toFixed(1)}-${max.toFixed(1)} kg`;
}

export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

export function formatFreshness(ageMinutes: number): string {
  if (ageMinutes < 60) {
    return `${ageMinutes}m old`;
  }
  const hours = Math.floor(ageMinutes / 60);
  const minutes = ageMinutes % 60;
  return `${hours}h ${minutes}m old`;
}

export function formatNumber(value: number, digits = 1): string {
  return new Intl.NumberFormat(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function sourceLabel(provenance: ForecastProvenance): string {
  const transition = `${provenance.source_mode_requested} -> ${provenance.source_mode_used}`;
  return provenance.is_fallback ? `${transition} fallback` : transition;
}

export function baselineLabel(provenance: ForecastProvenance): string {
  if (!provenance.baseline_engine) {
    return "Baseline pending";
  }
  return provenance.baseline_engine === "pygnome" ? "Baseline PyGNOME" : "Baseline custom particle";
}

export function modelDescriptor(provenance: ForecastProvenance): string | null {
  if (!provenance.model_architecture) {
    return null;
  }
  const scopeLabel =
    provenance.training_scope === "shared"
      ? "shared"
      : provenance.training_scope === "per_region"
        ? "regional"
        : null;
  const stageLabel = provenance.model_stage ? ` [${provenance.model_stage}]` : "";
  if (provenance.model_dataset_version) {
    return `${provenance.model_architecture} ${provenance.model_dataset_version}${scopeLabel ? ` (${scopeLabel})` : ""}${stageLabel}`;
  }
  return `${provenance.model_architecture}${scopeLabel ? ` (${scopeLabel})` : ""}${stageLabel}`;
}

export function modelLabel(provenance: ForecastProvenance): string {
  const descriptor = modelDescriptor(provenance);
  const overrideLabel = provenance.used_candidate_override ? " candidate override" : "";
  if (descriptor === null) {
    return "Baseline-only forecast";
  }
  if (provenance.used_inference_fallback) {
    return `Deep model fallback -> baseline (${descriptor})`;
  }
  return `Model ${descriptor}${overrideLabel}`;
}

export function forecastExecutionLabel(provenance: ForecastProvenance): string {
  const descriptor = modelDescriptor(provenance);
  if (provenance.used_inference_fallback && descriptor) {
    return `deep fallback (${descriptor})`;
  }
  if (descriptor) {
    return `model inference (${descriptor})`;
  }
  return "baseline forecast";
}

export function routeReason(route: RoutePlan | null): string | null {
  const reason = route?.metadata?.reason;
  return typeof reason === "string" ? reason : null;
}

export function confidenceBand(snapshot: ForecastSnapshot | null): string {
  if (!snapshot) {
    return "Unknown";
  }
  const meanConfidence = snapshot.summary.mean_confidence;
  if (typeof meanConfidence !== "number") {
    return "Unknown";
  }
  if (meanConfidence >= 0.75) {
    return "High";
  }
  if (meanConfidence >= 0.5) {
    return "Medium";
  }
  return "Recon";
}

export function topLine(snapshot: ForecastSnapshot | null): string {
  if (!snapshot || snapshot.top_hotspots.length === 0) {
    return "No active hotspot forecast";
  }
  const top = snapshot.top_hotspots[0];
  return `${top.cell_id.replaceAll("_", " ")} ${formatRange(top.expected_kg_min, top.expected_kg_max)}`;
}

export function buildStepLookup(forecast: ForecastSnapshot | null): Map<string, ForecastStep> {
  const stepByKey = new Map<string, ForecastStep>();
  forecast?.steps.forEach((step) => {
    stepByKey.set(`${step.cell_id}:${step.debris_class}`, step);
    if (!stepByKey.has(step.cell_id)) {
      stepByKey.set(step.cell_id, step);
    }
  });
  return stepByKey;
}

export function resolveRouteSteps(route: RoutePlan | null, forecast: ForecastSnapshot | null): ForecastStep[] {
  if (!route || !forecast) {
    return [];
  }
  const stepByKey = buildStepLookup(forecast);
  const cellIds = route.recommended_mode === "collection" ? route.ordered_cell_ids : route.alternates;
  return cellIds
    .map((cellId) => stepByKey.get(cellId) ?? stepByKey.get(cellId.split(":")[0]))
    .filter((step): step is ForecastStep => Boolean(step));
}

export function buildRouteCoordinates(
  route: RoutePlan | null,
  forecast: ForecastSnapshot | null,
  routeForm: RouteFormState,
): [number, number][] {
  const previewSteps = resolveRouteSteps(route, forecast);
  if (!previewSteps.length) {
    return [];
  }
  const coordinates: [number, number][] = [[routeForm.depotLon, routeForm.depotLat]];
  previewSteps.forEach((step) => {
    coordinates.push([step.lon, step.lat]);
  });
  if (route?.recommended_mode === "collection") {
    coordinates.push([routeForm.depotLon, routeForm.depotLat]);
  }
  return coordinates;
}

export function confidenceColor(step: ForecastStep): [number, number, number, number] {
  if (step.confidence >= 0.75) {
    return [255, 167, 38, 214];
  }
  if (step.confidence >= 0.5) {
    return [255, 210, 122, 196];
  }
  return [133, 162, 196, 176];
}

export function regionViewport(region: RegionInfo | null): { longitude: number; latitude: number; zoom: number } {
  if (!region) {
    return { latitude: 37.8066, longitude: -122.4659, zoom: 9.3 };
  }
  const latMin = region.bbox.lat_min ?? region.default_depot_lat - 0.3;
  const latMax = region.bbox.lat_max ?? region.default_depot_lat + 0.3;
  const lonMin = region.bbox.lon_min ?? region.default_depot_lon - 0.3;
  const lonMax = region.bbox.lon_max ?? region.default_depot_lon + 0.3;
  const latSpan = Math.max(0.01, latMax - latMin);
  const lonSpan = Math.max(0.01, lonMax - lonMin);
  const span = Math.max(latSpan, lonSpan);
  let zoom = 8.8;
  if (span < 0.35) {
    zoom = 10.5;
  } else if (span < 0.7) {
    zoom = 9.7;
  } else if (span < 1.3) {
    zoom = 8.9;
  } else if (span < 2.3) {
    zoom = 8.1;
  }
  return {
    latitude: (latMin + latMax) / 2,
    longitude: (lonMin + lonMax) / 2,
    zoom,
  };
}

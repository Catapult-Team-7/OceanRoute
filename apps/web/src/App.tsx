import { FormEvent, useEffect, useMemo, useState } from "react";

import { AppHeader } from "./components/AppHeader";
import { MissionPage } from "./components/MissionPage";
import { MlLabPage } from "./components/MlLabPage";
import { ProgressPage } from "./components/ProgressPage";
import { requestJson } from "./lib/api";
import { forecastExecutionLabel, formatTimestamp, sourceLabel } from "./lib/mission-utils";
import { useDemoStore } from "./store/demoStore";
import type {
  DatasetArtifact,
  FeedbackState,
  ForecastSnapshot,
  HealthStatus,
  ImpactDashboard,
  MissionRecord,
  MlTrainFormValues,
  ModelEvaluateResponse,
  ModelPromotionResponse,
  ModelRegistryEntry,
  ModelTrainResponse,
  RecommendedMode,
  RegionInfo,
  RoutePlan,
} from "./types";

const DEFAULT_FEEDBACK: FeedbackState = {
  foundStatus: "found",
  estimatedKg: 4,
  collectedKg: 4,
  photoUrl: "",
  note: "",
  routeDeviationReason: "",
};

export function App() {
  const view = useDemoStore((state) => state.view);
  const selectedRegionId = useDemoStore((state) => state.selectedRegionId);
  const filters = useDemoStore((state) => state.filters);
  const routeForm = useDemoStore((state) => state.routeForm);
  const setView = useDemoStore((state) => state.setView);
  const setSelectedRegionId = useDemoStore((state) => state.setSelectedRegionId);
  const updateFilters = useDemoStore((state) => state.updateFilters);
  const updateRouteForm = useDemoStore((state) => state.updateRouteForm);
  const setRouteDepot = useDemoStore((state) => state.setRouteDepot);

  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [regions, setRegions] = useState<RegionInfo[]>([]);
  const [forecast, setForecast] = useState<ForecastSnapshot | null>(null);
  const [forecastMissing, setForecastMissing] = useState(false);
  const [impact, setImpact] = useState<ImpactDashboard | null>(null);
  const [route, setRoute] = useState<RoutePlan | null>(null);
  const [missions, setMissions] = useState<MissionRecord[]>([]);
  const [datasets, setDatasets] = useState<DatasetArtifact[]>([]);
  const [models, setModels] = useState<ModelRegistryEntry[]>([]);
  const [evaluation, setEvaluation] = useState<ModelEvaluateResponse | null>(null);
  const [feedback, setFeedback] = useState<FeedbackState>(DEFAULT_FEEDBACK);
  const [loadingForecast, setLoadingForecast] = useState(true);
  const [loadingRoute, setLoadingRoute] = useState(false);
  const [busyMl, setBusyMl] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string>("");

  const selectedRegion = useMemo(
    () => regions.find((region) => region.id === selectedRegionId) ?? forecast?.region ?? null,
    [forecast?.region, regions, selectedRegionId],
  );

  useEffect(() => {
    void (async () => {
      try {
        const [status, availableRegions, dashboard, missionRows, datasetRows, modelRows] = await Promise.all([
          requestJson<HealthStatus>("/health"),
          requestJson<RegionInfo[]>("/regions"),
          requestJson<ImpactDashboard>("/impact/dashboard"),
          requestJson<MissionRecord[]>("/missions"),
          requestJson<DatasetArtifact[]>("/ml/datasets"),
          requestJson<ModelRegistryEntry[]>("/ml/models"),
        ]);
        setHealth(status);
        setRegions(availableRegions);
        setImpact(dashboard);
        setMissions(missionRows);
        setDatasets(datasetRows);
        setModels(modelRows);
        setSelectedRegionId(status.pilot_region);
        updateFilters({ horizonHour: status.default_horizon_hours });
        setRouteDepot(status.default_depot_lat, status.default_depot_lon);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Failed to load the SeaSweep demo shell.");
      }
    })();
  }, [setRouteDepot, setSelectedRegionId, updateFilters]);

  useEffect(() => {
    if (!selectedRegion) {
      return;
    }
    setRoute(null);
    setRouteDepot(selectedRegion.default_depot_lat, selectedRegion.default_depot_lon);
  }, [selectedRegion, setRouteDepot]);

  useEffect(() => {
    void loadForecast();
  }, [filters.debrisClass, filters.horizonHour, filters.minConfidence, selectedRegionId]);

  async function loadForecast() {
    if (!selectedRegionId) {
      return;
    }
    setLoadingForecast(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      params.set("region_id", selectedRegionId);
      params.set("horizon_hour", String(filters.horizonHour));
      params.set("min_confidence", String(filters.minConfidence));
      if (filters.debrisClass !== "all") {
        params.set("class", filters.debrisClass);
      }
      const snapshot = await requestJson<ForecastSnapshot>(`/forecast/latest?${params.toString()}`);
      setForecast(snapshot);
      setForecastMissing(false);
    } catch (caught) {
      if (caught instanceof Error && caught.message.includes("No forecast")) {
        setForecast(null);
        setForecastMissing(true);
      } else {
        setError(caught instanceof Error ? caught.message : "Failed to load forecast.");
      }
    } finally {
      setLoadingForecast(false);
    }
  }

  async function loadCatalog() {
    const [datasetRows, modelRows] = await Promise.all([
      requestJson<DatasetArtifact[]>("/ml/datasets"),
      requestJson<ModelRegistryEntry[]>("/ml/models"),
    ]);
    setDatasets(datasetRows);
    setModels(modelRows);
  }

  async function loadMissionsAndImpact() {
    const [missionRows, dashboard] = await Promise.all([
      requestJson<MissionRecord[]>("/missions"),
      requestJson<ImpactDashboard>("/impact/dashboard"),
    ]);
    setMissions(missionRows);
    setImpact(dashboard);
  }

  async function handleRunForecast() {
    if (!selectedRegionId) {
      return;
    }
    setError(null);
    setActionMessage("");
    setLoadingForecast(true);
    try {
      await requestJson("/forecast/run", {
        method: "POST",
        body: JSON.stringify({
          region_id: selectedRegionId,
          horizon_hours: 72,
          debris_classes: ["low", "high"],
          source_mode: "auto",
        }),
      });
      await loadForecast();
      setActionMessage(
        forecast
          ? `Forecast refreshed for ${forecast.region.name} at ${formatTimestamp(forecast.generated_at)} using ${forecastExecutionLabel(
              forecast.provenance,
            )}.`
          : `Forecast refreshed for ${selectedRegion?.name ?? selectedRegionId}.`,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to run forecast.");
    } finally {
      setLoadingForecast(false);
    }
  }

  async function handleOptimizeRoute() {
    if (!selectedRegionId) {
      return;
    }
    setLoadingRoute(true);
    setError(null);
    setActionMessage("");
    try {
      const nextRoute = await requestJson<RoutePlan>("/route/optimize", {
        method: "POST",
        body: JSON.stringify({
          region_id: selectedRegionId,
          depot_lat: routeForm.depotLat,
          depot_lon: routeForm.depotLon,
          mission_hours: routeForm.missionHours,
          vessel_speed_kmh: routeForm.vesselSpeedKmh,
          fuel_burn_lph: routeForm.fuelBurnLph,
          target_horizon_hour: filters.horizonHour,
          min_confidence: filters.minConfidence,
          min_objective_score: 0.0,
          objective_weights: {
            yield_weight: 1.0,
            distance_weight: 0.03,
            uncertainty_weight: 2.0,
            fuel_weight: 0.1,
          },
        }),
      });
      setRoute(nextRoute);
      setFeedback(DEFAULT_FEEDBACK);
      setActionMessage(
        nextRoute.recommended_mode === "collection"
          ? `Collection route created with ${nextRoute.ordered_cell_ids.length} stop(s).`
          : `Recon plan returned${nextRoute.metadata.reason ? `: ${String(nextRoute.metadata.reason)}` : "."}`,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to optimize route.");
    } finally {
      setLoadingRoute(false);
    }
  }

  async function handleSubmitFeedback(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!route) {
      return;
    }
    setSubmittingFeedback(true);
    setError(null);
    try {
      await requestJson("/observations/upload", {
        method: "POST",
        body: JSON.stringify({
          mission_id: route.mission_id,
          observed_at: new Date().toISOString(),
          lat: routeForm.depotLat,
          lon: routeForm.depotLon,
          found_status: feedback.foundStatus,
          debris_class: filters.debrisClass === "all" ? "low" : filters.debrisClass,
          estimated_kg: feedback.estimatedKg,
          confidence: 0.8,
          photo_url: feedback.photoUrl || null,
          note: feedback.note || null,
          route_deviation_reason: feedback.routeDeviationReason || null,
        }),
      });
      await requestJson("/cleanup/log", {
        method: "POST",
        body: JSON.stringify({
          mission_id: route.mission_id,
          completed_at: new Date().toISOString(),
          vessel_id: route.vessel_id,
          recommended_mode: route.recommended_mode as RecommendedMode,
          predicted_kg_min: route.expected_kg_min,
          predicted_kg_max: route.expected_kg_max,
          collected_kg: feedback.collectedKg,
          vessel_distance_km: route.expected_distance_km,
          vessel_hours: Math.max(route.expected_duration_min / 60, 0.5),
          hotspot_hits: feedback.foundStatus === "found" ? 1 : 0,
          hotspot_misses: feedback.foundStatus === "not_found" ? 1 : 0,
          false_search_km: feedback.foundStatus === "not_found" ? route.expected_distance_km : 0,
          fuel_liters: route.estimated_fuel_liters,
          route_deviation_reason: feedback.routeDeviationReason || null,
          notes: feedback.note || null,
        }),
      });
      await requestJson("/missions", {
        method: "POST",
        body: JSON.stringify({
          mission_id: route.mission_id,
          date: new Date().toISOString(),
          collected_kg: feedback.collectedKg,
          distance_km: route.expected_distance_km,
          hours: Math.max(route.expected_duration_min / 60, 0.5),
          mode: route.recommended_mode,
          notes: feedback.note || null,
        }),
      });
      await loadMissionsAndImpact();
      setActionMessage("Mission feedback saved to the impact ledger.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to submit mission feedback.");
    } finally {
      setSubmittingFeedback(false);
    }
  }

  async function handleTrainModel(values: MlTrainFormValues) {
    setBusyMl(true);
    setError(null);
    setActionMessage("");
    try {
      const dataset = datasets.find((entry) => entry.dataset_id === values.datasetId);
      const response = await requestJson<ModelTrainResponse>("/ml/train", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: values.datasetId,
          architecture: values.architecture,
          training_scope: values.trainingScope,
          activate: true,
          region_id: values.trainingScope === "per_region" ? selectedRegionId : null,
          region_ids: values.trainingScope === "shared" ? dataset?.region_ids ?? undefined : undefined,
          epochs: values.epochs,
          batch_size: values.batchSize,
          num_workers: values.numWorkers,
          device: "auto",
          promote_policy: "auto",
        }),
      });
      await loadCatalog();
      setEvaluation(null);
      setActionMessage(`Training submitted: ${response.model.architecture} ${response.model.dataset_version ?? ""}`.trim());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Training request failed.");
    } finally {
      setBusyMl(false);
    }
  }

  async function handleEvaluateModel(modelId: string) {
    setBusyMl(true);
    setError(null);
    try {
      const result = await requestJson<ModelEvaluateResponse>(`/ml/models/${modelId}/evaluate`);
      setEvaluation(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to evaluate model.");
    } finally {
      setBusyMl(false);
    }
  }

  async function handlePromoteModel(modelId: string) {
    setBusyMl(true);
    setError(null);
    try {
      const result = await requestJson<ModelPromotionResponse>(`/ml/models/${modelId}/promote`, {
        method: "POST",
      });
      await loadCatalog();
      setActionMessage(result.reason || `Promoted model ${modelId}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to promote model.");
    } finally {
      setBusyMl(false);
    }
  }

  const regionName = selectedRegion?.name ?? health?.pilot_region ?? "Loading region";
  const forecastStatus = forecast
    ? `${filters.horizonHour}h | ${forecast.top_hotspots.length} hotspot${forecast.top_hotspots.length === 1 ? "" : "s"}`
    : forecastMissing
      ? "No forecast yet"
      : "Forecast pending";
  const sourceStatus = forecast?.provenance ? sourceLabel(forecast.provenance) : health?.ingest_mode ?? "pending";

  return (
    <div className="demo-shell">
      <AppHeader
        view={view}
        onChangeView={setView}
        regionName={regionName}
        forecastStatus={forecastStatus}
        sourceStatus={sourceStatus}
      />

      {error ? <div className="banner error-banner">{error}</div> : null}
      {actionMessage ? <div className="banner success-banner">{actionMessage}</div> : null}

      {view === "mission" ? (
        <MissionPage
          regions={regions}
          selectedRegion={selectedRegion}
          selectedRegionId={selectedRegionId}
          onSelectRegion={setSelectedRegionId}
          filters={filters}
          onUpdateFilters={updateFilters}
          forecast={forecast}
          forecastMissing={forecastMissing}
          loadingForecast={loadingForecast}
          onRunForecast={handleRunForecast}
          route={route}
          loadingRoute={loadingRoute}
          routeForm={routeForm}
          onUpdateRouteForm={updateRouteForm}
          onOptimizeRoute={handleOptimizeRoute}
        />
      ) : view === "ml" ? (
        <MlLabPage
          selectedRegion={selectedRegion}
          datasets={datasets}
          models={models}
          evaluation={evaluation}
          busy={busyMl}
          onTrain={handleTrainModel}
          onEvaluate={handleEvaluateModel}
          onPromote={handlePromoteModel}
        />
      ) : (
        <ProgressPage
          impact={impact}
          missions={missions}
          forecast={forecast}
          route={route}
          feedback={feedback}
          onChangeFeedback={(patch) => setFeedback((current) => ({ ...current, ...patch }))}
          onSubmitFeedback={handleSubmitFeedback}
          submittingFeedback={submittingFeedback}
        />
      )}
    </div>
  );
}

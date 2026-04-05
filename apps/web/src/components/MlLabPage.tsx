import { useEffect, useMemo, useState } from "react";

import { formatNumber, formatTimestamp } from "../lib/mission-utils";
import type {
  DatasetArtifact,
  MlTrainFormValues,
  ModelEvaluateResponse,
  ModelRegistryEntry,
  RegionInfo,
} from "../types";

interface MlLabPageProps {
  selectedRegion: RegionInfo | null;
  datasets: DatasetArtifact[];
  models: ModelRegistryEntry[];
  evaluation: ModelEvaluateResponse | null;
  busy: boolean;
  onTrain: (values: MlTrainFormValues) => Promise<void>;
  onEvaluate: (modelId: string) => Promise<void>;
  onPromote: (modelId: string) => Promise<void>;
}

function formatMetric(metric: unknown): string {
  if (typeof metric === "number") {
    return formatNumber(metric, metric >= 100 ? 0 : 3);
  }
  if (Array.isArray(metric)) {
    return metric.join(", ");
  }
  if (metric && typeof metric === "object") {
    return JSON.stringify(metric);
  }
  return String(metric ?? "n/a");
}

export function MlLabPage({ selectedRegion, datasets, models, evaluation, busy, onTrain, onEvaluate, onPromote }: MlLabPageProps) {
  const recommendedDatasetId = datasets[0]?.dataset_id ?? "";
  const [form, setForm] = useState<MlTrainFormValues>({
    datasetId: recommendedDatasetId,
    architecture: "convlstm",
    trainingScope: "shared",
    epochs: 20,
    batchSize: 8,
    numWorkers: 4,
  });

  useEffect(() => {
    if (!form.datasetId && recommendedDatasetId) {
      setForm((current) => ({ ...current, datasetId: recommendedDatasetId }));
    }
  }, [form.datasetId, recommendedDatasetId]);

  const featuredModels = useMemo(
    () =>
      [...models].sort((left, right) => {
        if (left.stage === right.stage) {
          return right.created_at.localeCompare(left.created_at);
        }
        return left.stage === "champion" ? -1 : 1;
      }),
    [models],
  );

  return (
    <section className="three-column-page">
      <article className="panel-card">
        <div className="panel-header">
          <div>
            <p className="section-kicker">ML Lab</p>
            <h2>Train and promote demo models</h2>
            <p>Use SeaSweep dataset artifacts and existing `/api/ml/*` endpoints without importing Test’s old backend contracts.</p>
          </div>
          <span className="panel-badge">{featuredModels.length} models</span>
        </div>
        <form
          className="stack-form"
          onSubmit={async (event) => {
            event.preventDefault();
            await onTrain(form);
          }}
        >
          <label>
            Dataset
            <select value={form.datasetId} onChange={(event) => setForm((current) => ({ ...current, datasetId: event.target.value }))}>
              {datasets.map((dataset) => (
                <option key={dataset.dataset_id} value={dataset.dataset_id}>
                  {dataset.dataset_version} | {dataset.region_ids.join(", ")} | {dataset.sample_count} samples
                </option>
              ))}
            </select>
          </label>
          <label>
            Architecture
            <select
              value={form.architecture}
              onChange={(event) =>
                setForm((current) => ({ ...current, architecture: event.target.value as MlTrainFormValues["architecture"] }))
              }
            >
              <option value="convlstm">ConvLSTM</option>
              <option value="temporal_unet">Temporal U-Net</option>
              <option value="linear_residual">Linear residual</option>
            </select>
          </label>
          <label>
            Training scope
            <select
              value={form.trainingScope}
              onChange={(event) =>
                setForm((current) => ({ ...current, trainingScope: event.target.value as MlTrainFormValues["trainingScope"] }))
              }
            >
              <option value="shared">Shared</option>
              <option value="per_region">Per region</option>
            </select>
          </label>
          <div className="control-grid compact-grid">
            <label>
              Epochs
              <input
                type="number"
                min="1"
                max="50"
                value={form.epochs}
                onChange={(event) => setForm((current) => ({ ...current, epochs: Number(event.target.value) }))}
              />
            </label>
            <label>
              Batch size
              <input
                type="number"
                min="1"
                max="64"
                value={form.batchSize}
                onChange={(event) => setForm((current) => ({ ...current, batchSize: Number(event.target.value) }))}
              />
            </label>
            <label>
              Workers
              <input
                type="number"
                min="0"
                max="16"
                value={form.numWorkers}
                onChange={(event) => setForm((current) => ({ ...current, numWorkers: Number(event.target.value) }))}
              />
            </label>
          </div>
          <button className="primary-button" type="submit" disabled={busy || !form.datasetId}>
            {busy ? "Submitting..." : `Start ${form.architecture} training`}
          </button>
          <p className="info-note">
            Region context: {selectedRegion?.name ?? "shared demo scope"} | Training requests stay on the SeaSweep backend APIs.
          </p>
        </form>
      </article>

      <article className="panel-card">
        <div className="panel-header">
          <div>
            <h3>Dataset artifacts</h3>
            <p>Multi-region datasets from the current SeaSweep data lake.</p>
          </div>
          <span className="panel-badge">{datasets.length} datasets</span>
        </div>
        <div className="stack-list">
          {datasets.map((dataset) => (
            <article key={dataset.dataset_id} className="meta-card">
              <span>{dataset.dataset_version}</span>
              <strong>{dataset.region_ids.join(", ")}</strong>
              <p>{dataset.sample_count} samples | horizons {dataset.horizons.join(", ") || "n/a"}</p>
            </article>
          ))}
        </div>
      </article>

      <article className="panel-card">
        <div className="panel-header">
          <div>
            <h3>Model registry</h3>
            <p>Champion-first model catalog with evaluate and promote actions.</p>
          </div>
        </div>
        <div className="stack-list">
          {featuredModels.map((model) => (
            <article key={model.model_id} className="rank-card model-card">
              <div>
                <strong>
                  {model.architecture} {model.dataset_version ? `${model.dataset_version}` : ""}
                </strong>
                <p>
                  {model.stage} · {model.training_scope} · {formatTimestamp(model.created_at)}
                </p>
                <p>{model.compatible_regions.join(", ") || model.region_id}</p>
              </div>
              <div className="model-actions">
                <span className={`status-pill ${model.stage === "champion" ? "is-good" : ""}`}>{model.stage}</span>
                <button className="secondary-button" type="button" onClick={() => void onEvaluate(model.model_id)} disabled={busy}>
                  Evaluate
                </button>
                <button className="secondary-button" type="button" onClick={() => void onPromote(model.model_id)} disabled={busy}>
                  Promote
                </button>
              </div>
            </article>
          ))}
        </div>
        {evaluation ? (
          <div className="evaluation-panel">
            <h4>Latest evaluation</h4>
            <p>{evaluation.model_id}</p>
            <div className="stack-list compact-list">
              {Object.entries(evaluation.metrics).slice(0, 10).map(([key, value]) => (
                <article key={key} className="metric-row">
                  <span>{key}</span>
                  <strong>{formatMetric(value)}</strong>
                </article>
              ))}
            </div>
          </div>
        ) : null}
      </article>
    </section>
  );
}

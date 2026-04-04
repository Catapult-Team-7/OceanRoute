import { useEffect, useState } from "react";

import InfoHint from "../common/InfoHint";
import { API_BASE } from "../../utils/constants";
import { HACKATHON_API_DEFAULTS } from "../../utils/demoMissionData";

const DEFAULT_FORM = {
  epochs: 18,
  learning_rate: 0.05,
  month_window: 12,
  resolution: "2deg",
};

const MODEL_LIMITS = [
  {
    title: "Labels are still sparse",
    copy: "The ConvLSTM now trains on monthly aligned tensors, but target coverage still depends on where SOCAT observations exist.",
  },
  {
    title: "Biogeochemical drivers are incomplete",
    copy: "Currents, salinity, temperature, sea level, and atmospheric CO2 are supported, but chlorophyll and richer ecosystem drivers are still missing.",
  },
  {
    title: "Routing is partially forced",
    copy: "High-frequency Copernicus routing and local ERA5 wind enrichment are wired in, but routing quality improves a lot once global ERA5 grids and AIS constraints are added.",
  },
];

const NEXT_DATASETS = [
  "Global ERA5 grids or regional tiles for wind, pressure, radiation, and waves",
  "HYCOM or Copernicus current fields for advection and routing",
  "Copernicus salinity grids aligned to the same monthly mesh",
  "NASA Ocean Color chlorophyll-a fields for biological uptake",
  "Plastic/debris concentration observations for route supervision rather than hand-authored hotspots",
];

export default function MLLab() {
  const [status, setStatus] = useState(null);
  const [apiDrafts, setApiDrafts] = useState([]);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSavingApis, setIsSavingApis] = useState(false);
  const [previewById, setPreviewById] = useState({});
  const [previewingId, setPreviewingId] = useState("");
  const [syncingId, setSyncingId] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [backendHealth, setBackendHealth] = useState({ reachable: false, checked: false, detail: "" });
  const dataSummary = status?.data_summary || {};
  const connectorState = dataSummary.connector_state || {};
  const usingRealData = dataSummary.source === "real_observation_sample";

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
          throw new Error(`Health check returned ${response.status}`);
        }
        const data = await response.json();
        if (!cancelled) {
          setBackendHealth({
            reachable: data.status === "ok",
            checked: true,
            detail: data.status === "ok" ? `Backend reachable at ${API_BASE}` : `Unexpected health response from ${API_BASE}`,
          });
        }
      } catch (error) {
        if (!cancelled) {
          setBackendHealth({
            reachable: false,
            checked: true,
            detail: `Backend unreachable at ${API_BASE}. Start or restart the backend and try again.`,
          });
        }
      }
    }

    loadHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      try {
        const statusResponse = await fetch(`${API_BASE}/api/ml/status`);
        const statusData = await statusResponse.json();
        if (!cancelled) setStatus(statusData);
      } catch (error) {
        console.error("Failed to load ML status", error);
      }
    }

    loadStatus();
    const interval = window.setInterval(loadStatus, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadApis() {
      try {
        const response = await fetch(`${API_BASE}/api/ml/apis`);
        const data = await response.json();
        if (!cancelled) {
          setApiDrafts(data.apis?.length ? data.apis : HACKATHON_API_DEFAULTS);
        }
      } catch (error) {
        console.error("Failed to load API config", error);
        if (!cancelled) {
          setApiDrafts(HACKATHON_API_DEFAULTS);
        }
      }
    }

    loadApis();
    return () => {
      cancelled = true;
    };
  }, []);

  async function startTraining() {
    setIsSubmitting(true);
    try {
      if (!backendHealth.reachable) {
        throw new Error(`Backend unreachable at ${API_BASE}. Check that /health responds before training.`);
      }
      await saveApis();
      const response = await fetch(`${API_BASE}/api/ml/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(errorBody || `Training request failed with status ${response.status}`);
      }
      const data = await response.json();
      setStatus(data.training);
      setActionMessage(data.started ? "Training started." : "Training is already running.");
    } catch (error) {
      console.error("Failed to start training", error);
      setActionMessage(`Training failed to start: ${error.message}`);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function saveApis() {
    setIsSavingApis(true);
    try {
      const response = await fetch(`${API_BASE}/api/ml/apis`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(apiDrafts),
      });
      const data = await response.json();
      setApiDrafts(data.apis || []);
      setActionMessage("Connector settings saved.");
      return data.apis || [];
    } catch (error) {
      console.error("Failed to save API config", error);
      setActionMessage(`Failed to save connector settings: ${error.message}`);
      return null;
    } finally {
      setIsSavingApis(false);
    }
  }

  function enableHackathonSources() {
    setApiDrafts((current) =>
      (current.length ? current : HACKATHON_API_DEFAULTS).map((api) => {
        if (api.id === "socat" || api.id === "noaa_gml_co2" || api.id === "era5" || api.id === "copernicus_marine") {
          return {
            ...api,
            enabled: true,
            status: api.id === "copernicus_marine" ? "testing" : "connected",
            notes:
              api.id === "era5"
                ? "Training_Data/ERA"
                : api.id === "copernicus_marine"
                  ? "currents_dataset_id=cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m;" +
                    "salinity_dataset_id=cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m;" +
                    "temperature_dataset_id=cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m;" +
                    "path=Training_Data/Copernicus"
                  : api.notes,
          };
        }
        return api;
      })
    );
  }

  async function saveAndStartHackathonTraining() {
    enableHackathonSources();
    const nextDrafts = (apiDrafts.length ? apiDrafts : HACKATHON_API_DEFAULTS).map((api) => {
      if (api.id === "socat" || api.id === "noaa_gml_co2" || api.id === "era5" || api.id === "copernicus_marine") {
        return {
          ...api,
          enabled: true,
          status: api.id === "copernicus_marine" ? "testing" : "connected",
          notes:
            api.id === "era5"
              ? "Training_Data/ERA"
              : api.id === "copernicus_marine"
                ? "currents_dataset_id=cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m;" +
                  "salinity_dataset_id=cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m;" +
                  "temperature_dataset_id=cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m;" +
                  "path=Training_Data/Copernicus"
                : api.notes,
        };
      }
      return api;
    });
    setApiDrafts(nextDrafts);

    setIsSavingApis(true);
    try {
      await fetch(`${API_BASE}/api/ml/apis`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(nextDrafts),
      });
      setActionMessage("Hackathon defaults saved.");
    } catch (error) {
      console.error("Failed to save hackathon API config", error);
      setActionMessage(`Failed to save hackathon defaults: ${error.message}`);
    } finally {
      setIsSavingApis(false);
    }

    await startTraining();
  }

  async function previewConnector(connectorId) {
    setPreviewingId(connectorId);
    try {
      const response = await fetch(`${API_BASE}/api/connectors/${connectorId}/preview`, {
        method: "POST",
      });
      const data = await response.json();
      setPreviewById((current) => ({ ...current, [connectorId]: data }));
      setActionMessage(`${data.connector_name || connectorId} preview: ${data.status}`);
    } catch (error) {
      console.error("Failed to preview connector", error);
      setActionMessage(`Failed to preview ${connectorId}: ${error.message}`);
    } finally {
      setPreviewingId("");
    }
  }

  async function syncConnector(connectorId) {
    setSyncingId(connectorId);
    try {
      const response = await fetch(`${API_BASE}/api/connectors/${connectorId}/sync`, {
        method: "POST",
      });
      const data = await response.json();
      setPreviewById((current) => ({ ...current, [connectorId]: data }));
      setActionMessage(`${data.connector_name || connectorId} sync: ${data.status}`);
    } catch (error) {
      console.error("Failed to sync connector", error);
      setActionMessage(`Failed to sync ${connectorId}: ${error.message}`);
    } finally {
      setSyncingId("");
    }
  }

  return (
    <section className="ml-layout">
      <div className="ml-primary">
        <div className="ml-section">
          <p className="eyebrow">Training Workspace</p>
          <h2>OceanPulse ML Lab</h2>
          <p className="subtle">
            This lab trains only on real-source inputs. Right now that means SOCAT surface-ocean observations, NOAA
            GML monthly atmospheric CO2, your local ERA5 enrichment, and Copernicus monthly plus routing grids when
            they are synced locally.
          </p>
          <div className="connector-actions">
            <button type="button" className="primary-button" onClick={enableHackathonSources}>
              Enable Hackathon Defaults
            </button>
            <button type="button" className="secondary-button" onClick={saveAndStartHackathonTraining}>
              Save + Start Training
            </button>
          </div>
          {actionMessage ? <p className="subtle">{actionMessage}</p> : null}
          {backendHealth.checked ? (
            <p className={backendHealth.reachable ? "subtle" : "error-copy"}>{backendHealth.detail}</p>
          ) : null}
        </div>

        <div className="ml-section">
          <div className="section-header">
            <h3>Training Controls</h3>
            <button type="button" className="primary-button" onClick={startTraining} disabled={isSubmitting}>
              {status?.status === "running" ? "Training…" : "Start Training"}
            </button>
          </div>
          <div className="ml-form-grid">
            <label>
              Epochs
              <input
                type="number"
                value={form.epochs}
                min="5"
                max="200"
                onChange={(event) => setForm((current) => ({ ...current, epochs: Number(event.target.value) }))}
              />
            </label>
            <label>
              Learning rate
              <input
                type="number"
                step="0.01"
                value={form.learning_rate}
                onChange={(event) =>
                  setForm((current) => ({ ...current, learning_rate: Number(event.target.value) }))
                }
              />
            </label>
            <label>
              Month window
              <input
                type="number"
                value={form.month_window}
                min="3"
                max="24"
                onChange={(event) => setForm((current) => ({ ...current, month_window: Number(event.target.value) }))}
              />
            </label>
            <label>
              Resolution
              <select
                value={form.resolution}
                onChange={(event) => setForm((current) => ({ ...current, resolution: event.target.value }))}
              >
                <option value="2deg">2deg</option>
                <option value="1deg">1deg</option>
                <option value="0.5deg">0.5deg</option>
              </select>
            </label>
          </div>
        </div>

        <div className="ml-section">
          <div className="section-header">
            <h3>Training Status</h3>
            <span className={`status-pill ${status?.status || "idle"}`}>{status?.status || "idle"}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${(status?.progress || 0) * 100}%` }} />
          </div>
          {status?.metrics?.stage ? (
            <p className="subtle">
              Stage: <strong>{status.metrics.stage}</strong>
              {status?.metrics?.detail ? ` · ${status.metrics.detail}` : ""}
            </p>
          ) : null}
          <p className="subtle">
            Training source: <strong>{dataSummary.source || "unconfigured_real_pipeline"}</strong>
          </p>
          <div className="ml-metrics-grid">
            <article className="ml-metric-card">
              <span className="metric-label">
                Epoch
                <InfoHint label="Epoch" description="One full pass through the current training dataset." />
              </span>
              <strong>
                {status?.current_epoch || 0}/{status?.total_epochs || form.epochs}
              </strong>
            </article>
            <article className="ml-metric-card">
              <span className="metric-label">
                Validation loss
                <InfoHint
                  label="Validation loss"
                  description="Out-of-sample loss on held-back validation tensors. Lower is generally better."
                />
              </span>
              <strong>{status?.metrics?.val_loss ?? "--"}</strong>
            </article>
            <article className="ml-metric-card">
              <span className="metric-label">
                MAE
                <InfoHint
                  label="MAE"
                  description="Mean absolute error between predicted and target flux values on validation data."
                />
              </span>
              <strong>{status?.metrics?.mae ?? "--"}</strong>
            </article>
            <article className="ml-metric-card">
              <span className="metric-label">
                R²
                <InfoHint
                  label="R²"
                  description="Explained variance score on validation data. Closer to 1 means the model explains more of the target variance."
                />
              </span>
              <strong>{status?.metrics?.r2 ?? "--"}</strong>
            </article>
          </div>
          {typeof dataSummary.sample_count === "number" ? (
            <p className="subtle">
              Samples used: {dataSummary.sample_count}
              {dataSummary.latest_observation ? ` · Latest observation: ${dataSummary.latest_observation.slice(0, 10)}` : ""}
            </p>
          ) : null}
          {dataSummary.era5?.available ? (
            <p className="subtle">
              ERA5 enrichment: enabled from {dataSummary.era5.path}
              {typeof dataSummary.era5.months_loaded === "number" ? ` · ${dataSummary.era5.months_loaded} monthly records` : ""}
            </p>
          ) : null}
          {dataSummary.era5?.error ? <p className="subtle">ERA5 enrichment unavailable: {dataSummary.era5.error}</p> : null}
          {dataSummary.fallback_reason ? <p className="error-copy">Fallback reason: {dataSummary.fallback_reason}</p> : null}
          <div className="loss-strip">
            {(status?.loss_history || []).map((loss, index) => (
              <div key={`${loss}-${index}`} className="loss-bar" style={{ height: `${18 + loss * 26}px` }} />
            ))}
          </div>
          {status?.error ? <p className="error-copy">Training error: {status.error}</p> : null}
        </div>
      </div>

      <div className="ml-secondary">
        <div className="ml-section">
          <div className="section-header">
            <h3>Production APIs Needed</h3>
            <button type="button" className="primary-button" onClick={saveApis} disabled={isSavingApis}>
              {isSavingApis ? "Saving…" : "Save APIs"}
            </button>
          </div>
          <div className="api-list">
            {apiDrafts.map((api, index) => (
              <article key={api.name} className="api-card">
                <div className="section-header">
                  <strong>{api.name}</strong>
                  <span className={`status-pill ${api.status}`}>{api.status}</span>
                </div>
                <p>{api.purpose}</p>
                <small>{api.fields.join(" · ")}</small>
                <div className="connector-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => previewConnector(api.id)}
                    disabled={previewingId === api.id}
                  >
                    {previewingId === api.id ? "Previewing…" : "Preview Connector"}
                  </button>
                  {api.id === "copernicus_marine" ? (
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => syncConnector(api.id)}
                      disabled={syncingId === api.id}
                    >
                      {syncingId === api.id ? "Syncing…" : "Sync Copernicus"}
                    </button>
                  ) : null}
                </div>
                <div className="api-config-grid">
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={api.enabled}
                      onChange={(event) =>
                        setApiDrafts((current) =>
                          current.map((item, itemIndex) =>
                            itemIndex === index ? { ...item, enabled: event.target.checked } : item
                          )
                        )
                      }
                    />
                    Enabled
                  </label>
                  <label>
                    Source URL
                    <input
                      type="text"
                      value={api.url}
                      onChange={(event) =>
                        setApiDrafts((current) =>
                          current.map((item, itemIndex) =>
                            itemIndex === index ? { ...item, url: event.target.value } : item
                          )
                        )
                      }
                    />
                  </label>
                  <label>
                    Status
                    <select
                      value={api.status}
                      onChange={(event) =>
                        setApiDrafts((current) =>
                          current.map((item, itemIndex) =>
                            itemIndex === index ? { ...item, status: event.target.value } : item
                          )
                        )
                      }
                    >
                      <option value="planned">planned</option>
                      <option value="connected">connected</option>
                      <option value="testing">testing</option>
                    </select>
                  </label>
                  <label>
                    Notes
                    <input
                      type="text"
                      value={api.notes || ""}
                      placeholder={
                        api.id === "copernicus_marine"
                          ? "monthly_physics_dataset_id=<id>;routing_dataset_id=<id>;path=Training_Data/Copernicus"
                          : `Env: ${api.env_var}`
                      }
                      onChange={(event) =>
                        setApiDrafts((current) =>
                          current.map((item, itemIndex) =>
                            itemIndex === index ? { ...item, notes: event.target.value } : item
                          )
                        )
                      }
                    />
                  </label>
                </div>
                {previewById[api.id] ? (
                  <div className="preview-panel">
                    <strong>{previewById[api.id].status}</strong>
                    <p>{previewById[api.id].message}</p>
                    {previewById[api.id].search_url ? <small>{previewById[api.id].search_url}</small> : null}
                    {previewById[api.id].request_template ? (
                      <pre>{JSON.stringify(previewById[api.id].request_template, null, 2)}</pre>
                    ) : null}
                    {previewById[api.id].records ? (
                      <pre>{JSON.stringify(previewById[api.id].records, null, 2)}</pre>
                    ) : null}
                    {previewById[api.id].rows ? (
                      <pre>{JSON.stringify(previewById[api.id].rows, null, 2)}</pre>
                    ) : null}
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        </div>

        <div className="ml-section">
          <h3>Current Training Path</h3>
          <ol className="ml-steps">
            {usingRealData ? (
              <>
                <li>Load SOCAT surface-ocean observations from the configured source URL.</li>
                <li>Load NOAA GML monthly atmospheric CO2 and join it by year and month.</li>
                <li>Build monthly aligned tensors from ERA5 plus Copernicus salinity, currents, temperature, and sea level.</li>
                <li>Train the ConvLSTM on monthly sequences and save a checkpoint for later inference.</li>
                <li>Keep high-frequency Copernicus routing files ready for live-map inference.</li>
              </>
            ) : (
              <>
                <li>Enable SOCAT and NOAA GML connectors and keep their source URLs valid.</li>
                <li>Point ERA5 to your local folder or a real remote source.</li>
                <li>Sync Copernicus monthly physics and routing subsets once dataset IDs and credentials are configured.</li>
                <li>Start training only after those real inputs are confirmed.</li>
                <li>No synthetic fallback is used anywhere in this training path.</li>
              </>
            )}
          </ol>
          <p className="subtle">
            SOCAT: {connectorState.socat?.enabled ? "enabled" : "disabled"} · NOAA GML:{" "}
            {connectorState.noaa_gml_co2?.enabled ? "enabled" : "disabled"}
          </p>
          <p className="subtle">Enable these first: SOCAT, NOAA GML CO2, and your local ERA5 folder.</p>
        </div>

        <div className="ml-section">
          <h3>What Limits The Model Today</h3>
          <div className="ml-limit-list">
            {MODEL_LIMITS.map((item) => (
              <article key={item.title} className="ml-limit-card">
                <strong>{item.title}</strong>
                <p className="subtle">{item.copy}</p>
              </article>
            ))}
          </div>
        </div>

        <div className="ml-section">
          <h3>What Data Improves It Next</h3>
          <ol className="ml-steps">
            {NEXT_DATASETS.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}

import { useEffect, useState } from "react";

import { API_BASE } from "../../utils/constants";

const DEFAULT_FORM = {
  epochs: 18,
  learning_rate: 0.05,
  month_window: 12,
  resolution: "2deg",
};

const MODEL_LIMITS = [
  {
    title: "Model class is still lightweight",
    copy: "The current trainer is a fast regression baseline, not the full spatiotemporal ConvLSTM from the spec.",
  },
  {
    title: "Targets are only partially physical",
    copy: "We now use real SOCAT plus NOAA CO2, but the target still depends on estimated features where live gridded drivers are missing.",
  },
  {
    title: "ERA5 is local, not global",
    copy: "Your current ERA5 files are a point time-series at 0°, 0°, which helps temporal context but not global spatial training.",
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
  const dataSummary = status?.data_summary || {};
  const connectorState = dataSummary.connector_state || {};
  const usingRealData = dataSummary.source === "real_observation_sample";

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
          setApiDrafts(data.apis || []);
        }
      } catch (error) {
        console.error("Failed to load API config", error);
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
      const response = await fetch(`${API_BASE}/api/ml/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      setStatus(data.training);
    } catch (error) {
      console.error("Failed to start training", error);
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
    } catch (error) {
      console.error("Failed to save API config", error);
    } finally {
      setIsSavingApis(false);
    }
  }

  async function previewConnector(connectorId) {
    setPreviewingId(connectorId);
    try {
      const response = await fetch(`${API_BASE}/api/connectors/${connectorId}/preview`, {
        method: "POST",
      });
      const data = await response.json();
      setPreviewById((current) => ({ ...current, [connectorId]: data }));
    } catch (error) {
      console.error("Failed to preview connector", error);
    } finally {
      setPreviewingId("");
    }
  }

  return (
    <section className="ml-layout">
      <div className="ml-primary">
        <div className="ml-section">
          <p className="eyebrow">Training Workspace</p>
          <h2>OceanPulse ML Lab</h2>
          <p className="subtle">
            This lab can now train on a real observation sample built from SOCAT surface-ocean records plus NOAA GML
            monthly atmospheric CO2. If those connectors are not ready yet, OceanPulse falls back to the synthetic
            demo grid so the rest of the app stays usable.
          </p>
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
          <p className="subtle">
            Training source: <strong>{dataSummary.source || "synthetic_demo_grid"}</strong>
          </p>
          <div className="ml-metrics-grid">
            <article className="ml-metric-card">
              <span>Epoch</span>
              <strong>
                {status?.current_epoch || 0}/{status?.total_epochs || form.epochs}
              </strong>
            </article>
            <article className="ml-metric-card">
              <span>Validation loss</span>
              <strong>{status?.metrics?.val_loss ?? "--"}</strong>
            </article>
            <article className="ml-metric-card">
              <span>MAE</span>
              <strong>{status?.metrics?.mae ?? "--"}</strong>
            </article>
            <article className="ml-metric-card">
              <span>R²</span>
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
                      placeholder={`Env: ${api.env_var}`}
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
                <li>Enrich the observation sample with local ERA5 monthly conditions when that folder is available.</li>
                <li>Construct a first-pass air-sea flux target and train the regression model.</li>
              </>
            ) : (
              <>
                <li>Check whether SOCAT and NOAA GML connectors are enabled and have valid URLs.</li>
                <li>Fall back to the synthetic ocean grid if the real observation path is not ready.</li>
                <li>Train a lightweight regression model to keep the control plane and map outputs working.</li>
                <li>Switch automatically to the real observation sample once those connectors are configured.</li>
              </>
            )}
          </ol>
          <p className="subtle">
            SOCAT: {connectorState.socat?.enabled ? "enabled" : "disabled"} · NOAA GML:{" "}
            {connectorState.noaa_gml_co2?.enabled ? "enabled" : "disabled"}
          </p>
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

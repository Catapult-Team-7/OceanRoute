import { useEffect, useState } from "react";

import InfoHint from "../common/InfoHint";
import { API_BASE } from "../../utils/constants";

export default function ProgressDashboard() {
  const [status, setStatus] = useState(null);
  const [backendHealth, setBackendHealth] = useState({ reachable: false, detail: "Checking backend..." });

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        if (!cancelled) {
          setBackendHealth({
            reachable: data.status === "ok",
            detail: data.status === "ok" ? `Backend reachable at ${API_BASE}` : "Unexpected backend health response.",
          });
        }
      } catch (error) {
        if (!cancelled) {
          setBackendHealth({
            reachable: false,
            detail: `Backend unreachable at ${API_BASE}.`,
          });
        }
      }
    }

    async function loadStatus() {
      try {
        const response = await fetch(`${API_BASE}/api/ml/status`);
        const data = await response.json();
        if (!cancelled) setStatus(data);
      } catch (error) {
        if (!cancelled) setStatus(null);
      }
    }

    loadHealth();
    loadStatus();
    const interval = window.setInterval(loadStatus, 1200);
    const healthInterval = window.setInterval(loadHealth, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
      window.clearInterval(healthInterval);
    };
  }, []);

  const downloadProgress = status?.metrics?.download_progress;
  const connectorState = status?.data_summary?.connector_state || {};

  return (
    <section className="progress-layout">
      <div className="ml-section">
        <p className="eyebrow">Live Training Progress</p>
        <h2>Progress Monitor</h2>
        <p className={backendHealth.reachable ? "subtle" : "error-copy"}>{backendHealth.detail}</p>

        <div className="section-header">
          <h3>Run Status</h3>
          <span className={`status-pill ${status?.status || "idle"}`}>{status?.status || "idle"}</span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${(status?.progress || 0) * 100}%` }} />
        </div>
        <div className="progress-timeline">
          <article className="progress-card">
            <span className="metric-label">
              Current stage
              <InfoHint label="Current stage" description="The active step in the download, tensor build, or model training pipeline." />
            </span>
            <strong>{status?.metrics?.stage || "idle"}</strong>
            <small>{status?.metrics?.detail || "No training run in progress."}</small>
          </article>
          <article className="progress-card">
            <span className="metric-label">
              Download progress
              <InfoHint
                label="Download progress"
                description="Substep progress for live Copernicus data pulls when the monthly training grids are being synced."
              />
            </span>
            <strong>
              {downloadProgress ? `${downloadProgress.completed}/${downloadProgress.total}` : "—"}
            </strong>
            <small>{downloadProgress?.current || "No live download active."}</small>
          </article>
          <article className="progress-card">
            <span className="metric-label">
              Samples
              <InfoHint label="Samples" description="Number of monthly training examples available to the current run." />
            </span>
            <strong>{status?.data_summary?.sample_count ?? status?.metrics?.samples ?? "—"}</strong>
            <small>{status?.metrics?.tensor_samples ? `${status.metrics.tensor_samples} tensor samples` : "Waiting for tensors."}</small>
          </article>
          <article className="progress-card">
            <span className="metric-label">
              Checkpoint
              <InfoHint label="Checkpoint" description="Saved trained model artifact used for real map and point inference." />
            </span>
            <strong>{status?.model_ready ? "Ready" : "Not ready"}</strong>
            <small>{status?.model_summary?.checkpoint_path || "No checkpoint saved yet."}</small>
          </article>
        </div>
      </div>

      <div className="ml-section">
        <h3>Connectors</h3>
        <div className="progress-connectors">
          {Object.entries(connectorState).map(([key, value]) => (
            <article key={key} className="api-card">
              <div className="section-header">
                <strong>{key}</strong>
                <span className={`status-pill ${value.status || "planned"}`}>{value.status || "planned"}</span>
              </div>
              <p>{value.enabled ? "enabled" : "disabled"}</p>
              <small>{value.configured_url ? "source configured" : "source missing"}</small>
            </article>
          ))}
        </div>
      </div>

      <div className="ml-section">
        <h3>Training Metrics</h3>
        <div className="ml-metrics-grid">
          <article className="ml-metric-card">
            <span>Epoch</span>
            <strong>
              {status?.current_epoch || 0}/{status?.total_epochs || 0}
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
        {status?.error ? <p className="error-copy">Training error: {status.error}</p> : null}
      </div>
    </section>
  );
}

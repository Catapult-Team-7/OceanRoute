# OceanPulse

OceanPulse is a Catapult hackathon starter for an ocean carbon sink dashboard. This repo now includes:

- A FastAPI backend with demo heatmap, anomaly, stats, history, and forecast endpoints
- A React + Vite frontend with a map-focused dashboard and sidebar inspector
- A root-level starter that launches backend and frontend together
- Docker and SQL scaffolding so we can swap the demo repository out for PostGIS-backed ingestion later

## One starter

After setup, start the whole app from the repo root:

```bash
npm run dev
```

This launches:

- Backend at `http://127.0.0.1:8765`
- Frontend at `http://127.0.0.1:5173`
- Combined logs at `logs/oceanpulse-dev.log`

## Setup

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml-ingest.txt
cd ../frontend
npm install
cd ..
```

Use Python 3.11 for local backend setup. Python 3.13 may fail on heavier scientific packages because some wheels are not available yet.

## ERA5 API download

The real training pipeline expects local ERA5 CSVs outside the repo by default:

- `~/OceanPulseData/ERA`

Set up the official CDS API token first by creating `~/.cdsapirc`:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

You also need to accept the ERA5 dataset Terms of Use once on the CDS dataset page before the API will download data.

Then run:

```bash
cd backend
source .venv/bin/activate
python ingest/fetch_era5.py
```

That command downloads a normalized CSV into `~/OceanPulseData/ERA` with the columns the trainer expects:

- `valid_time`
- `u10`
- `v10`
- optional `sst`
- optional `msl`

You can override the location and date range if needed:

```bash
python ingest/fetch_era5.py --latitude 27.5 --longitude -140 --start-date 2024-01-01 --end-date 2025-12-31
```

## Supercomputer workflow

For HPC or batch environments, treat OceanPulse as three separate jobs:

1. `prepare` data artifacts
2. `train` from prepared tensors and save resumable checkpoints
3. `publish` verified map products for the frontend/API to serve

The batch entrypoints live under `backend/pipeline/`.

Prepare tensors and write a data manifest:

```bash
cd backend
source .venv/bin/activate
python -m pipeline.prepare_job --month-window 12 --resolution 2deg
```

Run a blocking training job:

```bash
python -m pipeline.train_job --epochs 40 --month-window 12 --resolution 2deg
```

Training now autosaves a resumable checkpoint every 10 epochs:

- latest served checkpoint: `backend/ml/checkpoints/oceanpulse_latest.pt`
- resumable in-progress state: `backend/ml/checkpoints/oceanpulse_resume.pt`

Publish a verified map artifact for the homepage/API:

```bash
python -m pipeline.publish_job --region global --resolution 1deg
```

Published products and manifests are written to:

- `backend/artifacts/manifests/latest_data_manifest.json`
- `backend/artifacts/manifests/latest_training_manifest.json`
- `backend/artifacts/published/map_bundle__global__1deg.json`

The homepage/API now prefers published map bundles first, which is much closer to how an HPC-backed system should serve products.

### Gautschi cluster example

Log in:

```bash
ssh yu1527@gautschi.rcac.purdue.edu
```

Clone or copy the repo into your home or project space, then from the repo root:

```bash
cd ~/Catapult-2026/backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml-ingest.txt
```

Then submit the staged jobs:

```bash
cd ~/Catapult-2026/backend
sbatch slurm/prepare_job.slurm
sbatch slurm/train_job.slurm
sbatch slurm/publish_job.slurm
```

Or run the full batch pipeline in one job:

```bash
cd ~/Catapult-2026/backend
sbatch slurm/run_full_pipeline.slurm
```

You can override defaults inline:

```bash
cd ~/Catapult-2026/backend
EPOCHS=120 MONTH_WINDOW=12 TRAIN_RESOLUTION=2deg PUBLISH_RESOLUTION=1deg sbatch slurm/run_full_pipeline.slurm
```

Useful Slurm checks:

```bash
squeue -u $USER
sacct -j <jobid>
tail -f ~/Catapult-2026/logs/oceanpulse-train-<jobid>.out
```

## Error logging

The backend now logs:

- Request start and finish with request IDs
- Validation errors with structured details
- HTTP errors with status and message
- Unhandled exceptions with full stack traces
- WebSocket connect, disconnect, and broadcast failures

If the UI or API misbehaves during the hackathon, check `logs/oceanpulse-dev.log` first.

## Current scope

This first implementation is designed for demo speed:

- The API returns synthetic but geographically-plausible ocean data
- Forecasts and anomalies are deterministic demo outputs, not trained ML outputs yet
- The database schema and ML module skeletons are in place for the next iteration

## Next build steps

1. Replace `backend/db/demo_data.py` with real database-backed CRUD.
2. Implement ingestion jobs under `backend/ingest/` for HYCOM, SST, SOCAT, and Copernicus data.
3. Train and wire the real forecasting pipeline in `backend/ml/`.
4. Add toast alerts, richer map controls, and time-series comparisons in the frontend.

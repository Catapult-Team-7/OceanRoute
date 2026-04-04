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

- Backend at `http://127.0.0.1:8000`
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

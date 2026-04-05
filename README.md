# OceanRoute v1

Multi-region floating-debris response support with an SF Bay pilot workflow and a Stage 2 data/ML runtime. The stack is a FastAPI forecast API plus a separate FastAPI inference service and a React/Leaflet operator dashboard. Forecasts combine:

- NOAA-style ingest with live-first plus sample fallback
- region-specific baseline grids and provenance
- canonical baseline manifests with dual-write tensor artifacts
- ensemble particle-advection baseline with windage, Stokes-drift proxy, diffusion, and beaching
- residual correction, calibrated uncertainty, and optional active learned adjustment
- dataset exports written as tensor artifacts plus sample indexes
- single-vessel route optimization with recon fallback
- mission feedback, impact ledger, routing benchmark artifacts, dataset builds, model registry entries, and inference-ready prediction artifacts

## Repo layout

- `apps/api`: FastAPI app, Alembic-backed schema, forecast/routing/impact/ML services, tests, CLI helpers
- `apps/api/app/inference_main.py`: separate FastAPI inference entrypoint for promoted models
- `apps/web`: Vite/React operator dashboard
- `data`: exports and debug artifacts such as latest forecast, routes, and benchmark reports
- `infra/windows`: scheduled forecast helpers for Windows ops
- `scripts`: local bootstrap helpers for DB setup and seeded demo data

## Runtime requirements

- Python 3.11
- Docker Desktop with `docker compose`
- Node `20.19.x`
- optional ML environment for deep training/inference: `apps/api/requirements-ml.txt` plus a matching `torch` wheel

The repo now enforces Node `20.19.x` with:

- repo-root `.nvmrc`
- `engines.node` in the root and web workspace `package.json`
- fail-fast Node version checks on web `dev`, `build`, and `test`

## Windows local setup

1. Install Python deps:

```powershell
cd C:\Users\clewr\Catapult-2026
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r apps\api\requirements.txt
```

Optional deep-training environment:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\Activate.ps1
pip install -r apps\api\requirements-ml.txt
# install torch separately to match your CPU/CUDA setup
```

2. Install frontend deps under Node `20.19.x`:

```powershell
cd C:\Users\clewr\Catapult-2026
npm install --no-audit --no-fund
```

3. Bootstrap Postgres and apply Alembic migrations:

```powershell
cd C:\Users\clewr\Catapult-2026
powershell -ExecutionPolicy Bypass -File scripts\bootstrap-db.ps1
```

If `.env` does not exist yet, the bootstrap script copies `.env.example` to `.env` before starting Postgres and running `alembic upgrade head`.

4. Seed the deterministic SF Bay mission scenario:

```powershell
cd C:\Users\clewr\Catapult-2026
powershell -ExecutionPolicy Bypass -File scripts\seed-sample-data.ps1
```

That script runs the full seeded loop:

- forecast run
- hotspot generation
- route optimization
- observation upload
- mission outcome logging
- impact dashboard update
- routing benchmark artifact generation

## Running the app

Fastest full local startup:

```powershell
cd C:\Users\clewr\Catapult-2026
powershell -ExecutionPolicy Bypass -File scripts\dev-up.ps1
```

Add `-BootstrapDb` when you want the launcher to run the local Postgres bootstrap first:

```powershell
cd C:\Users\clewr\Catapult-2026
powershell -ExecutionPolicy Bypass -File scripts\dev-up.ps1 -BootstrapDb
```

That launcher opens three PowerShell windows and starts:

- inference service on `http://127.0.0.1:8100`
- main API on `http://127.0.0.1:8000`
- frontend on `http://127.0.0.1:3000`

Start the API:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir apps\api --reload
```

Start the inference service in a second shell when you want the forecast API to call a separate runtime instead of the in-process fallback:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m uvicorn app.inference_main:app --app-dir apps\api --port 8100 --reload
```

Start the dashboard:

```powershell
cd C:\Users\clewr\Catapult-2026
npm run dev:web
```

The Vite dev server proxies `/api` to `http://localhost:8000`.

## One-command forecast run

Run a forecast cycle on demand:

```powershell
cd C:\Users\clewr\Catapult-2026
powershell -ExecutionPolicy Bypass -File infra\windows\run_forecast.ps1 -HorizonHours 24 -SourceMode auto
```

Register an hourly Windows task:

```powershell
cd C:\Users\clewr\Catapult-2026
powershell -ExecutionPolicy Bypass -File infra\windows\register-forecast-task.ps1 -IntervalMinutes 60
```

Run a different supported region from the CLI:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m app.cli run-forecast --region-id puget_sound --source-mode sample --horizon-hours 48
```

Run a forecast against an explicit candidate or champion model without changing promotion state:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m app.cli run-forecast --region-id sf_bay_estuary --source-mode sample --horizon-hours 24 --model-id <model-id>
```

Backfill historical baseline artifacts for one region:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m app.cli backfill-history --region-id long_island_sound --source-mode sample --days 28
```

`backfill-history` now defaults to the fast historical `dataset_only` mode. That mode writes only the minimal forecast-run rows plus baseline artifacts needed by the dataset builder and returns per-chunk timings. Use full live-style replay only when you explicitly need it:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m app.cli backfill-history --region-id sf_bay_estuary --source-mode sample --days 120 --mode dataset_only --chunk-days 28
.\.venv\Scripts\python.exe -m app.cli backfill-history --region-id sf_bay_estuary --source-mode sample --days 28 --mode live_parity
```

Supported built-in regions today:

- `sf_bay_estuary`
- `puget_sound`
- `long_island_sound`

## Seeded mission walkthrough

After running `scripts\seed-sample-data.ps1`, the canonical demo flow is already loaded into the DB. Open the UI and review:

- forecast trust summary on the operations map
- top hotspot ranking and exports
- route summary with linked forecast provenance
- impact ledger metrics

Useful API endpoints for the walkthrough:

- `GET /api/regions`
- `GET /api/forecast/latest`
- `GET /api/forecast/hotspots`
- `POST /api/route/optimize`
- `GET /api/route/{mission_id}`
- `POST /api/observations/upload`
- `POST /api/cleanup/log`
- `GET /api/impact/dashboard`
- `GET /api/impact/benchmarks/latest`
- `GET /api/export/geojson`
- `GET /api/export/pdf-brief`

ML and dataset endpoints:

- `POST /api/ml/datasets/build`
- `POST /api/ml/datasets/export`
- `GET /api/ml/datasets`
- `GET /api/ml/datasets/{dataset_id}`
- `POST /api/ml/train`
- `GET /api/ml/models`
- `GET /api/ml/models/{model_id}/evaluate`
- `GET /api/ml/models/{model_id}/export`
- `POST /api/ml/models/{model_id}/promote`
- `POST /api/forecast/backfill`

CLI equivalents:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m app.cli build-dataset --region-id sf_bay_estuary --lookback-hours 12 --target-horizons 24 48 72
.\.venv\Scripts\python.exe -m app.cli build-dataset --region-id sf_bay_estuary --lookback-hours 12 --target-horizons 24 48 72 --allow-partial-horizons
.\.venv\Scripts\python.exe -m app.cli inspect-dataset --dataset-id <dataset-id>
.\.venv\Scripts\python.exe -m app.cli train-model --region-id sf_bay_estuary --dataset-id <dataset-id> --architecture linear_residual
.\.venv\Scripts\python.exe -m app.cli evaluate-model --model-id <model-id>
.\.venv\Scripts\python.exe -m app.cli export-model --model-id <model-id>
.\.venv\Scripts\python.exe -m app.cli promote-model --model-id <model-id>
.\.venv\Scripts\python.exe -m app.cli list-ml --region-id sf_bay_estuary
```

Nested ML CLI:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m app.cli ml train --architecture convlstm --dataset-id <dataset-id> --regions sf_bay_estuary,puget_sound,long_island_sound --horizons 24,48,72 --training-scope shared --device auto --epochs 30 --batch-size 8 --promote-policy auto
.\.venv\Scripts\python.exe -m app.cli ml evaluate --model-id <model-id>
.\.venv\Scripts\python.exe -m app.cli ml export --model-id <model-id>
.\.venv\Scripts\python.exe -m app.cli ml list --region-id sf_bay_estuary
tensorboard --logdir data\training_runs
```

Dataset export now defaults to strict horizon coverage. If you request `24 48 72`, the export either produces all three horizons or fails with a coverage error. Use `--allow-partial-horizons` only when you explicitly want the exporter to shrink the horizon list to the common subset it can satisfy.

Current trainer behavior:

- `linear_residual` remains the lightweight fallback trainer and can auto-promote as a per-region or shared fallback model
- `temporal_unet` and `convlstm` now run through the offline deep-training package under `apps/api/app/ml/training/` and export inference-ready artifacts into `data\model_registry\`
- deep-model training requires the separate ML environment plus a matching `torch` wheel; if those are missing, deep requests fail fast with a clear 400 instead of pretending to train
- the nested `ml train` flow supports `shared`, `per_region`, and `both` scopes; `both` runs one shared job plus one per-region job per listed region
- successful deep training writes TensorBoard logs, checkpoints, plots, an exported deployable artifact, and then registers the resulting model automatically
- the forecast API records baseline engine, artifact provenance, model version, dataset version, and prediction artifact URI so the UI can surface trust metadata
- `model.ts` is the expected deep runtime artifact and is intentionally TorchScript; model metadata records whether export used `trace` or `script`
- `auto` promotion is conservative and now requires at least `50` filtered samples with at least `10` test samples; smaller runs stay `candidate` even if training/export succeeded
- candidate models can be tested directly in forecast runtime with `run-forecast --model-id <model-id>` or `POST /api/forecast/run` with `model_id`, without promoting them first
- deep runtime feature snapshots are now built from the exported model feature contract instead of a hardcoded channel/horizon assumption
- deep artifacts carry `lookback_hours`, `trained_horizons`, `tensor_layout`, `compatible_regions`, and the authoritative `input_channels` in their exported metadata

Recommended trust workflow:

1. Backfill more history for the region you want to validate first.
2. Rebuild a dataset with enough runs to produce a meaningful test split.
3. Train the deep model and inspect `ml evaluate` for sample counts, split counts, and promotion blockers.
4. Run one forecast with an explicit `model_id` to verify runtime loading and provenance.
5. Only then manually promote, or let `auto` promote after the data-sufficiency gate is satisfied.

## Verification

Backend tests:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m pytest -q apps\api\tests
```

The default backend pytest loop now skips `slow` and `e2e` tests. Use the helper scripts for the intended tier:

```powershell
cd C:\Users\clewr\Catapult-2026
powershell -ExecutionPolicy Bypass -File scripts\test-fast.ps1
powershell -ExecutionPolicy Bypass -File scripts\test-integration.ps1
powershell -ExecutionPolicy Bypass -File scripts\test-full.ps1
```

Frontend tests:

```powershell
cd C:\Users\clewr\Catapult-2026
npm run test:web
```

Frontend typecheck:

```powershell
cd C:\Users\clewr\Catapult-2026
node .\node_modules\typescript\bin\tsc -p apps\web\tsconfig.json --noEmit
```

Frontend build:

```powershell
cd C:\Users\clewr\Catapult-2026
npm run build:web
```

## Artifacts

Seeded and scheduled runs write the latest artifacts into `data\`:

- `data\forecasts\latest_forecast.json`
- `data\data_lake\datasets\...`
- `data\manifests\baseline\...`
- `data\predictions\...`
- `data\routes\latest_benchmark_report.json`
- `data\routes\*.json`
- `data\training_runs\...`
- `data\model_registry\...`
- `data\interim\latest_observation.json`
- `data\interim\latest_mission_outcome.json`

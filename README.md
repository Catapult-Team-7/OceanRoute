# OceanRoute v1

Regional floating-debris response support for the San Francisco Bay pilot. The stack is a FastAPI backend plus a React/Leaflet operator dashboard. Forecasts combine:

- NOAA-style ingest with live-first plus sample fallback
- deterministic drift baseline
- residual correction and calibrated uncertainty
- single-vessel route optimization with recon fallback
- mission feedback, impact ledger, and routing benchmark artifacts

## Repo layout

- `apps/api`: FastAPI app, Alembic-backed schema, forecast/routing/impact services, tests, CLI helpers
- `apps/web`: Vite/React operator dashboard
- `data`: exports and debug artifacts such as latest forecast, routes, and benchmark reports
- `infra/windows`: scheduled forecast helpers for Windows ops
- `scripts`: local bootstrap helpers for DB setup and seeded demo data

## Runtime requirements

- Python 3.11
- Docker Desktop with `docker compose`
- Node `20.19+` on Node 20

The repo now expects Node `20.19+` on Node 20 with:

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

2. Install frontend deps under Node `20.19+` on Node 20:

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

Start the API:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir apps\api --reload
```

Start the dashboard:

```powershell
cd C:\Users\clewr\Catapult-2026
npm run dev:web
```

The Vite dev server proxies `/api` to `http://localhost:8000`.

## GitHub Pages launch build

GitHub Pages cannot host the FastAPI backend, so the Pages build runs in a static demo mode backed by seeded JSON artifacts under `apps/web/public/demo`.

Local GitHub Pages-style build:

```bash
cd /Users/kruz/GithubRepos/Catapult-2026
env PATH=/opt/homebrew/opt/node@20/bin:$PATH \
  VITE_STATIC_DEMO=true \
  VITE_BASE_PATH=/Catapult-2026/ \
  npm run build --workspace apps/web
```

Preview that static build locally:

```bash
cd /Users/kruz/GithubRepos/Catapult-2026/apps/web
env PATH=/opt/homebrew/opt/node@20/bin:$PATH npm run preview
```

The included GitHub Actions workflow in `.github/workflows/deploy-pages.yml` builds this same static demo and deploys it to GitHub Pages. It triggers automatically on pushes to `main` and can also be run manually with `workflow_dispatch`.

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

## Seeded mission walkthrough

After running `scripts\seed-sample-data.ps1`, the canonical demo flow is already loaded into the DB. Open the UI and review:

- forecast trust summary on the operations map
- top hotspot ranking and exports
- route summary with linked forecast provenance
- impact ledger metrics

Useful API endpoints for the walkthrough:

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

## Verification

Backend tests:

```powershell
cd C:\Users\clewr\Catapult-2026
.\.venv\Scripts\python.exe -m pytest -q apps\api\tests
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
- `data\routes\latest_benchmark_report.json`
- `data\routes\*.json`
- `data\interim\latest_observation.json`
- `data\interim\latest_mission_outcome.json`

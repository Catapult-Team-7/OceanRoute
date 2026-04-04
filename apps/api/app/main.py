from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db
from app.routers import exports, forecast, operations, routing
from app.services.artifact_service import ensure_data_directories
from app.services.scheduler_service import APSCHEDULER_AVAILABLE, start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_data_directories()
    init_db()
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()

app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="Regional floating-debris response decision support system for the SF Bay pilot.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "pilot_region": settings.pilot_region,
        "default_horizon_hours": settings.forecast_horizon_hours,
        "scheduler_enabled": settings.scheduler_enabled and APSCHEDULER_AVAILABLE,
        "database_url": settings.database_url.split("://", maxsplit=1)[0],
        "ingest_mode": settings.ingest_mode,
        "forecast_stale_after_minutes": settings.forecast_stale_after_minutes,
        "default_depot_lat": settings.default_depot_lat,
        "default_depot_lon": settings.default_depot_lon,
    }


app.include_router(forecast.router, prefix=settings.api_prefix)
app.include_router(routing.router, prefix=settings.api_prefix)
app.include_router(operations.router, prefix=settings.api_prefix)
app.include_router(exports.router, prefix=settings.api_prefix)

from __future__ import annotations

from app.config import settings
from app.db import SessionLocal
from app.schemas import ForecastRunRequest
from app.services.forecast_service import run_forecast

try:
    from apscheduler.schedulers.background import BackgroundScheduler

    APSCHEDULER_AVAILABLE = True
except Exception:  # pragma: no cover - only used when dependency is missing
    BackgroundScheduler = None  # type: ignore[assignment]
    APSCHEDULER_AVAILABLE = False


_scheduler: BackgroundScheduler | None = None


def _scheduled_run() -> None:
    session = SessionLocal()
    try:
        run_forecast(
            ForecastRunRequest(
                horizon_hours=settings.forecast_horizon_hours,
                source_mode=settings.ingest_mode,  # type: ignore[arg-type]
            ),
            session,
        )
    finally:
        session.close()


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None or not settings.scheduler_enabled or not APSCHEDULER_AVAILABLE:
        return
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _scheduled_run,
        "interval",
        minutes=settings.scheduler_interval_minutes,
        id="forecast_cycle",
        replace_existing=True,
    )
    scheduler.start()
    _scheduler = scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    _scheduler.shutdown(wait=False)
    _scheduler = None

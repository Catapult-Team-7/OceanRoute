from __future__ import annotations

from app.db import SessionLocal
from app.services.forecast_service import latest_forecast
from app.services.scheduler_service import _scheduled_run


def test_scheduled_run_creates_a_forecast() -> None:
    _scheduled_run()
    with SessionLocal() as session:
        snapshot = latest_forecast(session)
    assert snapshot is not None
    assert snapshot.steps

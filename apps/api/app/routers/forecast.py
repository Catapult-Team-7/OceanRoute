from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import (
    DebrisClass,
    ForecastRunRequest,
    ForecastRunResponse,
    ForecastSnapshot,
    HotspotQueryResponse,
)
from app.services.forecast_service import latest_forecast, run_forecast

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("/run", response_model=ForecastRunResponse)
def run_forecast_endpoint(request: ForecastRunRequest, db: Session = Depends(get_db)) -> ForecastRunResponse:
    return run_forecast(request, db)


@router.get("/latest", response_model=ForecastSnapshot)
def latest_forecast_endpoint(
    horizon_hour: int | None = Query(default=None, ge=1, le=72),
    debris_class: DebrisClass | None = Query(default=None, alias="class"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
) -> ForecastSnapshot:
    snapshot = latest_forecast(
        db,
        horizon_hour=horizon_hour,
        debris_class=debris_class,
        min_confidence=min_confidence,
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No forecast has been generated yet.")
    return snapshot


@router.get("/hotspots", response_model=HotspotQueryResponse)
def hotspots_endpoint(
    horizon_hour: int | None = Query(default=None, ge=1, le=72),
    debris_class: DebrisClass | None = Query(default=None, alias="class"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
) -> HotspotQueryResponse:
    snapshot = latest_forecast(
        db,
        horizon_hour=horizon_hour,
        debris_class=debris_class,
        min_confidence=min_confidence,
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No forecast has been generated yet.")
    return HotspotQueryResponse(
        run_id=snapshot.run_id,
        generated_at=snapshot.generated_at,
        horizon_hours=snapshot.horizon_hours,
        source_mode_requested=snapshot.source_mode_requested,
        source_mode_used=snapshot.source_mode_used,
        is_fallback=snapshot.is_fallback,
        is_stale=snapshot.is_stale,
        age_minutes=snapshot.age_minutes,
        stale_after_minutes=snapshot.stale_after_minutes,
        filters={
            "horizon_hour": horizon_hour,
            "debris_class": debris_class,
            "min_confidence": min_confidence,
        },
        hotspots=snapshot.steps,
        top_hotspots=snapshot.top_hotspots,
        provenance=snapshot.provenance,
    )

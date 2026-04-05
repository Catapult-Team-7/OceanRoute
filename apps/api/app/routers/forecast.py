from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import (
    DebrisClass,
    ForecastRunRequest,
    ForecastRunResponse,
    ForecastSnapshot,
    HistoricalBackfillRequest,
    HistoricalBackfillResponse,
    HotspotQueryResponse,
)
from app.services.forecast_service import latest_forecast, run_forecast
from app.services.historical_backfill_service import run_historical_backfill

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("/run", response_model=ForecastRunResponse)
def run_forecast_endpoint(request: ForecastRunRequest, db: Session = Depends(get_db)) -> ForecastRunResponse:
    try:
        return run_forecast(request, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/backfill", response_model=HistoricalBackfillResponse)
def backfill_forecast_endpoint(
    request: HistoricalBackfillRequest,
    db: Session = Depends(get_db),
) -> HistoricalBackfillResponse:
    try:
        return run_historical_backfill(request, db)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/latest", response_model=ForecastSnapshot)
def latest_forecast_endpoint(
    region_id: str | None = Query(default=None),
    horizon_hour: int | None = Query(default=None, ge=1, le=72),
    debris_class: DebrisClass | None = Query(default=None, alias="class"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
) -> ForecastSnapshot:
    snapshot = latest_forecast(
        db,
        region_id=region_id,
        horizon_hour=horizon_hour,
        debris_class=debris_class,
        min_confidence=min_confidence,
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No forecast has been generated yet.")
    return snapshot


@router.get("/hotspots", response_model=HotspotQueryResponse)
def hotspots_endpoint(
    region_id: str | None = Query(default=None),
    horizon_hour: int | None = Query(default=None, ge=1, le=72),
    debris_class: DebrisClass | None = Query(default=None, alias="class"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
) -> HotspotQueryResponse:
    snapshot = latest_forecast(
        db,
        region_id=region_id,
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
        region=snapshot.region,
        source_mode_requested=snapshot.source_mode_requested,
        source_mode_used=snapshot.source_mode_used,
        is_fallback=snapshot.is_fallback,
        is_stale=snapshot.is_stale,
        age_minutes=snapshot.age_minutes,
        stale_after_minutes=snapshot.stale_after_minutes,
        filters={
            "region_id": region_id,
            "horizon_hour": horizon_hour,
            "debris_class": debris_class,
            "min_confidence": min_confidence,
        },
        hotspots=snapshot.steps,
        top_hotspots=snapshot.top_hotspots,
        provenance=snapshot.provenance,
    )

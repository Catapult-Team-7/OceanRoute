from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from db.crud import get_forecast
from db.database import get_repo
from db.demo_data import DemoOceanRepository
from db.models import ForecastResponse
from ingest.real_training_data import RealDataLoadError

router = APIRouter()


@router.get("/forecast", response_model=ForecastResponse)
async def forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    horizon: int = Query(default=72, ge=24, le=72),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    try:
        return await run_in_threadpool(repo.get_point_forecast, lat, lon, horizon)
    except RealDataLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

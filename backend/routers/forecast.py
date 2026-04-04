from typing import Annotated

from fastapi import APIRouter, Depends, Query

from db.crud import get_forecast
from db.database import get_repo
from db.demo_data import DemoOceanRepository
from db.models import ForecastResponse

router = APIRouter()


@router.get("/forecast", response_model=ForecastResponse)
async def forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    horizon: int = Query(default=72, ge=24, le=72),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    return await get_forecast(repo, lat=lat, lon=lon, horizon=horizon)

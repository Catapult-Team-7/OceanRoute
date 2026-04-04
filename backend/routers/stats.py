from typing import Annotated

from fastapi import APIRouter, Depends, Query

from db.crud import get_history, get_stats
from db.database import get_repo
from db.demo_data import DemoOceanRepository

router = APIRouter()


@router.get("/stats")
async def stats(
    date: str | None = Query(default=None, description="YYYY-MM"),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    return await get_stats(repo, date=date)


@router.get("/history")
async def history(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    months: int = Query(default=12, ge=3, le=24),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    return {"lat": lat, "lon": lon, "history": await get_history(repo, lat=lat, lon=lon, months=months)}

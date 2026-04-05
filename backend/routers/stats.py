from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from db.crud import get_history, get_stats
from db.database import get_repo
from db.demo_data import DemoOceanRepository
from ingest.real_training_data import RealDataLoadError

router = APIRouter()


@router.get("/stats")
async def stats(
    date: str | None = Query(default=None, description="YYYY-MM"),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    return await run_in_threadpool(repo.get_global_stats, date)


@router.get("/history")
async def history(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    months: int = Query(default=12, ge=3, le=24),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    try:
        history_points = await run_in_threadpool(repo.get_point_history, lat, lon, months)
        return {"lat": lat, "lon": lon, "history": history_points}
    except RealDataLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool

from db.crud import get_recent_anomalies
from db.database import get_repo
from db.demo_data import DemoOceanRepository

router = APIRouter()


@router.get("/anomalies")
async def anomalies(
    threshold: float = Query(default=0.7, ge=0, le=1),
    limit: int = Query(default=20, ge=1, le=100),
    date: str | None = Query(default=None, description="YYYY-MM"),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    records = await run_in_threadpool(repo.get_recent_anomalies, threshold, limit, date)
    return {"anomalies": [record.model_dump() for record in records]}

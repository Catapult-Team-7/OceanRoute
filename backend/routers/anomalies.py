from typing import Annotated

from fastapi import APIRouter, Depends, Query

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
    records = await get_recent_anomalies(repo, threshold=threshold, limit=limit, date=date)
    return {"anomalies": [record.model_dump() for record in records]}

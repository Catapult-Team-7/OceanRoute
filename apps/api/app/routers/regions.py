from __future__ import annotations

from fastapi import APIRouter

from app.schemas import RegionInfo
from app.services.region_service import list_regions

router = APIRouter(prefix="/regions", tags=["regions"])


@router.get("", response_model=list[RegionInfo])
def list_regions_endpoint() -> list[RegionInfo]:
    return list_regions()

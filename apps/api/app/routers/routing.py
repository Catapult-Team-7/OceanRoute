from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import RouteOptimizeRequest, RoutePlan
from app.services.routing_service import get_route, optimize_route

router = APIRouter(prefix="/route", tags=["route"])


@router.post("/optimize", response_model=RoutePlan)
def optimize_route_endpoint(request: RouteOptimizeRequest, db: Session = Depends(get_db)) -> RoutePlan:
    return optimize_route(request, db)


@router.get("/{mission_id}", response_model=RoutePlan)
def route_by_id_endpoint(mission_id: str, db: Session = Depends(get_db)) -> RoutePlan:
    plan = get_route(mission_id, db)
    if plan is None:
        raise HTTPException(status_code=404, detail="Mission route not found.")
    return plan

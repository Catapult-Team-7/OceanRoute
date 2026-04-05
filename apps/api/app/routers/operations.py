from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.schemas import (
    ImpactDashboard,
    MissionOutcome,
    MissionRecord,
    ObservationUpload,
    RouteOptimizeRequest,
    RoutingBenchmarkReport,
)
from app.services.benchmark_service import latest_benchmark_report
from app.services.impact_service import (
    add_observation,
    impact_dashboard,
    list_mission_records,
    log_mission_outcome,
    save_mission_record,
)
from app.services.region_service import get_region_definition

router = APIRouter(tags=["operations"])


@router.post("/observations/upload", response_model=ObservationUpload)
def observations_upload_endpoint(payload: ObservationUpload, db: Session = Depends(get_db)) -> ObservationUpload:
    return add_observation(payload, db)


@router.post("/cleanup/log", response_model=MissionOutcome)
def cleanup_log_endpoint(payload: MissionOutcome, db: Session = Depends(get_db)) -> MissionOutcome:
    return log_mission_outcome(payload, db)


@router.get("/missions", response_model=list[MissionRecord])
def missions_endpoint(db: Session = Depends(get_db)) -> list[MissionRecord]:
    return list_mission_records(db)


@router.post("/missions", response_model=MissionRecord)
def save_mission_endpoint(payload: MissionRecord, db: Session = Depends(get_db)) -> MissionRecord:
    return save_mission_record(payload, db)


@router.get("/impact/dashboard", response_model=ImpactDashboard)
def impact_dashboard_endpoint(db: Session = Depends(get_db)) -> ImpactDashboard:
    return impact_dashboard(db)


@router.get("/impact/benchmarks/latest", response_model=RoutingBenchmarkReport)
def latest_benchmark_endpoint(region_id: str | None = Query(default=None), db: Session = Depends(get_db)) -> RoutingBenchmarkReport:
    region = get_region_definition(region_id)
    return latest_benchmark_report(
        db,
        RouteOptimizeRequest(
            region_id=region.id,
            depot_lat=region.default_depot_lat,
            depot_lon=region.default_depot_lon,
            mission_hours=4.0,
            vessel_speed_kmh=settings.vessel_speed_kmh,
            fuel_burn_lph=12.0,
            target_horizon_hour=24,
            min_confidence=0.2,
            min_objective_score=0.5,
        ),
    )

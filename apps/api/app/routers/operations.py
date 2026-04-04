from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.schemas import ImpactDashboard, MissionOutcome, ObservationUpload, RouteOptimizeRequest, RoutingBenchmarkReport
from app.services.benchmark_service import latest_benchmark_report
from app.services.impact_service import add_observation, impact_dashboard, log_mission_outcome

router = APIRouter(tags=["operations"])


@router.post("/observations/upload", response_model=ObservationUpload)
def observations_upload_endpoint(payload: ObservationUpload, db: Session = Depends(get_db)) -> ObservationUpload:
    return add_observation(payload, db)


@router.post("/cleanup/log", response_model=MissionOutcome)
def cleanup_log_endpoint(payload: MissionOutcome, db: Session = Depends(get_db)) -> MissionOutcome:
    return log_mission_outcome(payload, db)


@router.get("/impact/dashboard", response_model=ImpactDashboard)
def impact_dashboard_endpoint(db: Session = Depends(get_db)) -> ImpactDashboard:
    return impact_dashboard(db)


@router.get("/impact/benchmarks/latest", response_model=RoutingBenchmarkReport)
def latest_benchmark_endpoint(db: Session = Depends(get_db)) -> RoutingBenchmarkReport:
    return latest_benchmark_report(
        db,
        RouteOptimizeRequest(
            depot_lat=settings.default_depot_lat,
            depot_lon=settings.default_depot_lon,
            mission_hours=4.0,
            vessel_speed_kmh=settings.vessel_speed_kmh,
            fuel_burn_lph=12.0,
            target_horizon_hour=24,
            min_confidence=0.2,
            min_objective_score=0.5,
        ),
    )

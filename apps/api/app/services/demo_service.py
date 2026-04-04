from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.schemas import ForecastRunRequest, MissionOutcome, ObservationUpload, RouteOptimizeRequest
from app.services.benchmark_service import latest_benchmark_report
from app.services.forecast_service import latest_forecast, run_forecast
from app.services.impact_service import add_observation, impact_dashboard, log_mission_outcome
from app.services.routing_service import optimize_route


def _now() -> datetime:
    return datetime.now(timezone.utc)


def seed_demo_scenario(db: Session) -> dict[str, object]:
    forecast = run_forecast(
        ForecastRunRequest(
            horizon_hours=48,
            debris_classes=["low", "high"],
            source_strength=1.1,
            seed=17,
            source_mode="sample",
        ),
        db,
    )
    snapshot = latest_forecast(db, horizon_hour=24, min_confidence=0.2)
    if snapshot is None:
        raise RuntimeError("Expected a seeded forecast snapshot, but none was available.")

    route = optimize_route(
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
        db,
    )

    top_target = snapshot.top_hotspots[0]
    add_observation(
        ObservationUpload(
            mission_id=route.mission_id,
            observed_at=_now(),
            lat=top_target.lat,
            lon=top_target.lon,
            found_status="found",
            debris_class=top_target.debris_class,
            estimated_kg=round((route.expected_kg_min + route.expected_kg_max) / 3, 3),
            confidence=0.86,
            photo_url="https://example.com/sf-bay-demo-observation.jpg",
            note="Seeded SF Bay post-outflow debris confirmation.",
            route_deviation_reason="Shifted slightly north with the ebb tide before pickup.",
        ),
        db,
    )

    log_mission_outcome(
        MissionOutcome(
            mission_id=route.mission_id,
            completed_at=_now() + timedelta(hours=4),
            vessel_id=route.vessel_id,
            recommended_mode=route.recommended_mode,
            predicted_kg_min=route.expected_kg_min,
            predicted_kg_max=route.expected_kg_max,
            collected_kg=round(max(route.expected_kg_min + 2.1, route.expected_kg_max * 0.92), 3),
            vessel_distance_km=max(route.expected_distance_km, 4.0),
            vessel_hours=max(route.expected_duration_min / 60, 1.0),
            hotspot_hits=max(1, len(route.ordered_cell_ids)),
            hotspot_misses=0,
            false_search_km=0.6 if route.recommended_mode == "collection" else 1.2,
            fuel_liters=max(route.estimated_fuel_liters, 6.0),
            route_deviation_reason="Seeded scenario: confirmed drift along the northbound channel edge.",
            notes="Deterministic SF Bay seeded walkthrough for demos and regression tests.",
        ),
        db,
    )

    impact = impact_dashboard(db)
    benchmark = latest_benchmark_report(
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
    return {
        "seed_name": "sf_bay_post_outflow_shift",
        "forecast_run_id": forecast.run_id,
        "mission_id": route.mission_id,
        "route_mode": route.recommended_mode,
        "impact_dashboard": impact.model_dump(mode="json"),
        "benchmark_report": benchmark.model_dump(mode="json"),
    }

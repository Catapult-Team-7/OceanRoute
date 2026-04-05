from __future__ import annotations

from app.schemas import RouteCandidate, RouteOptimizeRequest
from app.services.routing_service import optimize_route


def test_route_optimizer_respects_mission_window_and_avoids_blocked_cells(db_session) -> None:
    request = RouteOptimizeRequest(
        depot_lat=37.8,
        depot_lon=-122.38,
        mission_hours=1.1,
        vessel_speed_kmh=20.0,
        fuel_burn_lph=8.0,
        min_objective_score=0.1,
        avoid_cell_ids=["c3"],
        candidates=[
            RouteCandidate(
                cell_id="c1",
                lat=37.81,
                lon=-122.37,
                expected_kg_min=4,
                expected_kg_max=8,
                confidence=0.8,
                uncertainty=0.2,
                service_time_min=10,
            ),
            RouteCandidate(
                cell_id="c2",
                lat=37.83,
                lon=-122.36,
                expected_kg_min=5,
                expected_kg_max=10,
                confidence=0.5,
                uncertainty=0.4,
                service_time_min=20,
            ),
            RouteCandidate(
                cell_id="c3",
                lat=37.9,
                lon=-122.3,
                expected_kg_min=12,
                expected_kg_max=24,
                confidence=0.9,
                uncertainty=0.1,
                service_time_min=30,
            ),
        ],
    )
    route = optimize_route(request, db_session)
    assert "c3" not in route.ordered_cell_ids
    assert route.expected_duration_min <= 66


def test_recon_fallback_when_objective_is_too_low(db_session) -> None:
    request = RouteOptimizeRequest(
        depot_lat=37.8,
        depot_lon=-122.38,
        mission_hours=0.8,
        vessel_speed_kmh=18.0,
        fuel_burn_lph=20.0,
        min_objective_score=9.0,
        candidates=[
            RouteCandidate(
                cell_id="far_uncertain",
                lat=38.15,
                lon=-121.95,
                expected_kg_min=1,
                expected_kg_max=2,
                confidence=0.2,
                uncertainty=0.85,
                service_time_min=8,
            ),
        ],
    )
    route = optimize_route(request, db_session)
    assert route.recommended_mode == "recon"
    assert route.ordered_cell_ids == []
    assert route.alternates == ["far_uncertain"]

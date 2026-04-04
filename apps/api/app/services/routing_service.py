from __future__ import annotations

from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ForecastRunModel, RoutePlanModel
from app.schemas import ForecastSnapshot, RouteCandidate, RouteLeg, RouteOptimizeRequest, RoutePlan
from app.services.artifact_service import write_route_snapshot
from app.services.forecast_service import require_latest_forecast
from app.services.provenance_service import build_forecast_provenance

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    ORTOOLS_AVAILABLE = True
except Exception:  # pragma: no cover - fallback used only when dependency is missing
    ORTOOLS_AVAILABLE = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2) + cos(radians(lat1)) * cos(radians(lat2)) * (sin(dlon / 2) ** 2)
    return 2 * radius * asin(sqrt(a))


def _candidate_mid_kg(candidate: RouteCandidate) -> float:
    return (candidate.expected_kg_min + candidate.expected_kg_max) / 2


def _candidate_score(candidate: RouteCandidate, request: RouteOptimizeRequest) -> float:
    weights = request.objective_weights
    return (
        (_candidate_mid_kg(candidate) * weights.yield_weight)
        - (candidate.uncertainty * weights.uncertainty_weight)
    )


def _default_candidates(request: RouteOptimizeRequest, db: Session) -> tuple[ForecastSnapshot, list[RouteCandidate]]:
    snapshot = require_latest_forecast(
        db,
        horizon_hour=request.target_horizon_hour,
        min_confidence=request.min_confidence,
    )
    candidates: list[RouteCandidate] = []
    for step in snapshot.steps:
        if step.restricted:
            continue
        candidates.append(
            RouteCandidate(
                cell_id=f"{step.cell_id}:{step.debris_class}",
                lat=step.lat,
                lon=step.lon,
                expected_kg_min=step.expected_kg_min,
                expected_kg_max=step.expected_kg_max,
                confidence=step.confidence,
                uncertainty=step.uncertainty,
                service_time_min=12,
                access_flag=not step.restricted,
            )
        )
    ranked = sorted(candidates, key=lambda item: (_candidate_mid_kg(item), item.confidence), reverse=True)[:18]
    return snapshot, ranked


def _build_recon_plan(
    request: RouteOptimizeRequest,
    *,
    reason: str,
    alternates: list[str],
    objective_score: float = 0.0,
    forecast_snapshot: ForecastSnapshot | None = None,
) -> RoutePlan:
    return RoutePlan(
        mission_id=str(uuid4()),
        created_at=_now(),
        vessel_id=request.vessel_id,
        recommended_mode="recon",
        target_horizon_hour=request.target_horizon_hour,
        ordered_cell_ids=[],
        expected_kg_min=0.0,
        expected_kg_max=0.0,
        expected_distance_km=0.0,
        expected_duration_min=0.0,
        estimated_fuel_liters=0.0,
        objective_score=round(objective_score, 3),
        legs=[],
        uncertainty_risk=1.0,
        alternates=alternates[:3],
        forecast_run_id=forecast_snapshot.run_id if forecast_snapshot else None,
        forecast_provenance=forecast_snapshot.provenance if forecast_snapshot else None,
        forecast_is_stale=forecast_snapshot.is_stale if forecast_snapshot else False,
        metadata={"reason": reason},
    )


def _fallback_route(
    request: RouteOptimizeRequest,
    candidates: list[RouteCandidate],
    *,
    forecast_snapshot: ForecastSnapshot | None = None,
) -> RoutePlan:
    ordered = [
        item.cell_id
        for item in sorted(candidates, key=lambda candidate: _candidate_score(candidate, request), reverse=True)
        if item.access_flag and item.cell_id not in set(request.avoid_cell_ids)
    ]
    if not ordered:
        return _build_recon_plan(
            request,
            reason="No accessible candidates were available for fallback routing.",
            alternates=[],
            forecast_snapshot=forecast_snapshot,
        )
    best = next(item for item in candidates if item.cell_id == ordered[0])
    travel_distance = _haversine_km(request.depot_lat, request.depot_lon, best.lat, best.lon) * 2
    duration_min = ((travel_distance / request.vessel_speed_kmh) * 60) + best.service_time_min
    fuel = (duration_min / 60) * request.fuel_burn_lph
    objective = (
        (_candidate_mid_kg(best) * request.objective_weights.yield_weight)
        - (travel_distance * request.objective_weights.distance_weight)
        - (best.uncertainty * request.objective_weights.uncertainty_weight)
        - (fuel * request.objective_weights.fuel_weight)
    )
    if objective < request.min_objective_score:
        return _build_recon_plan(
            request,
            reason="Fallback route did not clear the collection objective threshold.",
            alternates=ordered,
            objective_score=objective,
            forecast_snapshot=forecast_snapshot,
        )
    return RoutePlan(
        mission_id=str(uuid4()),
        created_at=_now(),
        vessel_id=request.vessel_id,
        recommended_mode="collection",
        target_horizon_hour=request.target_horizon_hour,
        ordered_cell_ids=[best.cell_id],
        expected_kg_min=best.expected_kg_min,
        expected_kg_max=best.expected_kg_max,
        expected_distance_km=round(travel_distance, 3),
        expected_duration_min=round(duration_min, 2),
        estimated_fuel_liters=round(fuel, 2),
        objective_score=round(objective, 3),
        legs=[
            RouteLeg(
                from_cell="depot",
                to_cell=best.cell_id,
                distance_km=round(travel_distance / 2, 3),
                travel_minutes=round(((travel_distance / 2) / request.vessel_speed_kmh) * 60, 2),
            ),
            RouteLeg(
                from_cell=best.cell_id,
                to_cell="depot",
                distance_km=round(travel_distance / 2, 3),
                travel_minutes=round(((travel_distance / 2) / request.vessel_speed_kmh) * 60, 2),
            ),
        ],
        uncertainty_risk=best.uncertainty,
        alternates=ordered[1:4],
        forecast_run_id=forecast_snapshot.run_id if forecast_snapshot else None,
        forecast_provenance=forecast_snapshot.provenance if forecast_snapshot else None,
        forecast_is_stale=forecast_snapshot.is_stale if forecast_snapshot else False,
        metadata={"solver": "fallback"},
    )


def _solve_with_ortools(
    request: RouteOptimizeRequest,
    candidates: list[RouteCandidate],
    *,
    forecast_snapshot: ForecastSnapshot | None = None,
) -> RoutePlan:
    if not ORTOOLS_AVAILABLE:
        return _fallback_route(request, candidates, forecast_snapshot=forecast_snapshot)

    locations = [(request.depot_lat, request.depot_lon)] + [(item.lat, item.lon) for item in candidates]
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cost(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        from_lat, from_lon = locations[from_node]
        to_lat, to_lon = locations[to_node]
        distance_km = _haversine_km(from_lat, from_lon, to_lat, to_lon)
        travel_hours = distance_km / request.vessel_speed_kmh
        fuel_cost = travel_hours * request.fuel_burn_lph * request.objective_weights.fuel_weight
        distance_cost_value = distance_km * request.objective_weights.distance_weight
        return int(round((distance_cost_value + fuel_cost) * 100))

    def time_minutes(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        from_lat, from_lon = locations[from_node]
        to_lat, to_lon = locations[to_node]
        distance_km = _haversine_km(from_lat, from_lon, to_lat, to_lon)
        travel_min = (distance_km / request.vessel_speed_kmh) * 60
        service_time = 0 if to_node == 0 else candidates[to_node - 1].service_time_min
        return int(round(travel_min + service_time))

    cost_index = routing.RegisterTransitCallback(distance_cost)
    time_index = routing.RegisterTransitCallback(time_minutes)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_index)
    routing.AddDimension(
        time_index,
        0,
        int(round(request.mission_hours * 60)),
        True,
        "Time",
    )

    must_visit_ids = set(request.must_visit_cell_ids)
    for node_number, candidate in enumerate(candidates, start=1):
        if candidate.cell_id in must_visit_ids:
            continue
        reward = max(
            0.0,
            (_candidate_mid_kg(candidate) * request.objective_weights.yield_weight)
            - (candidate.uncertainty * request.objective_weights.uncertainty_weight),
        )
        penalty = int(round(reward * 160))
        routing.AddDisjunction([manager.NodeToIndex(node_number)], penalty)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 3

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return _build_recon_plan(
            request,
            reason="OR-Tools could not find a feasible collection route within the mission window.",
            alternates=[item.cell_id for item in candidates[:3]],
            forecast_snapshot=forecast_snapshot,
        )

    ordered_candidates: list[RouteCandidate] = []
    legs: list[RouteLeg] = []
    index = routing.Start(0)
    previous_label = "depot"
    previous_lat = request.depot_lat
    previous_lon = request.depot_lon
    total_distance = 0.0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        next_index = solution.Value(routing.NextVar(index))
        next_node = manager.IndexToNode(next_index)
        if next_node != 0:
            candidate = candidates[next_node - 1]
            ordered_candidates.append(candidate)
        next_lat, next_lon = locations[next_node]
        distance = _haversine_km(previous_lat, previous_lon, next_lat, next_lon)
        if next_node != 0 or previous_label != "depot":
            total_distance += distance
            legs.append(
                RouteLeg(
                    from_cell=previous_label,
                    to_cell="depot" if next_node == 0 else candidates[next_node - 1].cell_id,
                    distance_km=round(distance, 3),
                    travel_minutes=round((distance / request.vessel_speed_kmh) * 60, 2),
                )
            )
        if next_node != 0:
            previous_label = candidates[next_node - 1].cell_id
            previous_lat = next_lat
            previous_lon = next_lon
        index = next_index

    ordered_cell_ids = [item.cell_id for item in ordered_candidates]
    duration_min = sum(leg.travel_minutes for leg in legs) + sum(item.service_time_min for item in ordered_candidates)
    estimated_fuel_liters = (duration_min / 60) * request.fuel_burn_lph
    expected_kg_min = sum(item.expected_kg_min for item in ordered_candidates)
    expected_kg_max = sum(item.expected_kg_max for item in ordered_candidates)
    uncertainty_risk = (
        sum(item.uncertainty for item in ordered_candidates) / len(ordered_candidates) if ordered_candidates else 1.0
    )
    objective_score = (
        (((expected_kg_min + expected_kg_max) / 2) * request.objective_weights.yield_weight)
        - (total_distance * request.objective_weights.distance_weight)
        - (uncertainty_risk * request.objective_weights.uncertainty_weight)
        - (estimated_fuel_liters * request.objective_weights.fuel_weight)
    )

    if not ordered_candidates or objective_score < request.min_objective_score:
        ranked = sorted(candidates, key=lambda item: _candidate_score(item, request), reverse=True)
        return _build_recon_plan(
            request,
            reason="No collection route cleared the minimum objective score.",
            alternates=[item.cell_id for item in ranked],
            objective_score=objective_score,
            forecast_snapshot=forecast_snapshot,
        )

    alternate_candidates = [
        item.cell_id
        for item in sorted(candidates, key=lambda candidate: _candidate_score(candidate, request), reverse=True)
        if item.cell_id not in set(ordered_cell_ids)
    ]
    return RoutePlan(
        mission_id=str(uuid4()),
        created_at=_now(),
        vessel_id=request.vessel_id,
        recommended_mode="collection",
        target_horizon_hour=request.target_horizon_hour,
        ordered_cell_ids=ordered_cell_ids,
        expected_kg_min=round(expected_kg_min, 3),
        expected_kg_max=round(expected_kg_max, 3),
        expected_distance_km=round(total_distance, 3),
        expected_duration_min=round(duration_min, 2),
        estimated_fuel_liters=round(estimated_fuel_liters, 2),
        objective_score=round(objective_score, 3),
        legs=legs,
        uncertainty_risk=round(uncertainty_risk, 4),
        alternates=alternate_candidates[:3],
        forecast_run_id=forecast_snapshot.run_id if forecast_snapshot else None,
        forecast_provenance=forecast_snapshot.provenance if forecast_snapshot else None,
        forecast_is_stale=forecast_snapshot.is_stale if forecast_snapshot else False,
        metadata={"solver": "ortools", "candidate_count": len(candidates)},
    )


def _persist_route(plan: RoutePlan, db: Session) -> RoutePlan:
    existing = db.get(RoutePlanModel, plan.mission_id)
    payload = plan.model_dump(mode="json")
    if existing is None:
        db.add(
            RoutePlanModel(
                mission_id=plan.mission_id,
                created_at=plan.created_at,
                forecast_run_id=plan.forecast_run_id,
                vessel_id=plan.vessel_id,
                recommended_mode=plan.recommended_mode,
                target_horizon_hour=plan.target_horizon_hour,
                ordered_cell_ids=plan.ordered_cell_ids,
                expected_kg_min=plan.expected_kg_min,
                expected_kg_max=plan.expected_kg_max,
                expected_distance_km=plan.expected_distance_km,
                expected_duration_min=plan.expected_duration_min,
                estimated_fuel_liters=plan.estimated_fuel_liters,
                objective_score=plan.objective_score,
                uncertainty_risk=plan.uncertainty_risk,
                alternates=plan.alternates,
                legs=[item.model_dump(mode="json") for item in plan.legs],
                metadata_json=payload["metadata"],
            )
        )
    else:
        existing.created_at = plan.created_at
        existing.forecast_run_id = plan.forecast_run_id
        existing.vessel_id = plan.vessel_id
        existing.recommended_mode = plan.recommended_mode
        existing.target_horizon_hour = plan.target_horizon_hour
        existing.ordered_cell_ids = plan.ordered_cell_ids
        existing.expected_kg_min = plan.expected_kg_min
        existing.expected_kg_max = plan.expected_kg_max
        existing.expected_distance_km = plan.expected_distance_km
        existing.expected_duration_min = plan.expected_duration_min
        existing.estimated_fuel_liters = plan.estimated_fuel_liters
        existing.objective_score = plan.objective_score
        existing.uncertainty_risk = plan.uncertainty_risk
        existing.alternates = plan.alternates
        existing.legs = [item.model_dump(mode="json") for item in plan.legs]
        existing.metadata_json = payload["metadata"]
    db.commit()
    write_route_snapshot(plan.mission_id, payload)
    return plan


def optimize_route(request: RouteOptimizeRequest, db: Session) -> RoutePlan:
    forecast_snapshot: ForecastSnapshot | None = None
    if request.candidates:
        candidates = request.candidates
    else:
        forecast_snapshot, candidates = _default_candidates(request, db)
    filtered = [
        candidate
        for candidate in candidates
        if candidate.access_flag
        and candidate.confidence >= request.min_confidence
        and candidate.cell_id not in set(request.avoid_cell_ids)
    ]
    if not filtered:
        ranked_candidates = [
            candidate.cell_id
            for candidate in sorted(candidates, key=lambda item: (_candidate_mid_kg(item), item.confidence), reverse=True)
        ]
        plan = _build_recon_plan(
            request,
            reason="No candidates met the confidence and access filters.",
            alternates=ranked_candidates,
            forecast_snapshot=forecast_snapshot,
        )
        return _persist_route(plan, db)

    plan = _solve_with_ortools(request, filtered, forecast_snapshot=forecast_snapshot)
    return _persist_route(plan, db)


def get_route(mission_id: str, db: Session) -> RoutePlan | None:
    model = db.execute(
        select(RoutePlanModel).where(RoutePlanModel.mission_id == mission_id)
    ).scalar_one_or_none()
    if model is None:
        return None
    forecast_provenance = None
    forecast_is_stale = False
    if model.forecast_run_id:
        linked_run = db.get(ForecastRunModel, model.forecast_run_id)
        if linked_run is not None:
            forecast_provenance = build_forecast_provenance(
                generated_at=linked_run.generated_at,
                horizon_hours=linked_run.horizon_hours,
                source_mode_requested=linked_run.source_mode_requested,
                source_mode_used=linked_run.source_mode_used,
                is_fallback=linked_run.is_fallback,
                source_notes=list(linked_run.source_notes),
            )
            forecast_is_stale = forecast_provenance.is_stale
    return RoutePlan(
        mission_id=model.mission_id,
        created_at=model.created_at,
        vessel_id=model.vessel_id,
        recommended_mode=model.recommended_mode,  # type: ignore[arg-type]
        target_horizon_hour=model.target_horizon_hour,
        ordered_cell_ids=list(model.ordered_cell_ids),
        expected_kg_min=model.expected_kg_min,
        expected_kg_max=model.expected_kg_max,
        expected_distance_km=model.expected_distance_km,
        expected_duration_min=model.expected_duration_min,
        estimated_fuel_liters=model.estimated_fuel_liters,
        objective_score=model.objective_score,
        legs=[RouteLeg.model_validate(item) for item in model.legs],
        uncertainty_risk=model.uncertainty_risk,
        alternates=list(model.alternates),
        forecast_run_id=model.forecast_run_id,
        forecast_provenance=forecast_provenance,
        forecast_is_stale=forecast_is_stale,
        metadata=dict(model.metadata_json),
    )

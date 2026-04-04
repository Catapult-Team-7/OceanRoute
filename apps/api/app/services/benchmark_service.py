from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.schemas import RouteCandidate, RouteOptimizeRequest, RoutePlan, RoutingBenchmarkReport, RoutingBenchmarkStrategy
from app.services.artifact_service import write_debug_json
from app.services.routing_service import _default_candidates, _haversine_km, _solve_with_ortools


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sequence_metrics(sequence: list[RouteCandidate], request: RouteOptimizeRequest) -> dict[str, float]:
    if not sequence:
        return {
            "expected_kg_min": 0.0,
            "expected_kg_max": 0.0,
            "expected_distance_km": 0.0,
            "expected_duration_min": 0.0,
            "uncertainty_penalty": request.objective_weights.uncertainty_weight,
            "hotspot_hit_rate_estimate": 0.0,
            "objective_score": 0.0,
        }

    total_distance = 0.0
    total_service_minutes = 0.0
    expected_kg_min = 0.0
    expected_kg_max = 0.0
    confidence_total = 0.0
    uncertainty_total = 0.0
    current_lat = request.depot_lat
    current_lon = request.depot_lon

    for candidate in sequence:
        total_distance += _haversine_km(current_lat, current_lon, candidate.lat, candidate.lon)
        total_service_minutes += candidate.service_time_min
        expected_kg_min += candidate.expected_kg_min
        expected_kg_max += candidate.expected_kg_max
        confidence_total += candidate.confidence
        uncertainty_total += candidate.uncertainty
        current_lat = candidate.lat
        current_lon = candidate.lon

    total_distance += _haversine_km(current_lat, current_lon, request.depot_lat, request.depot_lon)
    travel_minutes = (total_distance / request.vessel_speed_kmh) * 60
    expected_duration_min = travel_minutes + total_service_minutes
    uncertainty_penalty = (
        (uncertainty_total / len(sequence)) * request.objective_weights.uncertainty_weight
    )
    estimated_fuel = (expected_duration_min / 60) * request.fuel_burn_lph
    objective_score = (
        (((expected_kg_min + expected_kg_max) / 2) * request.objective_weights.yield_weight)
        - (total_distance * request.objective_weights.distance_weight)
        - uncertainty_penalty
        - (estimated_fuel * request.objective_weights.fuel_weight)
    )
    return {
        "expected_kg_min": round(expected_kg_min, 3),
        "expected_kg_max": round(expected_kg_max, 3),
        "expected_distance_km": round(total_distance, 3),
        "expected_duration_min": round(expected_duration_min, 2),
        "uncertainty_penalty": round(uncertainty_penalty, 3),
        "hotspot_hit_rate_estimate": round(confidence_total / len(sequence), 4),
        "objective_score": round(objective_score, 3),
    }


def _build_greedy_strategy(
    *,
    strategy: str,
    request: RouteOptimizeRequest,
    candidates: list[RouteCandidate],
    sort_key,
    reverse: bool = False,
) -> RoutingBenchmarkStrategy:
    ordered: list[RouteCandidate] = []
    for candidate in sorted(candidates, key=sort_key, reverse=reverse):
        trial = ordered + [candidate]
        metrics = _sequence_metrics(trial, request)
        if metrics["expected_duration_min"] <= (request.mission_hours * 60):
            ordered = trial

    metrics = _sequence_metrics(ordered, request)
    recommended_mode = "collection" if ordered and metrics["objective_score"] >= request.min_objective_score else "recon"
    return RoutingBenchmarkStrategy(
        strategy=strategy,  # type: ignore[arg-type]
        recommended_mode=recommended_mode,  # type: ignore[arg-type]
        ordered_cell_ids=[item.cell_id for item in ordered] if recommended_mode == "collection" else [],
        expected_kg_min=metrics["expected_kg_min"] if recommended_mode == "collection" else 0.0,
        expected_kg_max=metrics["expected_kg_max"] if recommended_mode == "collection" else 0.0,
        expected_distance_km=metrics["expected_distance_km"] if recommended_mode == "collection" else 0.0,
        uncertainty_penalty=metrics["uncertainty_penalty"],
        hotspot_hit_rate_estimate=metrics["hotspot_hit_rate_estimate"],
        objective_score=metrics["objective_score"],
    )


def _strategy_from_route(name: str, route: RoutePlan, request: RouteOptimizeRequest) -> RoutingBenchmarkStrategy:
    uncertainty_penalty = round(route.uncertainty_risk * request.objective_weights.uncertainty_weight, 3)
    return RoutingBenchmarkStrategy(
        strategy=name,  # type: ignore[arg-type]
        recommended_mode=route.recommended_mode,
        ordered_cell_ids=route.ordered_cell_ids,
        expected_kg_min=route.expected_kg_min,
        expected_kg_max=route.expected_kg_max,
        expected_distance_km=route.expected_distance_km,
        uncertainty_penalty=uncertainty_penalty,
        hotspot_hit_rate_estimate=round(max(0.0, 1.0 - route.uncertainty_risk), 4),
        objective_score=route.objective_score,
    )


def latest_benchmark_report(db: Session, request: RouteOptimizeRequest) -> RoutingBenchmarkReport:
    snapshot, candidates = _default_candidates(request, db)

    nearest = _build_greedy_strategy(
        strategy="nearest_hotspot",
        request=request,
        candidates=candidates,
        sort_key=lambda item: _haversine_km(request.depot_lat, request.depot_lon, item.lat, item.lon),
    )
    highest_yield = _build_greedy_strategy(
        strategy="highest_yield",
        request=request,
        candidates=candidates,
        sort_key=lambda item: ((item.expected_kg_min + item.expected_kg_max) / 2, item.confidence),
        reverse=True,
    )
    recon_aware_route = _solve_with_ortools(request, candidates, forecast_snapshot=snapshot)
    recon_aware = _strategy_from_route("recon_aware", recon_aware_route, request)

    compared = [nearest, highest_yield, recon_aware]
    winning = max(compared, key=lambda item: item.objective_score)
    report = RoutingBenchmarkReport(
        generated_at=_now(),
        forecast_run_id=snapshot.run_id,
        target_horizon_hour=request.target_horizon_hour,
        compared_strategies=compared,
        winning_strategy=winning.strategy,
    )
    write_debug_json("routes", "latest_benchmark_report.json", report.model_dump(mode="json"))
    return report

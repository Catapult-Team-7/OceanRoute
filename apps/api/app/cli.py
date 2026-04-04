from __future__ import annotations

import argparse
import json

from app.config import settings
from app.db import SessionLocal, init_db
from app.schemas import ForecastRunRequest, RouteOptimizeRequest
from app.services.benchmark_service import latest_benchmark_report
from app.services.demo_service import seed_demo_scenario
from app.services.forecast_service import run_forecast


def _run_forecast(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        response = run_forecast(
            ForecastRunRequest(
                horizon_hours=args.horizon_hours,
                debris_classes=["low", "high"],
                source_strength=args.source_strength,
                seed=args.seed,
                source_mode=args.source_mode,
            ),
            session,
        )
        print(response.model_dump_json(indent=2))
    finally:
        session.close()


def _seed_demo(_: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        payload = seed_demo_scenario(session)
        print(json.dumps(payload, indent=2, default=str))
    finally:
        session.close()


def _benchmark(_: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        report = latest_benchmark_report(
            session,
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
        print(report.model_dump_json(indent=2))
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SeaSweep operational helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast_parser = subparsers.add_parser("run-forecast", help="Run a forecast cycle.")
    forecast_parser.add_argument("--horizon-hours", type=int, default=24)
    forecast_parser.add_argument("--source-mode", choices=["auto", "sample", "live"], default="auto")
    forecast_parser.add_argument("--source-strength", type=float, default=1.0)
    forecast_parser.add_argument("--seed", type=int, default=42)
    forecast_parser.set_defaults(func=_run_forecast)

    seed_parser = subparsers.add_parser("seed-demo", help="Seed the canonical SF Bay mission loop.")
    seed_parser.set_defaults(func=_seed_demo)

    benchmark_parser = subparsers.add_parser("benchmark-route", help="Run the deterministic routing benchmark.")
    benchmark_parser.set_defaults(func=_benchmark)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

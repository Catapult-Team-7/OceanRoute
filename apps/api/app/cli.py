from __future__ import annotations

import argparse
import json

from app.config import settings
from app.db import SessionLocal, init_db
from app.schemas import (
    DatasetBuildRequest,
    DatasetExportRequest,
    ForecastRunRequest,
    HistoricalBackfillRequest,
    ModelTrainRequest,
    RouteOptimizeRequest,
)
from app.services.benchmark_service import latest_benchmark_report
from app.services.dataset_service import build_dataset, inspect_dataset, list_datasets
from app.services.demo_service import seed_demo_scenario
from app.services.forecast_service import run_forecast
from app.services.historical_backfill_service import run_historical_backfill
from app.services.model_registry_service import (
    evaluate_model,
    export_model_artifact,
    list_models,
    promote_model,
    train_model,
)
from app.services.region_service import get_region_definition


def _parse_csv(raw: str | None, *, cast=str) -> list:
    if raw is None:
        return []
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def _resolved_regions(args: argparse.Namespace) -> list[str]:
    regions = _parse_csv(getattr(args, "regions", None), cast=str)
    if regions:
        return regions
    region_id = getattr(args, "region_id", None)
    return [region_id or settings.pilot_region]


def _resolved_horizons(args: argparse.Namespace) -> list[int]:
    raw_horizons = getattr(args, "horizons", None)
    if raw_horizons:
        return _parse_csv(raw_horizons, cast=int)
    target_horizons = getattr(args, "target_horizons", None)
    if target_horizons:
        return list(target_horizons)
    return [24, 48, 72]


def _run_forecast(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        response = run_forecast(
            ForecastRunRequest(
                region_id=args.region_id,
                horizon_hours=args.horizon_hours,
                debris_classes=["low", "high"],
                source_strength=args.source_strength,
                seed=args.seed,
                source_mode=args.source_mode,
                model_id=args.model_id,
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


def _benchmark(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        region = get_region_definition(args.region_id)
        report = latest_benchmark_report(
            session,
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
        print(report.model_dump_json(indent=2))
    finally:
        session.close()


def _build_dataset(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        dataset = build_dataset(
            DatasetExportRequest(
                region_id=args.region_id,
                region_ids=_resolved_regions(args),
                dataset_id=getattr(args, "dataset_id", None),
                max_forecast_runs=args.max_forecast_runs,
                lookback_hours=args.lookback_hours,
                target_horizons=_resolved_horizons(args),
            ),
            session,
        )
        print(dataset.model_dump_json(indent=2))
    finally:
        session.close()


def _inspect_dataset(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        print(json.dumps(inspect_dataset(args.dataset_id, session), indent=2, default=str))
    finally:
        session.close()


def _train_model(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        regions = _resolved_regions(args)
        horizons = _resolved_horizons(args)
        scope = getattr(args, "training_scope", "shared")
        requests: list[ModelTrainRequest] = []
        if scope == "both":
            requests.append(
                ModelTrainRequest(
                    region_id=None,
                    region_ids=regions,
                    dataset_id=args.dataset_id,
                    architecture=args.architecture,
                    activate=not args.no_activate,
                    horizons=horizons,
                    training_scope="shared",
                    device=args.device,
                    promote_policy=args.promote_policy,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )
            )
            for region_id in regions:
                requests.append(
                    ModelTrainRequest(
                        region_id=region_id,
                        region_ids=[region_id],
                        dataset_id=args.dataset_id,
                        architecture=args.architecture,
                        activate=not args.no_activate,
                        horizons=horizons,
                        training_scope="per_region",
                        device=args.device,
                        promote_policy=args.promote_policy,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                    )
                )
        else:
            requests.append(
                ModelTrainRequest(
                    region_id=regions[0] if scope == "per_region" else None,
                    region_ids=regions,
                    dataset_id=args.dataset_id,
                    architecture=args.architecture,
                    activate=not args.no_activate,
                    horizons=horizons,
                    training_scope=scope,
                    device=args.device,
                    promote_policy=args.promote_policy,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )
            )
        responses = [train_model(request, session) for request in requests]
        if len(responses) == 1:
            print(responses[0].model_dump_json(indent=2))
        else:
            print(json.dumps({"runs": [response.model_dump(mode="json") for response in responses]}, indent=2, default=str))
    finally:
        session.close()


def _evaluate_model(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        print(evaluate_model(args.model_id, session).model_dump_json(indent=2))
    finally:
        session.close()


def _export_model(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        print(export_model_artifact(args.model_id, session).model_dump_json(indent=2))
    finally:
        session.close()


def _promote_model(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        print(promote_model(args.model_id, session).model_dump_json(indent=2))
    finally:
        session.close()


def _backfill_history(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        response = run_historical_backfill(
            HistoricalBackfillRequest(
                region_id=args.region_id,
                source_mode=args.source_mode,
                days=args.days,
                mode=args.mode,
                chunk_days=args.chunk_days,
                ensemble_members=args.ensemble_members,
                particles_per_member=args.particles_per_member,
                debris_classes=["low", "high"],
            ),
            session,
        )
        print(response.model_dump_json(indent=2))
    finally:
        session.close()


def _list_ml(args: argparse.Namespace) -> None:
    init_db()
    session = SessionLocal()
    try:
        payload = {
            "datasets": [item.model_dump(mode="json") for item in list_datasets(session, region_id=args.region_id)],
            "models": [item.model_dump(mode="json") for item in list_models(session, region_id=args.region_id)],
        }
        print(json.dumps(payload, indent=2, default=str))
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OceanRoute operational helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast_parser = subparsers.add_parser("run-forecast", help="Run a forecast cycle.")
    forecast_parser.add_argument("--region-id", default=settings.pilot_region)
    forecast_parser.add_argument("--horizon-hours", type=int, default=24)
    forecast_parser.add_argument("--source-mode", choices=["auto", "sample", "live"], default="auto")
    forecast_parser.add_argument("--source-strength", type=float, default=1.0)
    forecast_parser.add_argument("--seed", type=int, default=42)
    forecast_parser.add_argument("--model-id", default=None)
    forecast_parser.set_defaults(func=_run_forecast)

    seed_parser = subparsers.add_parser("seed-demo", help="Seed the canonical SF Bay mission loop.")
    seed_parser.set_defaults(func=_seed_demo)

    benchmark_parser = subparsers.add_parser("benchmark-route", help="Run the deterministic routing benchmark.")
    benchmark_parser.add_argument("--region-id", default=settings.pilot_region)
    benchmark_parser.set_defaults(func=_benchmark)

    dataset_parser = subparsers.add_parser("build-dataset", help="Build a canonical training dataset export.")
    dataset_parser.add_argument("--region-id", default=settings.pilot_region)
    dataset_parser.add_argument("--regions", default=None)
    dataset_parser.add_argument("--dataset-id", default=None)
    dataset_parser.add_argument("--max-forecast-runs", type=int, default=10)
    dataset_parser.add_argument("--lookback-hours", type=int, default=settings.training_lookback_hours)
    dataset_parser.add_argument("--target-horizons", nargs="+", type=int, default=[24, 48, 72])
    dataset_parser.set_defaults(func=_build_dataset)

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Inspect one dataset export and preview its index.")
    inspect_parser.add_argument("--dataset-id", required=True)
    inspect_parser.set_defaults(func=_inspect_dataset)

    train_parser = subparsers.add_parser("train-model", help="Train a regional model against a canonical dataset export.")
    train_parser.add_argument("--region-id", default=settings.pilot_region)
    train_parser.add_argument("--regions", default=None)
    train_parser.add_argument("--dataset-id", required=True)
    train_parser.add_argument("--architecture", choices=["linear_residual", "temporal_unet", "convlstm"], default="linear_residual")
    train_parser.add_argument("--no-activate", action="store_true")
    train_parser.add_argument("--horizons", default="24,48,72")
    train_parser.add_argument("--training-scope", choices=["shared", "per_region", "both"], default="shared")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--promote-policy", choices=["auto", "candidate_only", "always_activate"], default="auto")
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.set_defaults(func=_train_model)

    evaluate_parser = subparsers.add_parser("evaluate-model", help="Read the stored evaluation artifact for a model.")
    evaluate_parser.add_argument("--model-id", required=True)
    evaluate_parser.set_defaults(func=_evaluate_model)

    export_parser = subparsers.add_parser("export-model", help="Show the exported model artifact path.")
    export_parser.add_argument("--model-id", required=True)
    export_parser.set_defaults(func=_export_model)

    promote_parser = subparsers.add_parser("promote-model", help="Promote a candidate model to champion.")
    promote_parser.add_argument("--model-id", required=True)
    promote_parser.set_defaults(func=_promote_model)

    backfill_parser = subparsers.add_parser("backfill-history", help="Generate historical forecast/baseline artifacts.")
    backfill_parser.add_argument("--region-id", default=settings.pilot_region)
    backfill_parser.add_argument("--source-mode", choices=["sample", "live", "auto"], default="sample")
    backfill_parser.add_argument("--days", type=int, default=settings.historical_backfill_days)
    backfill_parser.add_argument("--mode", choices=["dataset_only", "live_parity"], default="dataset_only")
    backfill_parser.add_argument("--chunk-days", type=int, default=28)
    backfill_parser.add_argument("--ensemble-members", type=int, default=None)
    backfill_parser.add_argument("--particles-per-member", type=int, default=None)
    backfill_parser.set_defaults(func=_backfill_history)

    list_parser = subparsers.add_parser("list-ml", help="List ML datasets and models.")
    list_parser.add_argument("--region-id", default=None)
    list_parser.set_defaults(func=_list_ml)

    ml_parser = subparsers.add_parser("ml", help="Nested ML helpers.")
    ml_subparsers = ml_parser.add_subparsers(dest="ml_command", required=True)

    ml_train = ml_subparsers.add_parser("train", help="Train one shared or per-region model flow.")
    ml_train.add_argument("--region-id", default=settings.pilot_region)
    ml_train.add_argument("--regions", default=None)
    ml_train.add_argument("--dataset-id", required=True)
    ml_train.add_argument("--architecture", choices=["linear_residual", "temporal_unet", "convlstm"], default="linear_residual")
    ml_train.add_argument("--no-activate", action="store_true")
    ml_train.add_argument("--horizons", default="24,48,72")
    ml_train.add_argument("--training-scope", choices=["shared", "per_region", "both"], default="shared")
    ml_train.add_argument("--device", default="auto")
    ml_train.add_argument("--promote-policy", choices=["auto", "candidate_only", "always_activate"], default="auto")
    ml_train.add_argument("--epochs", type=int, default=2)
    ml_train.add_argument("--batch-size", type=int, default=4)
    ml_train.set_defaults(func=_train_model)

    ml_evaluate = ml_subparsers.add_parser("evaluate", help="Read the stored evaluation artifact for a model.")
    ml_evaluate.add_argument("--model-id", required=True)
    ml_evaluate.set_defaults(func=_evaluate_model)

    ml_export = ml_subparsers.add_parser("export", help="Show the exported model artifact path.")
    ml_export.add_argument("--model-id", required=True)
    ml_export.set_defaults(func=_export_model)

    ml_list = ml_subparsers.add_parser("list", help="List ML datasets and models.")
    ml_list.add_argument("--region-id", default=None)
    ml_list.set_defaults(func=_list_ml)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

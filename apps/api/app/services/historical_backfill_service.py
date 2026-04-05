from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from math import ceil
from time import perf_counter
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import BaselineArtifactModel, ForecastRunModel
from app.schemas import (
    BaselineArtifact,
    DriftBaselineInput,
    ForecastRunRequest,
    HistoricalBackfillRequest,
    HistoricalBackfillResponse,
    HistoricalBackfillTiming,
)
from app.services.baseline_runtime_service import baseline_artifact_to_model, build_baseline_artifact, run_baseline_diagnostics
from app.services.data_lake_service import write_baseline_manifests, write_static_masks
from app.services.forecast_service import run_forecast
from app.services.ingest_service import load_operational_context
from app.services.region_service import REGIONS


BACKFILL_STRIDE_DAYS = 7
DATASET_ONLY_DEFAULT_ENSEMBLE_MEMBERS = 4
DATASET_ONLY_DEFAULT_PARTICLES_PER_MEMBER = 96
LIVE_PARITY_DEFAULT_ENSEMBLE_MEMBERS = 10
LIVE_PARITY_DEFAULT_PARTICLES_PER_MEMBER = 240


def _normalized_hour_now() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _backfill_dates(days: int) -> list[datetime]:
    end = _normalized_hour_now()
    dates: list[datetime] = []
    offset = 0
    while offset < days:
        dates.append(end - timedelta(days=offset))
        offset += BACKFILL_STRIDE_DAYS
    return list(reversed(dates))


def _chunk_dates(dates: list[datetime], chunk_days: int) -> list[list[datetime]]:
    chunk_size = max(1, ceil(chunk_days / BACKFILL_STRIDE_DAYS))
    return [dates[index : index + chunk_size] for index in range(0, len(dates), chunk_size)]


def _existing_run_timestamps(region_id: str, db: Session) -> set[datetime]:
    rows = db.execute(
        select(ForecastRunModel.generated_at).where(ForecastRunModel.pilot_region == region_id)
    ).scalars().all()
    return {row.replace(minute=0, second=0, microsecond=0) for row in rows}


def _resolve_baseline_fidelity(request: HistoricalBackfillRequest) -> tuple[int, int]:
    if request.mode == "dataset_only":
        return (
            request.ensemble_members or DATASET_ONLY_DEFAULT_ENSEMBLE_MEMBERS,
            request.particles_per_member or DATASET_ONLY_DEFAULT_PARTICLES_PER_MEMBER,
        )
    return (
        request.ensemble_members or LIVE_PARITY_DEFAULT_ENSEMBLE_MEMBERS,
        request.particles_per_member or LIVE_PARITY_DEFAULT_PARTICLES_PER_MEMBER,
    )


def _minimal_summary(
    *,
    mode: str,
    frame_count: int,
    baseline_engine: str | None,
    baseline_artifact_uri: str | None,
    baseline_artifact_count: int,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "historical_mode": mode,
        "baseline_only": True,
        "step_count": 0,
        "frame_count": frame_count,
        "baseline_artifact_count": baseline_artifact_count,
    }
    if baseline_engine:
        summary["baseline_engine"] = baseline_engine
    if baseline_artifact_uri:
        summary["baseline_artifact_uri"] = baseline_artifact_uri
    return summary


def _run_payload(
    *,
    request: HistoricalBackfillRequest,
    run_id: str,
    context,
    baseline_engine: str | None,
    baseline_artifact_count: int,
) -> dict[str, object]:
    return {
        "id": run_id,
        "generated_at": context.generated_at.isoformat(),
        "horizon_hours": 72,
        "pilot_region": context.pilot_region,
        "source_mode_requested": context.source_mode_requested,
        "source_mode_used": context.source_mode_used,
        "is_fallback": context.is_fallback,
        "source_notes": list(context.source_notes),
        "frame_count": len(context.frames),
        "baseline_engine": baseline_engine,
        "baseline_artifact_count": baseline_artifact_count,
        "mode": request.mode,
    }


def _prewarm_static_masks(region_id: str, timestamps: list[datetime]) -> None:
    months = {
        timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        for timestamp in timestamps
    }
    for timestamp in sorted(months):
        write_static_masks(region_id, timestamp=timestamp)


def _build_dataset_only_chunk_payload(
    chunk_index: int,
    region_id: str,
    request_payload: dict[str, object],
    timestamps_raw: list[str],
) -> dict[str, object]:
    request = HistoricalBackfillRequest.model_validate(request_payload)
    timestamps = [datetime.fromisoformat(value) for value in timestamps_raw]
    ensemble_members, particles_per_member = _resolve_baseline_fidelity(request)
    setup_started = perf_counter()
    _prewarm_static_masks(region_id, timestamps)
    setup_ms = (perf_counter() - setup_started) * 1000
    source_load_ms = 0.0
    baseline_ms = 0.0
    artifact_write_ms = 0.0
    total_started = perf_counter()
    run_payloads: list[dict[str, object]] = []
    artifact_payloads: list[dict[str, object]] = []

    for index, generated_at in enumerate(timestamps):
        source_started = perf_counter()
        context = load_operational_context(
            horizon_hours=72,
            seed=(index % 31) + 11,
            source_mode=request.source_mode,
            region_id=region_id,
            generated_at_override=generated_at,
        )
        source_load_ms += (perf_counter() - source_started) * 1000
        run_id = str(uuid4())
        selected_baseline_engine: str | None = None
        run_artifacts: list[BaselineArtifact] = []

        for frame in context.frames:
            currents_u = {point.cell_id: point.current_u for point in frame.grid}
            currents_v = {point.cell_id: point.current_v for point in frame.grid}
            winds_u = {point.cell_id: point.wind_u for point in frame.grid}
            winds_v = {point.cell_id: point.wind_v for point in frame.grid}
            for debris_class in request.debris_classes:
                baseline_started = perf_counter()
                baseline_engine, diagnostics = run_baseline_diagnostics(
                    DriftBaselineInput(
                        run_id=run_id,
                        generated_at=context.generated_at,
                        valid_at=frame.valid_at,
                        horizon_hour=frame.horizon_hour,
                        debris_class=debris_class,
                        ensemble_members=ensemble_members,
                        particles_per_member=particles_per_member,
                        source_strength=1.0,
                        grid=frame.grid,
                    )
                )
                baseline_ms += (perf_counter() - baseline_started) * 1000
                selected_baseline_engine = baseline_engine

                artifact_started = perf_counter()
                run_artifacts.append(
                    build_baseline_artifact(
                        region_id=context.region.id,
                        region_name=context.region.name,
                        run_id=run_id,
                        debris_class=debris_class,
                        generated_at=context.generated_at,
                        forecast_valid_at=frame.valid_at,
                        horizon_hour=frame.horizon_hour,
                        source_mode_requested=context.source_mode_requested,
                        source_mode_used=context.source_mode_used,
                        is_fallback=context.is_fallback,
                        source_notes=context.source_notes,
                        baseline_engine=baseline_engine,
                        current_u_by_cell=currents_u,
                        current_v_by_cell=currents_v,
                        wind_u_by_cell=winds_u,
                        wind_v_by_cell=winds_v,
                        diagnostics=diagnostics,
                        compact_tensors=True,
                    )
                )
                artifact_write_ms += (perf_counter() - artifact_started) * 1000

        run_payloads.append(
            _run_payload(
                request=request,
                run_id=run_id,
                context=context,
                baseline_engine=selected_baseline_engine,
                baseline_artifact_count=len(run_artifacts),
            )
        )
        artifact_payloads.extend(artifact.model_dump(mode="json") for artifact in run_artifacts)

    return {
        "chunk_index": chunk_index,
        "timing": {
            "region_id": region_id,
            "chunk_index": chunk_index,
            "max_workers": request.max_workers,
            "timestamps_in_chunk": len(timestamps),
            "runs_created": len(run_payloads),
            "baseline_artifacts_created": len(artifact_payloads),
            "setup_ms": round(setup_ms, 3),
            "source_load_ms": round(source_load_ms, 3),
            "baseline_ms": round(baseline_ms, 3),
            "artifact_write_ms": round(artifact_write_ms, 3),
            "db_write_ms": 0.0,
            "parallel_overhead_ms": 0.0,
            "total_ms": round((perf_counter() - total_started) * 1000, 3),
        },
        "run_payloads": run_payloads,
        "artifact_payloads": artifact_payloads,
    }


def _finalize_dataset_only_chunk_payload(
    *,
    payload: dict[str, object],
    db: Session,
) -> HistoricalBackfillTiming:
    timing_payload = dict(payload["timing"])
    run_payloads = list(payload["run_payloads"])
    raw_artifacts = list(payload["artifact_payloads"])
    artifact_write_ms = float(timing_payload["artifact_write_ms"])
    db_write_ms = 0.0
    total_started = perf_counter()

    manifest_started = perf_counter()
    artifacts = [BaselineArtifact.model_validate(item) for item in raw_artifacts]
    finalized_artifacts = write_baseline_manifests(artifacts)
    artifact_write_ms += (perf_counter() - manifest_started) * 1000

    artifacts_by_run: dict[str, list[BaselineArtifact]] = {}
    for artifact in finalized_artifacts:
        artifacts_by_run.setdefault(artifact.run_id, []).append(artifact)

    forecast_rows: list[ForecastRunModel] = []
    for run_payload in run_payloads:
        run_id = str(run_payload["id"])
        run_artifacts = artifacts_by_run.get(run_id, [])
        baseline_artifact_uri = run_artifacts[0].manifest_uri if run_artifacts else None
        forecast_rows.append(
            ForecastRunModel(
                id=run_id,
                generated_at=datetime.fromisoformat(str(run_payload["generated_at"])),
                horizon_hours=int(run_payload["horizon_hours"]),
                pilot_region=str(run_payload["pilot_region"]),
                source_mode_requested=str(run_payload["source_mode_requested"]),
                source_mode_used=str(run_payload["source_mode_used"]),
                is_fallback=bool(run_payload["is_fallback"]),
                source_notes=list(run_payload["source_notes"]),
                summary=_minimal_summary(
                    mode=str(run_payload["mode"]),
                    frame_count=int(run_payload["frame_count"]),
                    baseline_engine=str(run_payload["baseline_engine"]) if run_payload["baseline_engine"] else None,
                    baseline_artifact_uri=baseline_artifact_uri,
                    baseline_artifact_count=len(run_artifacts),
                ),
            )
        )

    db_started = perf_counter()
    if forecast_rows:
        db.add_all(forecast_rows)
    if finalized_artifacts:
        db.add_all([baseline_artifact_to_model(artifact) for artifact in finalized_artifacts])
    db.commit()
    db_write_ms = (perf_counter() - db_started) * 1000

    return HistoricalBackfillTiming(
        region_id=str(timing_payload["region_id"]),
        chunk_index=int(timing_payload["chunk_index"]),
        max_workers=int(timing_payload["max_workers"]) if timing_payload["max_workers"] else None,
        timestamps_in_chunk=int(timing_payload["timestamps_in_chunk"]),
        runs_created=len(forecast_rows),
        baseline_artifacts_created=len(finalized_artifacts),
        setup_ms=float(timing_payload["setup_ms"]),
        source_load_ms=float(timing_payload["source_load_ms"]),
        baseline_ms=float(timing_payload["baseline_ms"]),
        artifact_write_ms=round(artifact_write_ms, 3),
        db_write_ms=round(db_write_ms, 3),
        parallel_overhead_ms=float(timing_payload.get("parallel_overhead_ms", 0.0)),
        total_ms=round((perf_counter() - total_started) * 1000 + float(timing_payload["total_ms"]), 3),
    )


def _run_dataset_only_chunk(
    *,
    chunk_index: int,
    region_id: str,
    request: HistoricalBackfillRequest,
    timestamps: list[datetime],
    db: Session,
) -> HistoricalBackfillTiming:
    payload = _build_dataset_only_chunk_payload(
        chunk_index,
        region_id,
        request.model_dump(mode="json"),
        [timestamp.isoformat() for timestamp in timestamps],
    )
    return _finalize_dataset_only_chunk_payload(payload=payload, db=db)


def _run_live_parity_chunk(
    *,
    chunk_index: int,
    region_id: str,
    request: HistoricalBackfillRequest,
    timestamps: list[datetime],
    db: Session,
) -> HistoricalBackfillTiming:
    total_started = perf_counter()
    artifact_count_before = db.execute(select(func.count(BaselineArtifactModel.artifact_id))).scalar_one()
    runs_created = 0
    for index, generated_at in enumerate(timestamps):
        run_forecast(
            ForecastRunRequest(
                region_id=region_id,
                horizon_hours=72,
                debris_classes=request.debris_classes,
                source_strength=1.0,
                seed=(index % 31) + 11,
                source_mode=request.source_mode,
            ),
            db,
            generated_at_override=generated_at,
        )
        runs_created += 1
    artifact_count_after = db.execute(select(func.count(BaselineArtifactModel.artifact_id))).scalar_one()
    return HistoricalBackfillTiming(
        region_id=region_id,
        chunk_index=chunk_index,
        max_workers=None,
        timestamps_in_chunk=len(timestamps),
        runs_created=runs_created,
        baseline_artifacts_created=int(artifact_count_after - artifact_count_before),
        setup_ms=0.0,
        source_load_ms=0.0,
        baseline_ms=0.0,
        artifact_write_ms=0.0,
        db_write_ms=0.0,
        parallel_overhead_ms=0.0,
        total_ms=round((perf_counter() - total_started) * 1000, 3),
    )


def run_historical_backfill(request: HistoricalBackfillRequest, db: Session) -> HistoricalBackfillResponse:
    if request.region_id == "all":
        region_ids = list(REGIONS)
    else:
        region_ids = [request.region_id]

    runs_created = 0
    timestamps_planned = 0
    timestamps_processed = 0
    timings: list[HistoricalBackfillTiming] = []
    artifact_count_before = db.execute(select(func.count(BaselineArtifactModel.artifact_id))).scalar_one()

    for region_id in region_ids:
        existing = _existing_run_timestamps(region_id, db)
        planned_dates = [
            generated_at.replace(minute=0, second=0, microsecond=0)
            for generated_at in _backfill_dates(request.days)
            if generated_at.replace(minute=0, second=0, microsecond=0) not in existing
        ]
        timestamps_planned += len(planned_dates)
        chunks = _chunk_dates(planned_dates, request.chunk_days)
        if request.mode == "dataset_only" and request.max_workers and request.max_workers > 1 and len(chunks) > 1:
            _prewarm_static_masks(region_id, planned_dates)
            request_payload = request.model_dump(mode="json")
            with ProcessPoolExecutor(max_workers=request.max_workers) as executor:
                futures = [
                    executor.submit(
                        _build_dataset_only_chunk_payload,
                        chunk_index,
                        region_id,
                        request_payload,
                        [timestamp.isoformat() for timestamp in chunk],
                    )
                    for chunk_index, chunk in enumerate(chunks, start=1)
                    if chunk
                ]
                chunk_payloads = [future.result() for future in futures]
            for payload in sorted(chunk_payloads, key=lambda item: int(item["chunk_index"])):
                timing = _finalize_dataset_only_chunk_payload(payload=payload, db=db)
                timings.append(timing)
                runs_created += timing.runs_created
                timestamps_processed += timing.runs_created
        else:
            for chunk_index, chunk in enumerate(chunks, start=1):
                if not chunk:
                    continue
                if request.mode == "live_parity":
                    timing = _run_live_parity_chunk(
                        chunk_index=chunk_index,
                        region_id=region_id,
                        request=request,
                        timestamps=chunk,
                        db=db,
                    )
                else:
                    timing = _run_dataset_only_chunk(
                        chunk_index=chunk_index,
                        region_id=region_id,
                        request=request,
                        timestamps=chunk,
                        db=db,
                    )
                timings.append(timing)
                runs_created += timing.runs_created
                timestamps_processed += timing.runs_created
                existing.update(chunk)

    artifact_count_after = db.execute(select(func.count(BaselineArtifactModel.artifact_id))).scalar_one()
    ready_run_count = db.execute(
        select(func.count(ForecastRunModel.id)).where(ForecastRunModel.pilot_region.in_(region_ids))
    ).scalar_one()
    return HistoricalBackfillResponse(
        region_id=request.region_id,
        source_mode=request.source_mode,
        days_backfilled=request.days,
        mode=request.mode,
        max_workers=request.max_workers,
        runs_created=runs_created,
        baseline_artifacts_created=int(artifact_count_after - artifact_count_before),
        dataset_ready_run_count=int(ready_run_count),
        timestamps_planned=timestamps_planned,
        timestamps_processed=timestamps_processed,
        chunk_count=len(timings),
        timings=timings,
    )

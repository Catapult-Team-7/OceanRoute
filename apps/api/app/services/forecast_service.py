from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ForecastRunModel, ForecastStepModel, ObservationModel
from app.schemas import (
    DEBRIS_CLASS_METADATA,
    DriftBaselineInput,
    ForecastProvenance,
    ForecastRunRequest,
    ForecastRunResponse,
    ForecastSnapshot,
    ForecastStep,
    HotspotSummary,
    OperationalGridFrame,
    ResidualModelInput,
)
from app.services.artifact_service import write_latest_snapshot
from app.services.ingest_service import load_operational_context
from app.services.physics_service import run_drift_baseline
from app.services.provenance_service import build_forecast_provenance
from app.services.residual_model_service import apply_residual_correction
from app.services.uncertainty_service import build_uncertainty


def _class_weight_range(label: str) -> tuple[float, float]:
    return (4.6, 8.4) if label == "low" else (7.5, 13.5)


def _nearest_cell_id(lat: float, lon: float, frame: OperationalGridFrame) -> str:
    best_cell_id = frame.grid[0].cell_id
    best_distance = float("inf")
    for point in frame.grid:
        distance = ((lat - point.lat) ** 2) + ((lon - point.lon) ** 2)
        if distance < best_distance:
            best_distance = distance
            best_cell_id = point.cell_id
    return best_cell_id


def _history_bias(frame: OperationalGridFrame, db: Session) -> dict[str, float]:
    observations = db.execute(
        select(ObservationModel).order_by(ObservationModel.observed_at.desc()).limit(250)
    ).scalars()
    bias = {point.cell_id: 0.0 for point in frame.grid}
    counts = {point.cell_id: 0 for point in frame.grid}
    for observation in observations:
        cell_id = _nearest_cell_id(observation.lat, observation.lon, frame)
        delta = 1.0 if observation.found_status == "found" else -0.7
        bias[cell_id] += delta * max(observation.confidence, 0.2)
        counts[cell_id] += 1
    for cell_id, total in bias.items():
        sample_count = counts[cell_id]
        if sample_count == 0:
            bias[cell_id] = 0.0
            continue
        bias[cell_id] = round(total / sample_count, 4)
    return bias


def _top_hotspots(steps: list[ForecastStep], limit: int = 8) -> list[HotspotSummary]:
    ranked = sorted(steps, key=lambda item: (item.expected_kg_max, item.confidence), reverse=True)
    return [
        HotspotSummary(
            valid_at=item.valid_at,
            horizon_hour=item.horizon_hour,
            cell_id=item.cell_id,
            lat=item.lat,
            lon=item.lon,
            debris_class=item.debris_class,
            probability=item.probability,
            expected_kg_min=item.expected_kg_min,
            expected_kg_max=item.expected_kg_max,
            confidence=item.confidence,
        )
        for item in ranked[:limit]
    ]


def _summary_for_steps(steps: list[ForecastStep], source_mode_used: str) -> dict[str, float | int | str]:
    if not steps:
        return {
            "step_count": 0,
            "top_expected_kg_max": 0.0,
            "mean_confidence": 0.0,
            "source_mode_used": source_mode_used,
        }
    return {
        "step_count": len(steps),
        "top_expected_kg_max": round(max(item.expected_kg_max for item in steps), 3),
        "mean_confidence": round(sum(item.confidence for item in steps) / len(steps), 4),
        "source_mode_used": source_mode_used,
    }


def _step_from_model(model: ForecastStepModel) -> ForecastStep:
    return ForecastStep(
        valid_at=model.valid_at,
        horizon_hour=model.horizon_hour,
        cell_id=model.cell_id,
        lat=model.lat,
        lon=model.lon,
        debris_class=model.debris_class,  # type: ignore[arg-type]
        probability=model.probability,
        expected_kg_min=model.expected_kg_min,
        expected_kg_max=model.expected_kg_max,
        uncertainty=model.uncertainty,
        confidence=model.confidence,
        beaching_risk=model.beaching_risk,
        restricted=model.restricted,
    )


def _provenance_from_run(run: ForecastRunModel) -> ForecastProvenance:
    return build_forecast_provenance(
        generated_at=run.generated_at,
        horizon_hours=run.horizon_hours,
        source_mode_requested=run.source_mode_requested,
        source_mode_used=run.source_mode_used,
        is_fallback=run.is_fallback,
        source_notes=list(run.source_notes),
    )


def _snapshot_from_run(run: ForecastRunModel, steps: list[ForecastStep]) -> ForecastSnapshot:
    provenance = _provenance_from_run(run)
    return ForecastSnapshot(
        run_id=run.id,
        generated_at=run.generated_at,
        horizon_hours=run.horizon_hours,
        pilot_region=run.pilot_region,
        source_mode_requested=run.source_mode_requested,  # type: ignore[arg-type]
        source_mode_used=run.source_mode_used,  # type: ignore[arg-type]
        is_fallback=run.is_fallback,
        is_stale=provenance.is_stale,
        age_minutes=provenance.age_minutes,
        stale_after_minutes=provenance.stale_after_minutes,
        source_notes=list(run.source_notes),
        debris_classes=[
            {
                "code": debris_class_key,
                "label": str(meta["label"]),
                "description": str(meta["description"]),
                "windage_factor": float(meta["windage_factor"]),
            }
            for debris_class_key, meta in DEBRIS_CLASS_METADATA.items()
        ],
        steps=steps,
        top_hotspots=_top_hotspots(steps),
        provenance=provenance,
        summary=_summary_for_steps(steps, run.source_mode_used),
    )


def run_forecast(request: ForecastRunRequest, db: Session) -> ForecastRunResponse:
    context = load_operational_context(
        horizon_hours=request.horizon_hours,
        seed=request.seed,
        source_mode=request.source_mode,
    )
    run_id = str(uuid4())
    step_models: list[ForecastStepModel] = []
    steps: list[ForecastStep] = []

    for frame in context.frames:
        history_bias = _history_bias(frame, db)
        shoreline = {point.cell_id: point.shoreline_proximity for point in frame.grid}
        currents = {point.cell_id: (point.current_u, point.current_v) for point in frame.grid}
        winds = {point.cell_id: (point.wind_u, point.wind_v) for point in frame.grid}

        for debris_class in request.debris_classes:
            baseline_input = DriftBaselineInput(
                run_id=run_id,
                generated_at=context.generated_at,
                valid_at=frame.valid_at,
                horizon_hour=frame.horizon_hour,
                debris_class=debris_class,
                source_strength=request.source_strength,
                grid=frame.grid,
            )
            baseline = run_drift_baseline(baseline_input)
            corrected = apply_residual_correction(
                ResidualModelInput(
                    run_id=run_id,
                    debris_class=debris_class,
                    horizon_hour=frame.horizon_hour,
                    baseline_density=baseline,
                    currents=currents,
                    winds=winds,
                    history_bias=history_bias,
                    shoreline=shoreline,
                )
            )
            uncertainty, confidence = build_uncertainty(
                frame.grid,
                corrected,
                horizon_hour=frame.horizon_hour,
                source_mode_used=context.source_mode_used,
            )
            max_density = max(corrected.values()) if corrected else 1.0
            kg_min_multiplier, kg_max_multiplier = _class_weight_range(debris_class)

            for point in frame.grid:
                density = corrected[point.cell_id]
                probability = max(0.0, min(1.0, density / (max_density + 1e-6)))
                expected_kg_min = probability * kg_min_multiplier * (1.05 - uncertainty[point.cell_id])
                expected_kg_max = probability * kg_max_multiplier * (1.14 - (uncertainty[point.cell_id] * 0.45))
                step = ForecastStep(
                    valid_at=frame.valid_at,
                    horizon_hour=frame.horizon_hour,
                    cell_id=point.cell_id,
                    lat=point.lat,
                    lon=point.lon,
                    debris_class=debris_class,
                    probability=round(probability, 4),
                    expected_kg_min=round(max(0.0, expected_kg_min), 3),
                    expected_kg_max=round(max(expected_kg_min, expected_kg_max), 3),
                    uncertainty=uncertainty[point.cell_id],
                    confidence=confidence[point.cell_id],
                    beaching_risk=round(min(1.0, point.shoreline_proximity * 0.72), 4),
                    restricted=point.restricted,
                )
                steps.append(step)
                step_models.append(
                    ForecastStepModel(
                        run_id=run_id,
                        valid_at=step.valid_at,
                        horizon_hour=step.horizon_hour,
                        cell_id=step.cell_id,
                        lat=step.lat,
                        lon=step.lon,
                        debris_class=step.debris_class,
                        probability=step.probability,
                        expected_kg_min=step.expected_kg_min,
                        expected_kg_max=step.expected_kg_max,
                        uncertainty=step.uncertainty,
                        confidence=step.confidence,
                        beaching_risk=step.beaching_risk,
                        restricted=step.restricted,
                    )
                )

    top_hotspots = _top_hotspots(steps)
    summary = _summary_for_steps(steps, context.source_mode_used)

    db.add(
        ForecastRunModel(
            id=run_id,
            generated_at=context.generated_at,
            horizon_hours=request.horizon_hours,
            pilot_region=context.pilot_region,
            source_mode_requested=context.source_mode_requested,
            source_mode_used=context.source_mode_used,
            is_fallback=context.is_fallback,
            source_notes=context.source_notes,
            summary=summary,
        )
    )
    db.add_all(step_models)
    db.commit()

    provenance = build_forecast_provenance(
        generated_at=context.generated_at,
        horizon_hours=request.horizon_hours,
        source_mode_requested=context.source_mode_requested,
        source_mode_used=context.source_mode_used,
        is_fallback=context.is_fallback,
        source_notes=context.source_notes,
    )
    snapshot = ForecastSnapshot(
        run_id=run_id,
        generated_at=context.generated_at,
        horizon_hours=request.horizon_hours,
        pilot_region=context.pilot_region,
        source_mode_requested=context.source_mode_requested,
        source_mode_used=context.source_mode_used,
        is_fallback=context.is_fallback,
        is_stale=provenance.is_stale,
        age_minutes=provenance.age_minutes,
        stale_after_minutes=provenance.stale_after_minutes,
        source_notes=context.source_notes,
        debris_classes=[
            {
                "code": debris_class,
                "label": str(DEBRIS_CLASS_METADATA[debris_class]["label"]),
                "description": str(DEBRIS_CLASS_METADATA[debris_class]["description"]),
                "windage_factor": float(DEBRIS_CLASS_METADATA[debris_class]["windage_factor"]),
            }
            for debris_class in request.debris_classes
        ],
        steps=steps,
        top_hotspots=top_hotspots,
        provenance=provenance,
        summary=summary,
    )
    write_latest_snapshot(snapshot.model_dump(mode="json"))
    return ForecastRunResponse(
        run_id=run_id,
        generated_at=context.generated_at,
        horizon_hours=request.horizon_hours,
        source_mode_requested=context.source_mode_requested,
        source_mode_used=context.source_mode_used,
        is_fallback=context.is_fallback,
        is_stale=provenance.is_stale,
        age_minutes=provenance.age_minutes,
        stale_after_minutes=provenance.stale_after_minutes,
        steps_generated=len(steps),
        top_hotspots=top_hotspots,
        provenance=provenance,
        summary=summary,
    )


def latest_forecast(
    db: Session,
    *,
    horizon_hour: int | None = None,
    debris_class: str | None = None,
    min_confidence: float = 0.0,
) -> ForecastSnapshot | None:
    run = db.execute(
        select(ForecastRunModel).order_by(ForecastRunModel.generated_at.desc()).limit(1)
    ).scalar_one_or_none()
    if run is None:
        return None

    step_query = select(ForecastStepModel).where(ForecastStepModel.run_id == run.id)
    if horizon_hour is not None:
        step_query = step_query.where(ForecastStepModel.horizon_hour == horizon_hour)
    if debris_class is not None:
        step_query = step_query.where(ForecastStepModel.debris_class == debris_class)
    if min_confidence > 0:
        step_query = step_query.where(ForecastStepModel.confidence >= min_confidence)

    step_models = db.execute(
        step_query.order_by(ForecastStepModel.horizon_hour, ForecastStepModel.expected_kg_max.desc())
    ).scalars().all()
    steps = [_step_from_model(model) for model in step_models]
    return _snapshot_from_run(run, steps)


def require_latest_forecast(
    db: Session,
    *,
    horizon_hour: int | None = None,
    debris_class: str | None = None,
    min_confidence: float = 0.0,
) -> ForecastSnapshot:
    snapshot = latest_forecast(
        db,
        horizon_hour=horizon_hour,
        debris_class=debris_class,
        min_confidence=min_confidence,
    )
    if snapshot is None:
        raise LookupError(
            f"No forecast has been generated yet for pilot region {settings.pilot_region}. Run /forecast/run first."
        )
    return snapshot

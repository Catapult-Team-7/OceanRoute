from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.ml.runtime_feature_service import build_runtime_feature_snapshot, resolve_prediction_horizons
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
    InferencePredictRequest,
    OperationalGridFrame,
    ResidualModelInput,
)
from app.services.artifact_service import write_latest_snapshot
from app.services.baseline_runtime_service import latest_baseline_artifact, persist_baseline_artifact, run_baseline_diagnostics
from app.services.data_lake_service import read_tensor
from app.services.ingest_service import load_operational_context
from app.services.inference_client_service import predict_with_inference_service
from app.services.model_registry_service import (
    apply_active_model_adjustments,
    get_active_model_entry,
    get_active_model_payload,
)
from app.services.provenance_service import build_forecast_provenance
from app.services.region_service import get_region_info
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


def _summary_for_steps(
    steps: list[ForecastStep],
    source_mode_used: str,
    active_model: dict[str, object] | None = None,
    baseline_engine: str | None = None,
) -> dict[str, object]:
    if not steps:
        summary: dict[str, object] = {
            "step_count": 0,
            "top_expected_kg_max": 0.0,
            "mean_confidence": 0.0,
            "source_mode_used": source_mode_used,
        }
        if baseline_engine:
            summary["baseline_engine"] = baseline_engine
        if active_model is not None and active_model.get("resolved_model_id"):
            summary["resolved_model_id"] = str(active_model["resolved_model_id"])
            summary["active_model_id"] = str(active_model["resolved_model_id"])
        if active_model is not None and active_model.get("training_scope"):
            summary["training_scope"] = str(active_model["training_scope"])
        if active_model is not None and active_model.get("requested_model_id"):
            summary["requested_model_id"] = str(active_model["requested_model_id"])
        if active_model is not None and active_model.get("model_stage"):
            summary["model_stage"] = str(active_model["model_stage"])
        if active_model is not None and active_model.get("used_candidate_override") is not None:
            summary["used_candidate_override"] = bool(active_model["used_candidate_override"])
        if active_model is not None and active_model.get("used_inference_fallback") is not None:
            summary["used_inference_fallback"] = bool(active_model["used_inference_fallback"])
        return summary
    summary = {
        "step_count": len(steps),
        "top_expected_kg_max": round(max(item.expected_kg_max for item in steps), 3),
        "mean_confidence": round(sum(item.confidence for item in steps) / len(steps), 4),
        "mean_beaching_fraction": round(sum(item.beaching_fraction for item in steps) / len(steps), 4),
        "max_ensemble_spread": round(max(item.ensemble_spread for item in steps), 4),
        "source_mode_used": source_mode_used,
    }
    if baseline_engine:
        summary["baseline_engine"] = baseline_engine
    if active_model is not None and active_model.get("resolved_model_id"):
        summary["resolved_model_id"] = str(active_model["resolved_model_id"])
        summary["active_model_id"] = str(active_model["resolved_model_id"])
    if active_model is not None and active_model.get("training_scope"):
        summary["training_scope"] = str(active_model["training_scope"])
    if active_model is not None and active_model.get("requested_model_id"):
        summary["requested_model_id"] = str(active_model["requested_model_id"])
    if active_model is not None and active_model.get("model_stage"):
        summary["model_stage"] = str(active_model["model_stage"])
    if active_model is not None and active_model.get("used_candidate_override") is not None:
        summary["used_candidate_override"] = bool(active_model["used_candidate_override"])
    if active_model is not None and active_model.get("used_inference_fallback") is not None:
        summary["used_inference_fallback"] = bool(active_model["used_inference_fallback"])
    return summary


def _apply_prediction_to_steps(
    steps: list[ForecastStep],
    *,
    prediction_prob: np.ndarray,
    prediction_kg: np.ndarray,
    prediction_uncertainty: np.ndarray | None,
    cell_map: dict[str, list[int]] | dict[str, tuple[int, int]],
    target_horizon: int,
    debris_class: str,
) -> list[ForecastStep]:
    updated: list[ForecastStep] = []
    probability_slice = prediction_prob[0] if prediction_prob.ndim == 3 else prediction_prob
    kg_slice = prediction_kg[0] if prediction_kg.ndim == 3 else prediction_kg
    uncertainty_slice = None
    if prediction_uncertainty is not None:
        uncertainty_slice = prediction_uncertainty[0] if prediction_uncertainty.ndim == 3 else prediction_uncertainty
    for step in steps:
        if step.horizon_hour == target_horizon and step.debris_class == debris_class and step.cell_id in cell_map:
            row, col = cell_map[step.cell_id]
            probability = float(np.clip(probability_slice[row, col], 0.0, 1.0))
            expected_mid = float(max(kg_slice[row, col], 0.0))
            updated_uncertainty = step.uncertainty if uncertainty_slice is None else float(np.clip(uncertainty_slice[row, col], 0.0, 1.0))
            updated.append(
                step.model_copy(
                    update={
                        "probability": round(probability, 4),
                        "expected_kg_min": round(max(0.0, expected_mid * 0.72), 3),
                        "expected_kg_max": round(max(expected_mid * 0.72, expected_mid * 1.18), 3),
                        "uncertainty": round(updated_uncertainty, 4),
                        "confidence": round(max(0.0, 1.0 - updated_uncertainty), 4),
                    }
                )
            )
        else:
            updated.append(step)
    return updated


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
        baseline_density=model.baseline_density,
        ensemble_spread=model.ensemble_spread,
        beaching_fraction=model.beaching_fraction,
        stokes_drift_u=model.stokes_drift_u,
        stokes_drift_v=model.stokes_drift_v,
        windage_fraction=model.windage_fraction,
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
        baseline_engine=str(run.summary.get("baseline_engine")) if run.summary.get("baseline_engine") else None,
        baseline_artifact_uri=str(run.summary.get("baseline_artifact_uri")) if run.summary.get("baseline_artifact_uri") else None,
        requested_model_id=str(run.summary.get("requested_model_id")) if run.summary.get("requested_model_id") else None,
        resolved_model_id=str(run.summary.get("resolved_model_id")) if run.summary.get("resolved_model_id") else None,
        model_id=str(run.summary.get("active_model_id")) if run.summary.get("active_model_id") else None,
        model_architecture=str(run.summary.get("model_architecture")) if run.summary.get("model_architecture") else None,
        model_dataset_version=str(run.summary.get("model_dataset_version")) if run.summary.get("model_dataset_version") else None,
        model_stage=str(run.summary.get("model_stage")) if run.summary.get("model_stage") else None,
        training_scope=str(run.summary.get("training_scope")) if run.summary.get("training_scope") else None,
        used_candidate_override=bool(run.summary.get("used_candidate_override", False)),
        used_inference_fallback=bool(run.summary.get("used_inference_fallback", False)),
        inference_service_version=str(run.summary.get("inference_service_version")) if run.summary.get("inference_service_version") else None,
        prediction_artifact_uri=str(run.summary.get("prediction_artifact_uri")) if run.summary.get("prediction_artifact_uri") else None,
    )


def _snapshot_from_run(run: ForecastRunModel, steps: list[ForecastStep]) -> ForecastSnapshot:
    provenance = _provenance_from_run(run)
    summary = _summary_for_steps(
        steps,
        run.source_mode_used,
        {
            "model_id": run.summary.get("active_model_id"),
            "resolved_model_id": run.summary.get("resolved_model_id"),
            "requested_model_id": run.summary.get("requested_model_id"),
            "model_stage": run.summary.get("model_stage"),
            "training_scope": run.summary.get("training_scope"),
            "used_candidate_override": run.summary.get("used_candidate_override", False),
            "used_inference_fallback": run.summary.get("used_inference_fallback", False),
        }
        if run.summary.get("active_model_id")
        or run.summary.get("requested_model_id")
        else None,
        baseline_engine=str(run.summary.get("baseline_engine")) if run.summary.get("baseline_engine") else None,
    )
    for key in (
        "active_model_id",
        "model_architecture",
        "model_dataset_version",
        "training_scope",
        "requested_model_id",
        "resolved_model_id",
        "model_stage",
        "used_candidate_override",
        "used_inference_fallback",
        "prediction_artifact_uri",
        "inference_service_version",
        "baseline_artifact_uri",
    ):
        if run.summary.get(key):
            summary[key] = str(run.summary[key])
    return ForecastSnapshot(
        run_id=run.id,
        generated_at=run.generated_at,
        horizon_hours=run.horizon_hours,
        pilot_region=run.pilot_region,
        region=get_region_info(run.pilot_region),
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
        summary=summary,
    )


def run_forecast(
    request: ForecastRunRequest,
    db: Session,
    *,
    generated_at_override: datetime | None = None,
) -> ForecastRunResponse:
    context = load_operational_context(
        horizon_hours=request.horizon_hours,
        seed=request.seed,
        source_mode=request.source_mode,
        region_id=request.region_id,
        generated_at_override=generated_at_override,
    )
    run_id = str(uuid4())
    steps: list[ForecastStep] = []
    baseline_artifacts: dict[tuple[str, int], str] = {}
    available_prediction_horizons = sorted({frame.horizon_hour for frame in context.frames})
    active_model_entry = get_active_model_entry(db, context.region.id, model_id=request.model_id)
    if request.model_id and active_model_entry is None:
        raise LookupError(f"Requested model {request.model_id} was not found or is not compatible with region {context.region.id}.")
    active_model_contract_payload = (
        get_active_model_payload(db, context.region.id, model_id=active_model_entry.model_id)
        if active_model_entry is not None
        else None
    )
    active_model_payload: dict[str, object] | None = None
    selected_baseline_engine: str | None = None
    used_inference_fallback = False
    used_candidate_override = bool(request.model_id and active_model_entry is not None and active_model_entry.stage == "candidate")

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
            baseline_engine, baseline = run_baseline_diagnostics(baseline_input)
            selected_baseline_engine = baseline_engine
            artifact = persist_baseline_artifact(
                db=db,
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
                current_u_by_cell={point.cell_id: point.current_u for point in frame.grid},
                current_v_by_cell={point.cell_id: point.current_v for point in frame.grid},
                wind_u_by_cell={point.cell_id: point.wind_u for point in frame.grid},
                wind_v_by_cell={point.cell_id: point.wind_v for point in frame.grid},
                diagnostics=baseline,
            )
            baseline_artifacts[(debris_class, frame.horizon_hour)] = artifact.manifest_uri
            corrected = apply_residual_correction(
                ResidualModelInput(
                    run_id=run_id,
                    debris_class=debris_class,
                    horizon_hour=frame.horizon_hour,
                    baseline_density=baseline.density,
                    currents=currents,
                    winds=winds,
                    history_bias=history_bias,
                    shoreline=shoreline,
                    ensemble_spread=baseline.ensemble_spread,
                    beaching_fraction=baseline.beaching_fraction,
                )
            )
            uncertainty, confidence = build_uncertainty(
                frame.grid,
                corrected,
                horizon_hour=frame.horizon_hour,
                source_mode_used=context.source_mode_used,
                ensemble_spread=baseline.ensemble_spread,
                beaching_fraction=baseline.beaching_fraction,
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
                    beaching_risk=round(min(1.0, (point.shoreline_proximity * 0.6) + (baseline.beaching_fraction[point.cell_id] * 0.4)), 4),
                    baseline_density=baseline.density[point.cell_id],
                    ensemble_spread=baseline.ensemble_spread[point.cell_id],
                    beaching_fraction=baseline.beaching_fraction[point.cell_id],
                    stokes_drift_u=round(baseline.stokes_drift[point.cell_id][0], 4),
                    stokes_drift_v=round(baseline.stokes_drift[point.cell_id][1], 4),
                    windage_fraction=baseline.windage_fraction,
                    restricted=point.restricted,
                )
                steps.append(step)

    prediction_artifact_uri: str | None = None
    inference_service_version: str | None = None
    if active_model_entry is not None and active_model_entry.architecture in {"temporal_unet", "convlstm"}:
        for debris_class in request.debris_classes:
            try:
                if active_model_contract_payload is None:
                    raise ValueError(
                        f"Requested model {active_model_entry.model_id} did not have a readable runtime feature contract."
                    )
                feature_artifact_uri, _, runtime_contract = build_runtime_feature_snapshot(
                    model_payload=active_model_contract_payload,
                    region_id=context.region.id,
                    forecast_run_id=run_id,
                    debris_class=debris_class,
                    generated_at=context.generated_at,
                    db=db,
                )
                prediction_horizons = resolve_prediction_horizons(
                    trained_horizons=list(runtime_contract["trained_horizons"]),
                    available_horizons=available_prediction_horizons,
                )
                if not prediction_horizons:
                    raise ValueError(
                        f"Requested model {active_model_entry.model_id} is not compatible with forecast horizons "
                        f"{available_prediction_horizons}. Trained horizons: {runtime_contract['trained_horizons']}."
                    )
                prediction = predict_with_inference_service(
                    InferencePredictRequest(
                        region_id=context.region.id,
                        forecast_run_id=run_id,
                        debris_class=debris_class,
                        feature_artifact_uri=feature_artifact_uri,
                        model_id=active_model_entry.model_id,
                        target_horizons=prediction_horizons,
                    ),
                    db,
                )
                if request.model_id and prediction.used_fallback:
                    raise ValueError(
                        f"Requested model {request.model_id} could not be used for inference without fallback."
                    )
                used_inference_fallback = used_inference_fallback or prediction.used_fallback
                prediction_prob = read_tensor(prediction.artifact.hotspot_probability_uri)
                prediction_kg = read_tensor(prediction.artifact.expected_kg_uri)
                prediction_uncertainty = read_tensor(prediction.artifact.uncertainty_uri) if prediction.artifact.uncertainty_uri else None
                horizon_lookup = {horizon: index for index, horizon in enumerate(prediction.artifact.target_horizons)}
                for target_horizon in prediction.artifact.target_horizons:
                    baseline_artifact = latest_baseline_artifact(db, run_id=run_id, debris_class=debris_class, horizon_hour=target_horizon)
                    if baseline_artifact is None:
                        continue
                    horizon_index = horizon_lookup[target_horizon]
                    steps = _apply_prediction_to_steps(
                        steps,
                        prediction_prob=prediction_prob[horizon_index],
                        prediction_kg=prediction_kg[horizon_index],
                        prediction_uncertainty=None if prediction_uncertainty is None else prediction_uncertainty[horizon_index],
                        cell_map=baseline_artifact.metadata.get("cell_map", {}),
                        target_horizon=target_horizon,
                        debris_class=debris_class,
                    )
                prediction_artifact_uri = prediction.artifact.manifest_uri
                inference_service_version = prediction.artifact.inference_service_version
                active_model_payload = {
                    "requested_model_id": request.model_id,
                    "resolved_model_id": prediction.loaded_model_id,
                    "model_id": prediction.loaded_model_id,
                    "architecture": active_model_entry.architecture,
                    "dataset_version": active_model_entry.dataset_version or settings.dataset_version,
                    "model_stage": active_model_entry.stage,
                    "training_scope": prediction.artifact.training_scope or active_model_entry.training_scope,
                    "trained_horizons": list(runtime_contract["trained_horizons"]),
                    "lookback_hours": int(runtime_contract["lookback_hours"]),
                    "used_candidate_override": used_candidate_override,
                    "used_inference_fallback": prediction.used_fallback,
                }
            except Exception as exc:
                if request.model_id:
                    raise ValueError(
                        f"Requested model {request.model_id} could not be used for inference: {exc}"
                    ) from exc
                used_inference_fallback = True
                continue

    steps, linear_payload = apply_active_model_adjustments(
        steps,
        context.region.id,
        db,
        model_id=request.model_id,
    )
    if active_model_payload is None:
        active_model_payload = linear_payload
    if active_model_payload is not None:
        active_model_payload = {
            **active_model_payload,
            "requested_model_id": request.model_id,
            "resolved_model_id": active_model_payload.get("resolved_model_id") or active_model_payload.get("model_id"),
            "used_candidate_override": used_candidate_override,
            "used_inference_fallback": used_inference_fallback,
        }
    step_models = [
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
            baseline_density=step.baseline_density,
            ensemble_spread=step.ensemble_spread,
            beaching_fraction=step.beaching_fraction,
            stokes_drift_u=step.stokes_drift_u,
            stokes_drift_v=step.stokes_drift_v,
            windage_fraction=step.windage_fraction,
            restricted=step.restricted,
        )
        for step in steps
    ]

    top_hotspots = _top_hotspots(steps)
    summary = _summary_for_steps(steps, context.source_mode_used, active_model_payload, selected_baseline_engine)
    if baseline_artifacts:
        summary["baseline_artifact_uri"] = next(iter(baseline_artifacts.values()))
    if active_model_payload and active_model_payload.get("architecture"):
        summary["model_architecture"] = str(active_model_payload["architecture"])
    if active_model_payload and active_model_payload.get("dataset_version"):
        summary["model_dataset_version"] = str(active_model_payload["dataset_version"])
    if active_model_payload and active_model_payload.get("resolved_model_id"):
        summary["resolved_model_id"] = str(active_model_payload["resolved_model_id"])
    if active_model_payload and active_model_payload.get("training_scope"):
        summary["training_scope"] = str(active_model_payload["training_scope"])
    if active_model_payload and active_model_payload.get("requested_model_id"):
        summary["requested_model_id"] = str(active_model_payload["requested_model_id"])
    if active_model_payload and active_model_payload.get("model_stage"):
        summary["model_stage"] = str(active_model_payload["model_stage"])
    summary["used_candidate_override"] = used_candidate_override
    summary["used_inference_fallback"] = used_inference_fallback
    if prediction_artifact_uri:
        summary["prediction_artifact_uri"] = prediction_artifact_uri
    if inference_service_version:
        summary["inference_service_version"] = inference_service_version

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
        baseline_engine=selected_baseline_engine,
        baseline_artifact_uri=next(iter(baseline_artifacts.values())) if baseline_artifacts else None,
        requested_model_id=request.model_id,
        resolved_model_id=str(active_model_payload["resolved_model_id"]) if active_model_payload and active_model_payload.get("resolved_model_id") else None,
        model_id=str(active_model_payload["model_id"]) if active_model_payload and active_model_payload.get("model_id") else None,
        model_architecture=str(active_model_payload["architecture"]) if active_model_payload and active_model_payload.get("architecture") else None,
        model_dataset_version=str(active_model_payload["dataset_version"]) if active_model_payload and active_model_payload.get("dataset_version") else None,
        model_stage=str(active_model_payload["model_stage"]) if active_model_payload and active_model_payload.get("model_stage") else None,
        training_scope=str(active_model_payload["training_scope"]) if active_model_payload and active_model_payload.get("training_scope") else None,
        used_candidate_override=used_candidate_override,
        used_inference_fallback=used_inference_fallback,
        inference_service_version=inference_service_version,
        prediction_artifact_uri=prediction_artifact_uri,
    )
    snapshot = ForecastSnapshot(
        run_id=run_id,
        generated_at=context.generated_at,
        horizon_hours=request.horizon_hours,
        pilot_region=context.pilot_region,
        region=context.region,
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
        region=context.region,
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
    region_id: str | None = None,
    horizon_hour: int | None = None,
    debris_class: str | None = None,
    min_confidence: float = 0.0,
) -> ForecastSnapshot | None:
    run_query = select(ForecastRunModel).join(ForecastStepModel, ForecastStepModel.run_id == ForecastRunModel.id).distinct()
    if region_id is not None:
        run_query = run_query.where(ForecastRunModel.pilot_region == region_id)
    run = db.execute(run_query.order_by(ForecastRunModel.generated_at.desc()).limit(1)).scalar_one_or_none()
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
    region_id: str | None = None,
    horizon_hour: int | None = None,
    debris_class: str | None = None,
    min_confidence: float = 0.0,
) -> ForecastSnapshot:
    snapshot = latest_forecast(
        db,
        region_id=region_id,
        horizon_hour=horizon_hour,
        debris_class=debris_class,
        min_confidence=min_confidence,
    )
    if snapshot is None:
        requested_region = region_id or settings.pilot_region
        raise LookupError(
            f"No forecast has been generated yet for region {requested_region}. Run /forecast/run first."
        )
    return snapshot

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import BaselineArtifactModel
from app.schemas import BaselineArtifact, BaselineEngine, ForcingRefs
from app.services.data_lake_service import write_baseline_manifest, write_baseline_tensors
from app.services.physics_service import DriftBaselineDiagnostics, run_drift_baseline_ensemble
from app.services.pygnome_adapter_service import PYGNOME_AVAILABLE, run_pygnome_compatible_baseline


def resolve_baseline_engine(preferred: str | None = None) -> BaselineEngine:
    choice = (preferred or settings.baseline_engine_preference or "auto").lower()
    if choice == "pygnome" and PYGNOME_AVAILABLE:
        return "pygnome"
    if choice == "custom_particle":
        return "custom_particle"
    if choice == "auto" and PYGNOME_AVAILABLE:
        return "pygnome"
    return "custom_particle"


def _run_engine(engine: BaselineEngine, payload) -> DriftBaselineDiagnostics:
    if engine == "pygnome":
        return run_pygnome_compatible_baseline(payload)
    return run_drift_baseline_ensemble(payload)


def run_baseline_diagnostics(payload, *, preferred_engine: str | None = None) -> tuple[BaselineEngine, DriftBaselineDiagnostics]:
    engine = resolve_baseline_engine(preferred_engine)
    return engine, _run_engine(engine, payload)


def build_forcing_refs(region_name: str, *, source_mode_used: str) -> ForcingRefs:
    source_prefix = "NOAA/IOOS live" if source_mode_used == "live" else "deterministic sample"
    return ForcingRefs(
        current_source=f"{source_prefix} currents for {region_name}",
        wind_source=f"{source_prefix} winds for {region_name}",
        wave_source=f"{source_prefix} Stokes proxy for {region_name}",
        drifter_source="Global Drifter Program aligned labels",
        shoreline_source="Regional shoreline mask",
        bathymetry_source="Regional bathymetry mask",
    )


def build_baseline_artifact(
    *,
    region_id: str,
    region_name: str,
    run_id: str,
    debris_class: str,
    generated_at: datetime,
    forecast_valid_at: datetime,
    horizon_hour: int,
    source_mode_requested: str,
    source_mode_used: str,
    is_fallback: bool,
    source_notes: list[str],
    baseline_engine: BaselineEngine,
    current_u_by_cell: dict[str, float],
    current_v_by_cell: dict[str, float],
    wind_u_by_cell: dict[str, float],
    wind_v_by_cell: dict[str, float],
    diagnostics: DriftBaselineDiagnostics,
) -> BaselineArtifact:
    grid_spec, tensor_uris, cell_map = write_baseline_tensors(
        region_id=region_id,
        generated_at=generated_at,
        run_id=run_id,
        debris_class=debris_class,
        horizon_hour=horizon_hour,
        density_by_cell=diagnostics.density,
        current_u_by_cell=current_u_by_cell,
        current_v_by_cell=current_v_by_cell,
        wind_u_by_cell=wind_u_by_cell,
        wind_v_by_cell=wind_v_by_cell,
        ensemble_spread_by_cell=diagnostics.ensemble_spread,
        beaching_fraction_by_cell=diagnostics.beaching_fraction,
        stokes_u_by_cell={cell_id: values[0] for cell_id, values in diagnostics.stokes_drift.items()},
        stokes_v_by_cell={cell_id: values[1] for cell_id, values in diagnostics.stokes_drift.items()},
    )
    return BaselineArtifact(
        artifact_id=str(uuid4()),
        region_id=region_id,
        run_id=run_id,
        debris_class=debris_class,  # type: ignore[arg-type]
        generated_at=generated_at,
        forecast_valid_at=forecast_valid_at,
        horizon_hour=horizon_hour,
        grid_spec=grid_spec,
        forcing_refs=build_forcing_refs(region_name, source_mode_used=source_mode_used),
        baseline_engine=baseline_engine,
        source_mode_requested=source_mode_requested,  # type: ignore[arg-type]
        source_mode_used=source_mode_used,  # type: ignore[arg-type]
        is_fallback=is_fallback,
        source_notes=source_notes,
        density_uri=tensor_uris["density_uri"],
        current_u_uri=tensor_uris["current_u_uri"],
        current_v_uri=tensor_uris["current_v_uri"],
        wind_u_uri=tensor_uris["wind_u_uri"],
        wind_v_uri=tensor_uris["wind_v_uri"],
        ensemble_spread_uri=tensor_uris["ensemble_spread_uri"],
        beaching_fraction_uri=tensor_uris["beaching_fraction_uri"],
        stokes_u_uri=tensor_uris["stokes_u_uri"],
        stokes_v_uri=tensor_uris["stokes_v_uri"],
        stokes_magnitude_uri=tensor_uris["stokes_magnitude_uri"],
        manifest_uri="",
        parquet_index_uri="",
        metadata={
            "cell_map": {key: [value[0], value[1]] for key, value in cell_map.items()},
            "total_particles": diagnostics.total_particles,
            "beached_particles": diagnostics.beached_particles,
            "ensemble_members": diagnostics.ensemble_members,
        },
    )


def baseline_artifact_to_model(artifact: BaselineArtifact) -> BaselineArtifactModel:
    return BaselineArtifactModel(
        artifact_id=artifact.artifact_id,
        region_id=artifact.region_id,
        run_id=artifact.run_id,
        debris_class=artifact.debris_class,
        generated_at=artifact.generated_at,
        forecast_valid_at=artifact.forecast_valid_at,
        horizon_hour=artifact.horizon_hour,
        baseline_engine=artifact.baseline_engine,
        source_mode_requested=artifact.source_mode_requested,
        source_mode_used=artifact.source_mode_used,
        is_fallback=artifact.is_fallback,
        manifest_uri=artifact.manifest_uri,
        parquet_index_uri=artifact.parquet_index_uri,
        density_uri=artifact.density_uri,
        current_u_uri=artifact.current_u_uri,
        current_v_uri=artifact.current_v_uri,
        wind_u_uri=artifact.wind_u_uri,
        wind_v_uri=artifact.wind_v_uri,
        ensemble_spread_uri=artifact.ensemble_spread_uri,
        beaching_fraction_uri=artifact.beaching_fraction_uri,
        stokes_u_uri=artifact.stokes_u_uri,
        stokes_v_uri=artifact.stokes_v_uri,
        stokes_magnitude_uri=artifact.stokes_magnitude_uri,
        grid_spec_json=artifact.grid_spec.model_dump(mode="json"),
        forcing_refs_json=artifact.forcing_refs.model_dump(mode="json"),
        source_notes=list(artifact.source_notes),
        metadata_json=artifact.metadata,
    )


def persist_baseline_artifact(
    *,
    db: Session,
    region_id: str,
    region_name: str,
    run_id: str,
    debris_class: str,
    generated_at: datetime,
    forecast_valid_at: datetime,
    horizon_hour: int,
    source_mode_requested: str,
    source_mode_used: str,
    is_fallback: bool,
    source_notes: list[str],
    baseline_engine: BaselineEngine,
    current_u_by_cell: dict[str, float],
    current_v_by_cell: dict[str, float],
    wind_u_by_cell: dict[str, float],
    wind_v_by_cell: dict[str, float],
    diagnostics: DriftBaselineDiagnostics,
) -> BaselineArtifact:
    artifact = write_baseline_manifest(
        build_baseline_artifact(
            region_id=region_id,
            region_name=region_name,
            run_id=run_id,
            debris_class=debris_class,
            generated_at=generated_at,
            forecast_valid_at=forecast_valid_at,
            horizon_hour=horizon_hour,
            source_mode_requested=source_mode_requested,
            source_mode_used=source_mode_used,
            is_fallback=is_fallback,
            source_notes=source_notes,
            baseline_engine=baseline_engine,
            current_u_by_cell=current_u_by_cell,
            current_v_by_cell=current_v_by_cell,
            wind_u_by_cell=wind_u_by_cell,
            wind_v_by_cell=wind_v_by_cell,
            diagnostics=diagnostics,
        )
    )
    db.add(baseline_artifact_to_model(artifact))
    return artifact


def latest_baseline_artifact(
    db: Session,
    *,
    run_id: str,
    debris_class: str,
    horizon_hour: int,
) -> BaselineArtifact | None:
    row = db.execute(
        select(BaselineArtifactModel)
        .where(
            BaselineArtifactModel.run_id == run_id,
            BaselineArtifactModel.debris_class == debris_class,
            BaselineArtifactModel.horizon_hour == horizon_hour,
        )
        .order_by(BaselineArtifactModel.generated_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    return BaselineArtifact(
        artifact_id=row.artifact_id,
        region_id=row.region_id,
        run_id=row.run_id,
        debris_class=row.debris_class,  # type: ignore[arg-type]
        generated_at=row.generated_at,
        forecast_valid_at=row.forecast_valid_at,
        horizon_hour=row.horizon_hour,
        grid_spec=row.grid_spec_json,  # type: ignore[arg-type]
        forcing_refs=row.forcing_refs_json,  # type: ignore[arg-type]
        baseline_engine=row.baseline_engine,  # type: ignore[arg-type]
        source_mode_requested=row.source_mode_requested,  # type: ignore[arg-type]
        source_mode_used=row.source_mode_used,  # type: ignore[arg-type]
        is_fallback=row.is_fallback,
        source_notes=list(row.source_notes),
        density_uri=row.density_uri,
        current_u_uri=row.current_u_uri,
        current_v_uri=row.current_v_uri,
        wind_u_uri=row.wind_u_uri,
        wind_v_uri=row.wind_v_uri,
        ensemble_spread_uri=row.ensemble_spread_uri,
        beaching_fraction_uri=row.beaching_fraction_uri,
        stokes_u_uri=row.stokes_u_uri,
        stokes_v_uri=row.stokes_v_uri,
        stokes_magnitude_uri=row.stokes_magnitude_uri,
        manifest_uri=row.manifest_uri,
        parquet_index_uri=row.parquet_index_uri,
        metadata=dict(row.metadata_json),
    )

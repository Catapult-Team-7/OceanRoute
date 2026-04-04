from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import BaselineArtifactModel, DatasetArtifactModel, ForecastRunModel, MissionOutcomeModel, ObservationModel
from app.schemas import (
    DEBRIS_CLASS_METADATA,
    BaselineArtifact,
    DatasetArtifact,
    DatasetBuildRequest,
    DatasetExportRequest,
)
from app.services.data_lake_service import read_table_rows, read_tensor, write_dataset_export
from app.services.region_service import get_region_definition


INPUT_CHANNELS = [
    "current_u",
    "current_v",
    "wind_u",
    "wind_v",
    "baseline_density",
    "baseline_ensemble_spread",
    "baseline_beaching_fraction",
    "stokes_magnitude",
    "windage",
    "land_mask",
    "coastline_mask",
    "region_mask",
]

TARGET_CHANNELS = [
    "hotspot_probability",
    "expected_kg",
    "uncertainty",
]

SHARED_DATASET_REGION_ID = "multi_region"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_request(request: DatasetBuildRequest | DatasetExportRequest) -> DatasetExportRequest:
    if isinstance(request, DatasetExportRequest):
        region_ids = request.region_ids
        if region_ids is None and request.region_id is not None:
            region_ids = [request.region_id]
        return request.model_copy(update={"region_ids": region_ids})
    return DatasetExportRequest(
        region_id=request.region_id,
        region_ids=[request.region_id],
        max_forecast_runs=request.max_forecast_runs,
        lookback_hours=settings.training_lookback_hours,
        target_horizons=[24, 48, 72],
        label_strategy="observed_or_proxy",
    )


def _artifact_from_row(row: BaselineArtifactModel) -> BaselineArtifact:
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


def _group_artifacts(rows: list[BaselineArtifactModel]) -> dict[tuple[str, str, str], dict[int, BaselineArtifact]]:
    grouped: dict[tuple[str, str, str], dict[int, BaselineArtifact]] = {}
    for row in rows:
        artifact = _artifact_from_row(row)
        grouped.setdefault((artifact.region_id, artifact.run_id, artifact.debris_class), {})[artifact.horizon_hour] = artifact
    return grouped


def _load_mask(uri: str | None, shape: tuple[int, int]) -> np.ndarray:
    if uri is None:
        return np.zeros(shape, dtype=np.float32)
    return read_tensor(uri).astype(np.float32)


def _proxy_labels(target_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.any(target_density > 0):
        threshold = float(np.quantile(target_density[target_density > 0], 0.65))
    else:
        threshold = 0.0
    probability = (target_density >= threshold).astype(np.float32)
    kilograms = target_density.astype(np.float32)
    return probability, kilograms


def _git_ref() -> str:
    try:
        project_root = Path(__file__).resolve().parents[4]
        head_path = project_root / ".git" / "HEAD"
        if not head_path.exists():
            return "unknown"
        head_contents = head_path.read_text(encoding="utf-8").strip()
        if head_contents.startswith("ref:"):
            return head_contents.split("/")[-1]
        return head_contents[:12]
    except Exception:
        return "unknown"


def _feature_stats(x_tensor: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for channel_index, name in enumerate(INPUT_CHANNELS):
        values = x_tensor[:, :, channel_index].astype(np.float64)
        stats[name] = {
            "mean": round(float(values.mean()), 6),
            "std": round(float(max(values.std(), 1e-6)), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
        }
    return stats


def _region_ids_from_request(request: DatasetExportRequest) -> list[str]:
    if request.region_ids:
        return request.region_ids
    if request.region_id:
        return [request.region_id]
    return [settings.pilot_region]


def _observation_overrides(
    *,
    region_id: str,
    cell_map: dict[str, list[int]] | dict[str, tuple[int, int]],
    target_valid_at: datetime,
    observations: list[ObservationModel],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, bool]:
    region = get_region_definition(region_id)
    probability = np.full(shape, -1.0, dtype=np.float32)
    kilograms = np.full(shape, -1.0, dtype=np.float32)
    used_observations = False
    for observation in observations:
        if abs((observation.observed_at - target_valid_at).total_seconds()) > 86400:
            continue
        nearest_cell = min(
            region.cells,
            key=lambda cell: ((float(cell["lat"]) - observation.lat) ** 2) + ((float(cell["lon"]) - observation.lon) ** 2),
        )
        cell_id = str(nearest_cell["cell_id"])
        if cell_id not in cell_map:
            continue
        row, col = cell_map[cell_id]
        probability[row, col] = 1.0 if observation.found_status == "found" else 0.0
        kilograms[row, col] = max(float(observation.estimated_kg), 0.0)
        used_observations = True
    return probability, kilograms, used_observations


def _build_time_window_features(
    *,
    artifacts: list[BaselineArtifact],
    debris_class: str,
    shoreline_mask: np.ndarray,
    restricted_mask: np.ndarray,
    bathymetry_mask: np.ndarray,
) -> np.ndarray:
    coastline_mask = shoreline_mask.astype(np.float32)
    land_mask = (coastline_mask >= 0.7).astype(np.float32)
    region_mask = np.ones_like(coastline_mask, dtype=np.float32)
    windage = np.full_like(coastline_mask, float(DEBRIS_CLASS_METADATA[debris_class]["windage_factor"]), dtype=np.float32)
    frames: list[np.ndarray] = []
    for artifact in artifacts:
        frames.append(
            np.stack(
                [
                    read_tensor(artifact.current_u_uri) if artifact.current_u_uri else np.zeros_like(coastline_mask),
                    read_tensor(artifact.current_v_uri) if artifact.current_v_uri else np.zeros_like(coastline_mask),
                    read_tensor(artifact.wind_u_uri) if artifact.wind_u_uri else np.zeros_like(coastline_mask),
                    read_tensor(artifact.wind_v_uri) if artifact.wind_v_uri else np.zeros_like(coastline_mask),
                    read_tensor(artifact.density_uri),
                    read_tensor(artifact.ensemble_spread_uri),
                    read_tensor(artifact.beaching_fraction_uri),
                    read_tensor(artifact.stokes_magnitude_uri),
                    windage,
                    land_mask,
                    coastline_mask,
                    region_mask,
                ],
                axis=0,
            ).astype(np.float32)
        )
    return np.stack(frames, axis=0).astype(np.float32)


def build_dataset(request: DatasetBuildRequest | DatasetExportRequest, db: Session) -> DatasetArtifact:
    normalized = _normalize_request(request)
    region_ids = _region_ids_from_request(normalized)
    target_horizons = sorted(dict.fromkeys(normalized.target_horizons))

    run_rows: list[ForecastRunModel] = []
    for region_id in region_ids:
        run_rows.extend(
            db.execute(
                select(ForecastRunModel)
                .where(ForecastRunModel.pilot_region == region_id)
                .order_by(ForecastRunModel.generated_at.asc())
                .limit(normalized.max_forecast_runs)
            ).scalars().all()
        )
    if not run_rows:
        raise LookupError(f"No forecast runs were available for regions: {', '.join(region_ids)}.")

    run_ids = [run.id for run in run_rows]
    artifact_rows = db.execute(
        select(BaselineArtifactModel)
        .where(
            BaselineArtifactModel.region_id.in_(region_ids),
            BaselineArtifactModel.run_id.in_(run_ids),
        )
        .order_by(BaselineArtifactModel.generated_at.asc(), BaselineArtifactModel.horizon_hour.asc())
    ).scalars().all()
    if not artifact_rows:
        raise LookupError(f"No baseline artifacts were available for regions: {', '.join(region_ids)}.")

    grouped = _group_artifacts(artifact_rows)
    observations = db.execute(select(ObservationModel).order_by(ObservationModel.observed_at.asc())).scalars().all()
    mission_outcomes = db.execute(select(MissionOutcomeModel).order_by(MissionOutcomeModel.completed_at.asc())).scalars().all()
    mission_kg_proxy = max(1.0, sum(item.collected_kg for item in mission_outcomes) / max(len(mission_outcomes), 1))
    lookback_required = list(range(1, normalized.lookback_hours + 1))
    eligible_groups = [artifacts for artifacts in grouped.values() if all(hour in artifacts for hour in lookback_required)]
    effective_target_horizons = [hour for hour in target_horizons if eligible_groups and all(hour in artifacts for artifacts in eligible_groups)]
    if not effective_target_horizons:
        available_horizons = sorted(
            {
                hour
                for artifacts in eligible_groups
                for hour in target_horizons
                if hour in artifacts
            }
        )
        effective_target_horizons = available_horizons

    x_samples: list[np.ndarray] = []
    y_probability_samples: list[np.ndarray] = []
    y_kg_samples: list[np.ndarray] = []
    y_uncertainty_samples: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    provenance_counts = {"proxy": 0, "observations": 0}

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[1][min(item[1])].generated_at,
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    )
    for (region_id, run_id, debris_class), artifacts_by_horizon in ordered_groups:
        if not all(hour in artifacts_by_horizon for hour in lookback_required):
            continue
        if not effective_target_horizons or not all(hour in artifacts_by_horizon for hour in effective_target_horizons):
            continue
        template = artifacts_by_horizon[lookback_required[0]]
        shape = (template.grid_spec.height, template.grid_spec.width)
        shoreline_mask = _load_mask(template.grid_spec.shoreline_mask_uri, shape)
        restricted_mask = _load_mask(template.grid_spec.restricted_mask_uri, shape)
        bathymetry_mask = _load_mask(template.grid_spec.bathymetry_mask_uri, shape)
        cell_map = {key: value for key, value in template.metadata.get("cell_map", {}).items()}
        lookback_artifacts = [artifacts_by_horizon[hour] for hour in lookback_required]
        x_window = _build_time_window_features(
            artifacts=lookback_artifacts,
            debris_class=debris_class,
            shoreline_mask=shoreline_mask,
            restricted_mask=restricted_mask,
            bathymetry_mask=bathymetry_mask,
        )

        probability_targets: list[np.ndarray] = []
        kg_targets: list[np.ndarray] = []
        uncertainty_targets: list[np.ndarray] = []
        any_observations = False
        valid_targets = []
        for target_horizon in effective_target_horizons:
            target_artifact = artifacts_by_horizon[target_horizon]
            valid_targets.append(target_artifact.forecast_valid_at.isoformat())
            target_density = read_tensor(target_artifact.density_uri).astype(np.float32)
            target_uncertainty = np.clip(read_tensor(target_artifact.ensemble_spread_uri).astype(np.float32), 0.0, 1.0)
            y_prob, y_kg = _proxy_labels(target_density)
            obs_prob, obs_kg, used_observations = _observation_overrides(
                region_id=region_id,
                cell_map=cell_map,
                target_valid_at=target_artifact.forecast_valid_at,
                observations=observations,
                shape=target_density.shape,
            )
            if used_observations:
                any_observations = True
                observed_mask = obs_prob >= 0
                y_prob = np.where(observed_mask, obs_prob, y_prob)
                y_kg = np.where(obs_kg >= 0, obs_kg, y_kg)
            if not np.any(y_kg > 0):
                y_kg = y_kg + (target_density * mission_kg_proxy * 0.1)
            probability_targets.append(y_prob[np.newaxis, ...])
            kg_targets.append(y_kg[np.newaxis, ...])
            uncertainty_targets.append(target_uncertainty[np.newaxis, ...])

        sample_index = len(x_samples)
        sample_id = str(uuid4())
        x_samples.append(x_window)
        y_probability_samples.append(np.stack(probability_targets, axis=0).astype(np.float32))
        y_kg_samples.append(np.stack(kg_targets, axis=0).astype(np.float32))
        y_uncertainty_samples.append(np.stack(uncertainty_targets, axis=0).astype(np.float32))
        label_source = "observations" if any_observations else "proxy"
        provenance_counts[label_source] += 1
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "sample_index": sample_index,
                "region_id": region_id,
                "run_id": run_id,
                "debris_class": debris_class,
                "target_horizons": effective_target_horizons,
                "forecast_valid_at": valid_targets,
                "label_provenance": label_source,
                "x_index": sample_index,
                "y_index": sample_index,
            }
        )

    if not x_samples:
        raise LookupError(f"No complete time-window samples were available for regions: {', '.join(region_ids)}.")

    total_samples = len(x_samples)
    train_cutoff = int(total_samples * 0.70)
    val_cutoff = int(total_samples * 0.85)
    split_counts = {"train": 0, "val": 0, "test": 0}
    for sample_row in manifest_rows:
        index = int(sample_row["sample_index"])
        if index < train_cutoff:
            split = "train"
        elif index < val_cutoff:
            split = "val"
        else:
            split = "test"
        split_counts[split] += 1
        split_rows.append(
            {
                "sample_id": sample_row["sample_id"],
                "sample_index": index,
                "region_id": sample_row["region_id"],
                "run_id": sample_row["run_id"],
                "split": split,
            }
        )

    x_tensor = np.stack(x_samples, axis=0).astype(np.float32)
    y_probability_tensor = np.stack(y_probability_samples, axis=0).astype(np.float32)
    y_kg_tensor = np.stack(y_kg_samples, axis=0).astype(np.float32)
    y_uncertainty_tensor = np.stack(y_uncertainty_samples, axis=0).astype(np.float32)
    feature_stats = _feature_stats(x_tensor)
    dataset_id = normalized.dataset_id or str(uuid4())
    created_at = _now()
    region_id = region_ids[0] if len(region_ids) == 1 else SHARED_DATASET_REGION_ID
    metadata = {
        "schema_version": settings.dataset_version,
        "dataset_id": dataset_id,
        "region_ids": region_ids,
        "horizons": effective_target_horizons,
        "input_channels": INPUT_CHANNELS,
        "target_channels": TARGET_CHANNELS,
        "tensor_shapes": {
            "X": list(x_tensor.shape),
            "Y_hotspot_probability": list(y_probability_tensor.shape),
            "Y_expected_kg": list(y_kg_tensor.shape),
            "Y_uncertainty": list(y_uncertainty_tensor.shape),
        },
        "source_provenance": {
            "forecast_run_ids": run_ids,
            "label_provenance_counts": provenance_counts,
        },
        "git_ref": _git_ref(),
    }
    export_paths = write_dataset_export(
        dataset_id=dataset_id,
        region_id=region_id,
        dataset_version=settings.dataset_version,
        created_at=created_at,
        x_tensor=x_tensor,
        y_probability=y_probability_tensor,
        y_kg=y_kg_tensor,
        y_uncertainty=y_uncertainty_tensor,
        manifest_rows=manifest_rows,
        split_rows=split_rows,
        feature_stats=feature_stats,
        metadata=metadata,
    )

    artifact = DatasetArtifact(
        dataset_id=dataset_id,
        region_id=region_id,
        created_at=created_at,
        dataset_version=settings.dataset_version,
        label_type="hotspot_presence",
        sample_count=total_samples,
        feature_count=len(INPUT_CHANNELS),
        feature_names=list(INPUT_CHANNELS),
        artifact_path=str(Path(export_paths["metadata_path"]).parent),
        zarr_uri=export_paths["zarr_uri"],
        parquet_index_uri=export_paths["manifest_path"],
        manifest_path=export_paths["manifest_path"],
        splits_path=export_paths["splits_path"],
        feature_stats_path=export_paths["feature_stats_path"],
        metadata_path=export_paths["metadata_path"],
        split_counts=split_counts,
        schema_versions={
            "baseline_schema_version": settings.baseline_schema_version,
            "feature_schema_version": settings.feature_schema_version,
            "label_schema_version": settings.label_schema_version,
            "dataset_version": settings.dataset_version,
        },
        region_ids=region_ids,
        horizons=effective_target_horizons,
        input_channels=list(INPUT_CHANNELS),
        target_channels=list(TARGET_CHANNELS),
        tensor_shapes=metadata["tensor_shapes"],
        metadata={
            "feature_stats": feature_stats,
            "source_provenance": metadata["source_provenance"],
            "git_ref": metadata["git_ref"],
        },
    )
    db.add(
        DatasetArtifactModel(
            dataset_id=artifact.dataset_id,
            region_id=artifact.region_id,
            created_at=artifact.created_at,
            dataset_version=artifact.dataset_version,
            label_type=artifact.label_type,
            sample_count=artifact.sample_count,
            feature_count=artifact.feature_count,
            feature_names=artifact.feature_names,
            artifact_path=artifact.artifact_path,
            zarr_uri=artifact.zarr_uri,
            parquet_index_uri=artifact.parquet_index_uri,
            manifest_path=artifact.manifest_path,
            splits_path=artifact.splits_path,
            feature_stats_path=artifact.feature_stats_path,
            metadata_path=artifact.metadata_path,
            split_counts=artifact.split_counts,
            schema_versions=artifact.schema_versions,
            region_ids=artifact.region_ids,
            horizons=artifact.horizons,
            input_channels=artifact.input_channels,
            target_channels=artifact.target_channels,
            tensor_shapes=artifact.tensor_shapes,
            metadata_json=artifact.metadata,
        )
    )
    db.commit()
    return artifact


def _to_dataset_artifact(row: DatasetArtifactModel) -> DatasetArtifact:
    return DatasetArtifact(
        dataset_id=row.dataset_id,
        region_id=row.region_id,
        created_at=row.created_at,
        dataset_version=row.dataset_version,
        label_type=row.label_type,  # type: ignore[arg-type]
        sample_count=row.sample_count,
        feature_count=row.feature_count,
        feature_names=list(row.feature_names),
        artifact_path=row.artifact_path,
        zarr_uri=row.zarr_uri,
        parquet_index_uri=row.parquet_index_uri,
        manifest_path=row.manifest_path,
        splits_path=row.splits_path,
        feature_stats_path=row.feature_stats_path,
        metadata_path=row.metadata_path,
        split_counts=dict(row.split_counts),
        schema_versions=dict(row.schema_versions),
        region_ids=list(row.region_ids),
        horizons=list(row.horizons),
        input_channels=list(row.input_channels),
        target_channels=list(row.target_channels),
        tensor_shapes=dict(row.tensor_shapes),
        metadata=dict(row.metadata_json),
    )


def list_datasets(db: Session, region_id: str | None = None) -> list[DatasetArtifact]:
    query = select(DatasetArtifactModel).order_by(DatasetArtifactModel.created_at.desc())
    rows = db.execute(query).scalars().all()
    artifacts = [_to_dataset_artifact(row) for row in rows]
    if region_id is None:
        return artifacts
    return [item for item in artifacts if item.region_id == region_id or region_id in item.region_ids]


def inspect_dataset(dataset_id: str, db: Session) -> dict[str, object]:
    row = db.get(DatasetArtifactModel, dataset_id)
    if row is None:
        raise LookupError(f"Dataset {dataset_id} was not found.")
    metadata_path = Path(row.metadata_path) if row.metadata_path else None
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path and metadata_path.exists() else {}
    manifest_preview = read_table_rows(row.manifest_path)[:5] if row.manifest_path else []
    split_preview = read_table_rows(row.splits_path)[:5] if row.splits_path else []
    return {
        "dataset": _to_dataset_artifact(row).model_dump(mode="json"),
        "metadata": metadata_payload,
        "sample_index_preview": manifest_preview,
        "split_preview": split_preview,
    }

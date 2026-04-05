from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.schemas import BaselineArtifact, DatasetArtifact, ForecastStep, GridSpec, PredictionArtifact
from app.services.artifact_service import ensure_data_directories
from app.services.tensor_grid_service import static_region_masks, tensorize_cell_values

try:  # pragma: no cover - optional dependency
    import zarr

    ZARR_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    zarr = None
    ZARR_AVAILABLE = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _month_bucket(timestamp: datetime) -> tuple[str, str]:
    return timestamp.strftime("%Y"), timestamp.strftime("%m")


def _safe_path(*parts: str) -> Path:
    ensure_data_directories()
    return settings.data_root.joinpath("data_lake", *parts)


def training_runs_root() -> Path:
    ensure_data_directories()
    path = settings.data_root / "training_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_registry_root() -> Path:
    ensure_data_directories()
    path = settings.data_root / "model_registry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _write_tensor(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ZARR_AVAILABLE:
        store_path = path.with_suffix(".zarr")
        if store_path.exists():
            if store_path.is_dir():
                shutil.rmtree(store_path)
            else:
                store_path.unlink()
        zarr.save(str(store_path), array)
        return str(store_path)
    np.save(path.with_suffix(".npy"), array)
    return str(path.with_suffix(".npy"))


def read_tensor(uri: str) -> np.ndarray:
    def _legacy_group_fallback(group_path: Path, array_name: str) -> np.ndarray | None:
        candidates = [
            group_path / f"{array_name}.npy",
            group_path.parent / f"{array_name}.npy",
        ]
        for candidate in candidates:
            if candidate.exists():
                return np.load(candidate)
        return None

    if "::" in uri:
        group_uri, array_name = uri.split("::", maxsplit=1)
        group_path = Path(group_uri)
        if ZARR_AVAILABLE and group_path.suffix == ".zarr":
            try:
                group = zarr.open_group(str(group_path), mode="r")
                return np.asarray(group[array_name], dtype=np.float32)
            except Exception:
                legacy = _legacy_group_fallback(group_path, array_name)
                if legacy is not None:
                    return legacy
        legacy = _legacy_group_fallback(group_path, array_name)
        if legacy is not None:
            return legacy
        raise ValueError(f"Unsupported tensor-group artifact URI: {uri}")
    path = Path(uri)
    if path.suffix == ".zarr" and ZARR_AVAILABLE:
        loaded = zarr.load(str(path))
        return np.asarray(loaded, dtype=np.float32)
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported tensor artifact URI: {uri}")


def _write_table(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    try:
        frame.to_parquet(path, index=False)
        return str(path)
    except Exception:
        fallback = path.with_suffix(".json")
        fallback.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
        return str(fallback)


def _write_tensor_group(path: Path, arrays: dict[str, np.ndarray]) -> tuple[str, dict[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ZARR_AVAILABLE:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        group = zarr.open_group(str(path), mode="w")
        for name, array in arrays.items():
            group.create_array(name, data=np.asarray(array, dtype=np.float32), overwrite=True)
        return str(path), {name: f"{path}::{name}" for name in arrays}

    path.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(path / f"{name}.npy", array)
    return str(path), {name: f"{path}::{name}" for name in arrays}


def append_monthly_index(index_family: str, timestamp: datetime, region_id: str, row: dict[str, Any]) -> str:
    year, month = _month_bucket(timestamp)
    index_path = _safe_path("indexes", index_family, region_id, year, f"{month}.parquet")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row]
    if index_path.exists():
        try:
            existing = pd.read_parquet(index_path)
            rows = existing.to_dict(orient="records") + rows
        except Exception:
            fallback_json = index_path.with_suffix(".json")
            if fallback_json.exists():
                existing_rows = json.loads(fallback_json.read_text(encoding="utf-8"))
                rows = existing_rows + rows
    return _write_table(index_path, rows)


def write_static_masks(region_id: str) -> dict[str, str]:
    grid_spec, shoreline, restricted, bathymetry = static_region_masks(region_id)
    year, month = _month_bucket(_now())
    base = _safe_path("features", "static", region_id, year, month)
    return {
        "shoreline_mask_uri": _write_tensor(base / "shoreline_mask", shoreline),
        "restricted_mask_uri": _write_tensor(base / "restricted_mask", restricted),
        "bathymetry_mask_uri": _write_tensor(base / "bathymetry_mask", bathymetry),
        "grid_spec": json.loads(grid_spec.model_dump_json()),
    }


def write_baseline_tensors(
    *,
    region_id: str,
    generated_at: datetime,
    run_id: str,
    debris_class: str,
    horizon_hour: int,
    density_by_cell: dict[str, float],
    current_u_by_cell: dict[str, float],
    current_v_by_cell: dict[str, float],
    wind_u_by_cell: dict[str, float],
    wind_v_by_cell: dict[str, float],
    ensemble_spread_by_cell: dict[str, float],
    beaching_fraction_by_cell: dict[str, float],
    stokes_u_by_cell: dict[str, float],
    stokes_v_by_cell: dict[str, float],
) -> tuple[GridSpec, dict[str, str], dict[str, tuple[int, int]]]:
    year, month = _month_bucket(generated_at)
    base = _safe_path("baselines", region_id, year, month, run_id, f"{debris_class}-h{horizon_hour}")
    grid_spec, density_tensor, cell_map = tensorize_cell_values(region_id, density_by_cell)
    _, current_u_tensor, _ = tensorize_cell_values(region_id, current_u_by_cell)
    _, current_v_tensor, _ = tensorize_cell_values(region_id, current_v_by_cell)
    _, wind_u_tensor, _ = tensorize_cell_values(region_id, wind_u_by_cell)
    _, wind_v_tensor, _ = tensorize_cell_values(region_id, wind_v_by_cell)
    _, spread_tensor, _ = tensorize_cell_values(region_id, ensemble_spread_by_cell)
    _, beaching_tensor, _ = tensorize_cell_values(region_id, beaching_fraction_by_cell)
    _, stokes_u_tensor, _ = tensorize_cell_values(region_id, stokes_u_by_cell)
    _, stokes_v_tensor, _ = tensorize_cell_values(region_id, stokes_v_by_cell)
    stokes_magnitude = np.sqrt((stokes_u_tensor**2) + (stokes_v_tensor**2)).astype(np.float32)
    masks = write_static_masks(region_id)
    grid_spec.shoreline_mask_uri = masks["shoreline_mask_uri"]
    grid_spec.restricted_mask_uri = masks["restricted_mask_uri"]
    grid_spec.bathymetry_mask_uri = masks["bathymetry_mask_uri"]
    return (
        grid_spec,
        {
            "density_uri": _write_tensor(base / "density", density_tensor),
            "current_u_uri": _write_tensor(base / "current_u", current_u_tensor),
            "current_v_uri": _write_tensor(base / "current_v", current_v_tensor),
            "wind_u_uri": _write_tensor(base / "wind_u", wind_u_tensor),
            "wind_v_uri": _write_tensor(base / "wind_v", wind_v_tensor),
            "ensemble_spread_uri": _write_tensor(base / "ensemble_spread", spread_tensor),
            "beaching_fraction_uri": _write_tensor(base / "beaching_fraction", beaching_tensor),
            "stokes_u_uri": _write_tensor(base / "stokes_u", stokes_u_tensor),
            "stokes_v_uri": _write_tensor(base / "stokes_v", stokes_v_tensor),
            "stokes_magnitude_uri": _write_tensor(base / "stokes_magnitude", stokes_magnitude),
        },
        cell_map,
    )


def write_baseline_manifest(artifact: BaselineArtifact) -> BaselineArtifact:
    year, month = _month_bucket(artifact.generated_at)
    manifest_path = _safe_path("manifests", "baseline", artifact.region_id, year, month, f"{artifact.artifact_id}.json")
    manifest_uri = _write_json(manifest_path, artifact.model_dump(mode="json"))
    parquet_row = {
        "artifact_id": artifact.artifact_id,
        "region_id": artifact.region_id,
        "run_id": artifact.run_id,
        "debris_class": artifact.debris_class,
        "generated_at": artifact.generated_at.isoformat(),
        "forecast_valid_at": artifact.forecast_valid_at.isoformat(),
        "horizon_hour": artifact.horizon_hour,
        "baseline_engine": artifact.baseline_engine,
        "source_mode_used": artifact.source_mode_used,
        "manifest_uri": manifest_uri,
        "density_uri": artifact.density_uri,
        "current_u_uri": artifact.current_u_uri,
        "current_v_uri": artifact.current_v_uri,
        "wind_u_uri": artifact.wind_u_uri,
        "wind_v_uri": artifact.wind_v_uri,
        "ensemble_spread_uri": artifact.ensemble_spread_uri,
        "beaching_fraction_uri": artifact.beaching_fraction_uri,
        "stokes_u_uri": artifact.stokes_u_uri,
        "stokes_v_uri": artifact.stokes_v_uri,
        "stokes_magnitude_uri": artifact.stokes_magnitude_uri,
    }
    parquet_index_uri = append_monthly_index("baseline", artifact.generated_at, artifact.region_id, parquet_row)
    return artifact.model_copy(update={"manifest_uri": manifest_uri, "parquet_index_uri": parquet_index_uri})


def write_feature_snapshot(
    *,
    region_id: str,
    forecast_run_id: str,
    debris_class: str,
    generated_at: datetime,
    tensor: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    year, month = _month_bucket(generated_at)
    base = _safe_path("features", "runtime", region_id, year, month, forecast_run_id, debris_class)
    tensor_uri = _write_tensor(base / "X", tensor)
    manifest_uri = _write_json(
        base / "manifest.json",
        {
            "feature_artifact_uri": tensor_uri,
            "forecast_run_id": forecast_run_id,
            "region_id": region_id,
            "debris_class": debris_class,
            "created_at": generated_at.isoformat(),
            "metadata": metadata,
        },
    )
    return tensor_uri, manifest_uri


def write_prediction_artifact(artifact: PredictionArtifact, hotspot_probability: np.ndarray, expected_kg: np.ndarray) -> PredictionArtifact:
    year, month = _month_bucket(artifact.created_at)
    base = _safe_path("predictions", artifact.region_id, year, month, artifact.forecast_run_id, artifact.debris_class)
    hotspot_uri = _write_tensor(base / "hotspot_probability", hotspot_probability.astype(np.float32))
    kg_uri = _write_tensor(base / "expected_kg", expected_kg.astype(np.float32))
    uncertainty_uri = _write_tensor(base / "uncertainty", artifact.metadata["uncertainty_array"].astype(np.float32)) if "uncertainty_array" in artifact.metadata else None
    metadata = dict(artifact.metadata)
    metadata.pop("uncertainty_array", None)
    updated = artifact.model_copy(
        update={
            "hotspot_probability_uri": hotspot_uri,
            "expected_kg_uri": kg_uri,
            "uncertainty_uri": uncertainty_uri,
            "metadata": metadata,
        }
    )
    manifest_uri = _write_json(base / f"{artifact.artifact_id}.json", updated.model_dump(mode="json"))
    parquet_index_uri = append_monthly_index(
        "prediction",
        artifact.created_at,
        artifact.region_id,
        {
            "artifact_id": artifact.artifact_id,
            "forecast_run_id": artifact.forecast_run_id,
            "region_id": artifact.region_id,
            "debris_class": artifact.debris_class,
            "created_at": artifact.created_at.isoformat(),
            "model_id": artifact.model_id,
            "model_architecture": artifact.model_architecture,
            "feature_artifact_uri": artifact.feature_artifact_uri,
            "manifest_uri": manifest_uri,
            "hotspot_probability_uri": hotspot_uri,
            "expected_kg_uri": kg_uri,
            "uncertainty_uri": uncertainty_uri,
        },
    )
    return updated.model_copy(update={"manifest_uri": manifest_uri, "parquet_index_uri": parquet_index_uri})


def write_dataset_export(
    *,
    dataset_id: str,
    region_id: str,
    dataset_version: str,
    created_at: datetime,
    x_tensor: np.ndarray,
    y_probability: np.ndarray,
    y_kg: np.ndarray,
    y_uncertainty: np.ndarray,
    manifest_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    feature_stats: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, str]:
    base = _safe_path("datasets", dataset_id)
    tensors_uri, tensor_uris = _write_tensor_group(
        base / "tensors.zarr",
        {
            "X": x_tensor,
            "Y_hotspot_probability": y_probability,
            "Y_expected_kg": y_kg,
            "Y_uncertainty": y_uncertainty,
        },
    )
    manifest_path = _write_table(base / "manifest.parquet", manifest_rows)
    splits_path = _write_table(base / "splits.parquet", split_rows)
    feature_stats_path = _write_json(base / "feature_stats.json", feature_stats)
    metadata_path = _write_json(
        base / "metadata.json",
        {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "region_id": region_id,
            "created_at": created_at.isoformat(),
            "zarr_uri": tensors_uri,
            "tensor_uris": tensor_uris,
            "manifest_path": manifest_path,
            "splits_path": splits_path,
            "feature_stats_path": feature_stats_path,
            **metadata,
        },
    )
    return {
        "zarr_uri": tensors_uri,
        "manifest_path": manifest_path,
        "splits_path": splits_path,
        "feature_stats_path": feature_stats_path,
        "metadata_path": metadata_path,
        "x_uri": tensor_uris["X"],
        "y_probability_uri": tensor_uris["Y_hotspot_probability"],
        "y_kg_uri": tensor_uris["Y_expected_kg"],
        "y_uncertainty_uri": tensor_uris["Y_uncertainty"],
    }


def read_table_rows(uri: str) -> list[dict[str, Any]]:
    path = Path(uri)
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    return json.loads(path.read_text(encoding="utf-8"))


def load_prediction_tensors(artifact: PredictionArtifact) -> tuple[np.ndarray, np.ndarray]:
    return read_tensor(artifact.hotspot_probability_uri), read_tensor(artifact.expected_kg_uri)

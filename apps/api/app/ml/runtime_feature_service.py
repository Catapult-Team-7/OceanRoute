from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.ml.dataset_service import INPUT_CHANNELS
from app.schemas import DEBRIS_CLASS_METADATA
from app.services.baseline_runtime_service import latest_baseline_artifact
from app.services.data_lake_service import read_tensor, write_feature_snapshot


DEFAULT_TENSOR_LAYOUT = {
    "feature_snapshot": "T,C,Y,X",
    "model_input": "B,T,C,Y,X",
    "target": "B,H,1,Y,X",
}

SUPPORTED_RUNTIME_CHANNELS = {
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
    "restricted_mask",
    "bathymetry_mask",
}


def _artifact_paths(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("artifact_paths")
    return raw if isinstance(raw, dict) else {}


def _load_feature_schema(payload: dict[str, Any]) -> dict[str, Any]:
    schema_path = _artifact_paths(payload).get("feature_schema") or payload.get("feature_schema_path")
    if not schema_path:
        return {}
    path = Path(str(schema_path))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_runtime_feature_contract(payload: dict[str, Any]) -> dict[str, Any]:
    feature_schema = _load_feature_schema(payload)
    input_channels = list(feature_schema.get("input_channels") or payload.get("input_channels") or INPUT_CHANNELS)
    unsupported = [channel for channel in input_channels if channel not in SUPPORTED_RUNTIME_CHANNELS]
    if unsupported:
        raise ValueError(
            f"Model {payload.get('model_id', '<unknown>')} requires unsupported runtime channels: {unsupported}."
        )
    trained_horizons = [
        int(item)
        for item in (
            feature_schema.get("trained_horizons")
            or feature_schema.get("horizons")
            or payload.get("trained_horizons")
            or payload.get("horizons")
            or [24, 48, 72]
        )
    ]
    lookback_hours = int(
        feature_schema.get("lookback_hours")
        or payload.get("lookback_hours")
        or settings.training_lookback_hours
    )
    tensor_layout = feature_schema.get("tensor_layout") or payload.get("tensor_layout") or DEFAULT_TENSOR_LAYOUT
    tensor_shapes = feature_schema.get("tensor_shapes") or {}
    return {
        "model_id": payload.get("model_id"),
        "input_channels": input_channels,
        "trained_horizons": trained_horizons,
        "lookback_hours": lookback_hours,
        "tensor_layout": tensor_layout,
        "compatible_regions": list(feature_schema.get("compatible_regions") or payload.get("compatible_regions") or []),
        "normalization_stats_path": _artifact_paths(payload).get("normalization_stats")
        or payload.get("normalization_stats_path"),
        "feature_schema_path": _artifact_paths(payload).get("feature_schema") or payload.get("feature_schema_path"),
        "tensor_shapes": tensor_shapes,
    }


def resolve_prediction_horizons(*, trained_horizons: list[int], available_horizons: list[int]) -> list[int]:
    available = set(int(item) for item in available_horizons)
    return [int(horizon) for horizon in trained_horizons if int(horizon) in available]


def validate_runtime_feature_tensor(x_tensor: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    tensor = np.asarray(x_tensor, dtype=np.float32)
    if tensor.ndim not in {4, 5}:
        raise ValueError(
            f"Runtime feature tensor for model {contract.get('model_id', '<unknown>')} must have 4 or 5 dims; got {tensor.shape}."
        )
    time_axis = 1 if tensor.ndim == 5 else 0
    channel_axis = 2 if tensor.ndim == 5 else 1
    expected_time = int(contract["lookback_hours"])
    expected_channels = len(contract["input_channels"])
    if tensor.shape[time_axis] != expected_time:
        raise ValueError(
            f"Runtime feature tensor lookback mismatch for model {contract.get('model_id', '<unknown>')}: "
            f"expected {expected_time}, got {tensor.shape[time_axis]}."
        )
    if tensor.shape[channel_axis] != expected_channels:
        raise ValueError(
            f"Runtime feature tensor channel mismatch for model {contract.get('model_id', '<unknown>')}: "
            f"expected {expected_channels}, got {tensor.shape[channel_axis]}."
        )
    expected_x_shape = contract.get("tensor_shapes", {}).get("X")
    if isinstance(expected_x_shape, list):
        normalized = list(tensor.shape[1:]) if tensor.ndim == 5 else list(tensor.shape)
        if len(expected_x_shape) == 5:
            expected_x_shape = expected_x_shape[1:]
        if len(expected_x_shape) == len(normalized) and [int(value) for value in expected_x_shape] != normalized:
            raise ValueError(
                f"Runtime feature tensor shape mismatch for model {contract.get('model_id', '<unknown>')}: "
                f"expected {expected_x_shape}, got {normalized}."
            )
    return tensor


def build_runtime_feature_snapshot(
    *,
    model_payload: dict[str, Any],
    region_id: str,
    forecast_run_id: str,
    debris_class: str,
    generated_at,
    db: Session,
) -> tuple[str, str, dict[str, Any]]:
    contract = resolve_runtime_feature_contract(model_payload)
    lookback_hours = int(contract["lookback_hours"])
    lookback_artifacts = []
    missing_hours: list[int] = []
    for hour in range(1, lookback_hours + 1):
        artifact = latest_baseline_artifact(db, run_id=forecast_run_id, debris_class=debris_class, horizon_hour=hour)
        if artifact is None:
            missing_hours.append(hour)
            continue
        lookback_artifacts.append(artifact)
    if missing_hours:
        raise ValueError(
            f"Runtime features were unavailable because baseline artifacts for lookback hours {missing_hours} are missing."
        )

    template = lookback_artifacts[0]
    shape = (template.grid_spec.height, template.grid_spec.width)
    shoreline_mask = (
        read_tensor(template.grid_spec.shoreline_mask_uri).astype(np.float32)
        if template.grid_spec.shoreline_mask_uri
        else np.zeros(shape, dtype=np.float32)
    )
    restricted_mask = (
        read_tensor(template.grid_spec.restricted_mask_uri).astype(np.float32)
        if template.grid_spec.restricted_mask_uri
        else np.zeros(shape, dtype=np.float32)
    )
    bathymetry_mask = (
        read_tensor(template.grid_spec.bathymetry_mask_uri).astype(np.float32)
        if template.grid_spec.bathymetry_mask_uri
        else np.zeros(shape, dtype=np.float32)
    )
    coastline_mask = shoreline_mask.astype(np.float32)
    land_mask = (coastline_mask >= 0.7).astype(np.float32)
    region_mask = np.ones_like(coastline_mask, dtype=np.float32)
    windage = np.full_like(
        coastline_mask,
        float(DEBRIS_CLASS_METADATA[debris_class]["windage_factor"]),
        dtype=np.float32,
    )

    frames: list[np.ndarray] = []
    for artifact in lookback_artifacts:
        dynamic_channels = {
            "current_u": read_tensor(artifact.current_u_uri) if artifact.current_u_uri else np.zeros(shape, dtype=np.float32),
            "current_v": read_tensor(artifact.current_v_uri) if artifact.current_v_uri else np.zeros(shape, dtype=np.float32),
            "wind_u": read_tensor(artifact.wind_u_uri) if artifact.wind_u_uri else np.zeros(shape, dtype=np.float32),
            "wind_v": read_tensor(artifact.wind_v_uri) if artifact.wind_v_uri else np.zeros(shape, dtype=np.float32),
            "baseline_density": read_tensor(artifact.density_uri),
            "baseline_ensemble_spread": read_tensor(artifact.ensemble_spread_uri),
            "baseline_beaching_fraction": read_tensor(artifact.beaching_fraction_uri),
            "stokes_magnitude": read_tensor(artifact.stokes_magnitude_uri),
            "windage": windage,
            "land_mask": land_mask,
            "coastline_mask": coastline_mask,
            "region_mask": region_mask,
            "restricted_mask": restricted_mask,
            "bathymetry_mask": bathymetry_mask,
        }
        frame = []
        for channel in contract["input_channels"]:
            if channel not in dynamic_channels:
                raise ValueError(f"Runtime feature builder does not support channel '{channel}'.")
            frame.append(np.asarray(dynamic_channels[channel], dtype=np.float32))
        frames.append(np.stack(frame, axis=0).astype(np.float32))

    x_tensor = np.stack(frames, axis=0).astype(np.float32)
    validate_runtime_feature_tensor(x_tensor, contract)
    feature_artifact_uri, manifest_uri = write_feature_snapshot(
        region_id=region_id,
        forecast_run_id=forecast_run_id,
        debris_class=debris_class,
        generated_at=generated_at,
        tensor=x_tensor,
        metadata={
            "input_channels": contract["input_channels"],
            "lookback_hours": contract["lookback_hours"],
            "trained_horizons": contract["trained_horizons"],
            "tensor_layout": contract["tensor_layout"],
            "compatible_regions": contract["compatible_regions"],
            "baseline_artifact_ids": [artifact.artifact_id for artifact in lookback_artifacts],
            "grid_spec": template.grid_spec.model_dump(mode="json"),
            "feature_schema_path": contract["feature_schema_path"],
            "normalization_stats_path": contract["normalization_stats_path"],
        },
    )
    return feature_artifact_uri, manifest_uri, contract

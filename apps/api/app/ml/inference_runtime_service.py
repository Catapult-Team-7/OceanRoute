from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from sqlalchemy.orm import Session

from app.models import ModelRegistryModel, PredictionArtifactModel
from app.schemas import InferencePredictRequest, InferencePredictResponse, PredictionArtifact
from app.ml.model_registry_service import get_active_model_entry, get_active_model_payload
from app.services.data_lake_service import read_tensor, write_prediction_artifact

try:  # pragma: no cover - optional dependency
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


INFERENCE_SERVICE_VERSION = "v2"
_MODEL_CACHE: dict[str, object] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _heuristic_prediction(x_tensor: np.ndarray, target_horizons: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_density = x_tensor[:, -1, 4:5]
    spread = x_tensor[:, -1, 5:6]
    beaching = x_tensor[:, -1, 6:7]
    probability = np.clip(
        baseline_density / np.maximum(np.max(baseline_density, axis=(2, 3), keepdims=True), 1e-6),
        0.0,
        1.0,
    )
    expected_kg = np.maximum(0.0, baseline_density * (1.08 - (spread * 0.12)))
    uncertainty = np.clip((spread * 0.95) + (beaching * 0.05), 0.0, 1.0)
    count = len(target_horizons)
    return (
        np.repeat(probability, count, axis=1).astype(np.float32),
        np.repeat(expected_kg, count, axis=1).astype(np.float32),
        np.repeat(uncertainty, count, axis=1).astype(np.float32),
    )


def _load_model(model_id: str, artifact_path: str):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available for deep inference.")
    model = torch.jit.load(artifact_path)
    model.eval()
    _MODEL_CACHE[model_id] = model
    return model


def _run_exported_model(export_artifact_path: str, model_id: str, x_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = _load_model(model_id, export_artifact_path)
    with torch.no_grad():
        tensor_x = torch.from_numpy(x_tensor.astype(np.float32))
        probability, kilograms, uncertainty = model(tensor_x)
        return (
            probability.detach().cpu().numpy(),
            kilograms.detach().cpu().numpy(),
            uncertainty.detach().cpu().numpy(),
        )


def warm_load_model(model_id: str, db: Session) -> dict[str, str]:
    row = db.get(ModelRegistryModel, model_id)
    if row is None:
        raise LookupError(f"Model {model_id} was not found.")
    region_hint = row.compatible_regions[0] if row.compatible_regions else row.region_id
    payload = get_active_model_payload(db, region_hint, model_id=model_id)
    if payload is None:
        raise LookupError(f"Model {model_id} did not have a readable artifact payload.")
    if not payload.get("artifact_paths", {}).get("model"):
        return {"status": "skipped", "model_id": model_id}
    _load_model(model_id, str(payload["artifact_paths"]["model"]))
    return {"status": "ready", "model_id": model_id}


def predict_from_feature_artifact(request: InferencePredictRequest, db: Session) -> InferencePredictResponse:
    active_model = get_active_model_entry(db, request.region_id, model_id=request.model_id)
    if active_model is None:
        raise LookupError(f"No compatible model was available for region {request.region_id}.")
    payload = get_active_model_payload(db, request.region_id, model_id=active_model.model_id)
    if payload is None:
        raise LookupError(f"No active model payload was available for region {request.region_id}.")

    x_tensor = read_tensor(request.feature_artifact_uri).astype(np.float32)
    if x_tensor.ndim == 4:
        x_tensor = x_tensor[np.newaxis, ...]

    used_fallback = False
    if payload.get("architecture") in {"temporal_unet", "convlstm"}:
        artifact_model_path = str(payload.get("artifact_paths", {}).get("model", ""))
        try:
            probability, expected_kg, uncertainty = _run_exported_model(artifact_model_path, str(payload["model_id"]), x_tensor)
            model_horizons = [int(item) for item in payload.get("horizons", request.target_horizons)]
            horizon_indexes = [model_horizons.index(horizon) for horizon in request.target_horizons if horizon in model_horizons]
            if len(horizon_indexes) != len(request.target_horizons):
                raise ValueError("Requested horizons were not available in the exported model artifact.")
            probability = probability[:, horizon_indexes]
            expected_kg = expected_kg[:, horizon_indexes]
            uncertainty = uncertainty[:, horizon_indexes]
        except Exception:
            probability, expected_kg, uncertainty = _heuristic_prediction(x_tensor, request.target_horizons)
            used_fallback = True
    else:
        probability, expected_kg, uncertainty = _heuristic_prediction(x_tensor, request.target_horizons)
        used_fallback = True

    probability_out = probability[0]
    expected_kg_out = expected_kg[0]
    uncertainty_out = uncertainty[0]
    artifact = write_prediction_artifact(
        PredictionArtifact(
            artifact_id=str(uuid4()),
            forecast_run_id=request.forecast_run_id,
            region_id=request.region_id,
            debris_class=request.debris_class,
            created_at=_now(),
            model_id=str(payload["model_id"]),
            model_architecture=payload["architecture"],  # type: ignore[arg-type]
            model_dataset_version=str(payload.get("dataset_version", "v2")),
            training_scope=payload.get("training_scope"),
            target_horizons=request.target_horizons,
            inference_service_version=INFERENCE_SERVICE_VERSION,
            feature_artifact_uri=request.feature_artifact_uri,
            hotspot_probability_uri="",
            expected_kg_uri="",
            uncertainty_uri=None,
            feature_schema_path=payload.get("artifact_paths", {}).get("feature_schema"),
            normalization_stats_path=payload.get("artifact_paths", {}).get("normalization_stats"),
            parquet_index_uri="",
            manifest_uri="",
            metadata={"used_fallback": used_fallback, "uncertainty_array": uncertainty_out},
        ),
        probability_out,
        expected_kg_out,
    )
    db.add(
        PredictionArtifactModel(
            artifact_id=artifact.artifact_id,
            forecast_run_id=artifact.forecast_run_id,
            region_id=artifact.region_id,
            debris_class=artifact.debris_class,
            created_at=artifact.created_at,
            model_id=artifact.model_id,
            model_architecture=artifact.model_architecture,
            model_dataset_version=artifact.model_dataset_version,
            training_scope=artifact.training_scope,
            inference_service_version=artifact.inference_service_version,
            feature_artifact_uri=artifact.feature_artifact_uri,
            hotspot_probability_uri=artifact.hotspot_probability_uri,
            expected_kg_uri=artifact.expected_kg_uri,
            uncertainty_uri=artifact.uncertainty_uri,
            feature_schema_path=artifact.feature_schema_path,
            normalization_stats_path=artifact.normalization_stats_path,
            parquet_index_uri=artifact.parquet_index_uri,
            manifest_uri=artifact.manifest_uri,
            metadata_json=artifact.metadata,
        )
    )
    db.commit()
    return InferencePredictResponse(
        artifact=artifact,
        loaded_model_id=artifact.model_id,
        used_fallback=used_fallback,
    )

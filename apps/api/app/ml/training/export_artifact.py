from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app.services.data_lake_service import model_registry_root

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


def _export_torch_model(model, model_path: Path, feature_schema: dict[str, Any]) -> str | None:
    if model is None or torch is None:
        return None
    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save(str(model_path))
        return "script"
    except Exception:
        tensor_shapes = feature_schema.get("tensor_shapes", {})
        x_shape = tensor_shapes.get("X")
        if not isinstance(x_shape, list) or len(x_shape) != 5:
            raise
        _, time_steps, channels, height, width = [int(value) for value in x_shape]
        example_input = torch.zeros((1, time_steps, channels, height, width), dtype=torch.float32)
        traced = torch.jit.trace(model, example_input)
        traced.save(str(model_path))
        return "trace"


def export_model_artifact(
    *,
    model,
    model_id: str,
    architecture: str,
    dataset_id: str,
    dataset_version: str,
    training_scope: str,
    trained_regions: list[str],
    compatible_regions: list[str],
    horizons: list[int],
    input_channels: list[str],
    output_heads: list[str],
    training_command: str,
    created_at: str,
    eval_metrics: dict[str, Any],
    normalization_stats_path: str,
    feature_schema: dict[str, Any],
    best_checkpoint_path: str | None,
    framework: str,
    lookback_hours: int,
    tensor_layout: dict[str, Any],
) -> dict[str, str]:
    artifact_dir = model_registry_root() / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    feature_schema_path = artifact_dir / "feature_schema.json"
    feature_schema_path.write_text(json.dumps(feature_schema, indent=2, default=str), encoding="utf-8")
    normalization_path = artifact_dir / "normalization_stats.json"
    normalization_path.write_text(Path(normalization_stats_path).read_text(encoding="utf-8"), encoding="utf-8")
    eval_path = artifact_dir / "eval_metrics.json"
    eval_path.write_text(json.dumps(eval_metrics, indent=2, default=str), encoding="utf-8")

    checkpoint_copy = artifact_dir / "checkpoint.ckpt"
    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        shutil.copyfile(best_checkpoint_path, checkpoint_copy)

    model_path = artifact_dir / "model.ts"
    export_format = _export_torch_model(model, model_path, feature_schema)

    metadata = {
        "model_id": model_id,
        "architecture": architecture,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "training_scope": training_scope,
        "trained_regions": trained_regions,
        "compatible_regions": compatible_regions,
        "horizons": horizons,
        "trained_horizons": horizons,
        "lookback_hours": lookback_hours,
        "tensor_layout": tensor_layout,
        "input_channels": input_channels,
        "output_heads": output_heads,
        "training_command": training_command,
        "created_at": created_at,
        "eval_metrics": eval_metrics,
        "artifact_paths": {
            "model": str(model_path),
            "feature_schema": str(feature_schema_path),
            "normalization_stats": str(normalization_path),
            "eval_metrics": str(eval_path),
            "checkpoint": str(checkpoint_copy),
        },
        "promotion_status": "candidate",
        "framework": framework,
        "export_format": export_format,
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {
        "artifact_dir": str(artifact_dir),
        "metadata_path": str(metadata_path),
        "model_path": str(model_path),
        "feature_schema_path": str(feature_schema_path),
        "normalization_stats_path": str(normalization_path),
        "eval_metrics_path": str(eval_path),
        "checkpoint_path": str(checkpoint_copy),
    }

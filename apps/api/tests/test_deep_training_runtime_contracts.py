from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from app.models import ModelRegistryModel
from app.schemas import InferencePredictResponse, PredictionArtifact
from app.services.data_lake_service import read_tensor


pytestmark = [pytest.mark.integration, pytest.mark.ml, pytest.mark.slow]


def test_dataset_export_writes_canonical_contract(client) -> None:
    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 72,
            "debris_classes": ["low", "high"],
            "seed": 12,
            "source_mode": "sample",
        },
    )
    assert forecast_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/export",
        json={
            "region_id": "sf_bay_estuary",
            "max_forecast_runs": 4,
            "lookback_hours": 12,
            "target_horizons": [24, 48, 72],
        },
    )
    assert dataset_response.status_code == 200
    dataset_payload = dataset_response.json()
    assert dataset_payload["input_channels"] == [
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
    assert dataset_payload["target_channels"] == ["hotspot_probability", "expected_kg", "uncertainty"]
    assert dataset_payload["region_ids"] == ["sf_bay_estuary"]
    assert dataset_payload["horizons"] == [24, 48, 72]
    assert Path(dataset_payload["manifest_path"]).exists()
    assert Path(dataset_payload["splits_path"]).exists()
    assert Path(dataset_payload["feature_stats_path"]).exists()
    assert Path(dataset_payload["metadata_path"]).exists()

    inspect_response = client.get(f"/api/ml/datasets/{dataset_payload['dataset_id']}")
    assert inspect_response.status_code == 200
    inspect_payload = inspect_response.json()
    assert inspect_payload["metadata"]["tensor_shapes"]["X"][2] == 12
    assert inspect_payload["metadata"]["horizons"] == [24, 48, 72]
    assert inspect_payload["sample_index_preview"]
    assert inspect_payload["split_preview"]


def test_dataset_export_fails_strictly_when_requested_horizons_are_missing(client) -> None:
    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 48,
            "debris_classes": ["low", "high"],
            "seed": 13,
            "source_mode": "sample",
        },
    )
    assert forecast_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/export",
        json={
            "region_id": "sf_bay_estuary",
            "max_forecast_runs": 4,
            "lookback_hours": 12,
            "target_horizons": [24, 48, 72],
        },
    )
    assert dataset_response.status_code == 400
    assert "allow_partial_horizons=true" in dataset_response.json()["detail"]


def test_dataset_export_can_opt_into_partial_horizon_resolution(client) -> None:
    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 48,
            "debris_classes": ["low", "high"],
            "seed": 14,
            "source_mode": "sample",
        },
    )
    assert forecast_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/export",
        json={
            "region_id": "sf_bay_estuary",
            "max_forecast_runs": 4,
            "lookback_hours": 12,
            "target_horizons": [24, 48, 72],
            "allow_partial_horizons": True,
        },
    )
    assert dataset_response.status_code == 200
    dataset_payload = dataset_response.json()
    assert dataset_payload["horizons"] == [24, 48]
    assert dataset_payload["metadata"]["requested_horizons"] == [24, 48, 72]
    assert dataset_payload["metadata"]["allow_partial_horizons"] is True


def test_runtime_prefers_per_region_champion_over_shared_champion(client) -> None:
    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 48,
            "debris_classes": ["low", "high"],
            "seed": 21,
            "source_mode": "sample",
        },
    )
    assert forecast_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/build",
        json={"region_id": "sf_bay_estuary", "label_type": "hotspot_presence", "max_forecast_runs": 4},
    )
    dataset_id = dataset_response.json()["dataset_id"]

    shared_train = client.post(
        "/api/ml/train",
        json={
            "region_ids": ["sf_bay_estuary"],
            "dataset_id": dataset_id,
            "architecture": "linear_residual",
            "training_scope": "shared",
            "activate": True,
            "promote_policy": "always_activate",
        },
    )
    assert shared_train.status_code == 200
    shared_model_id = shared_train.json()["model"]["model_id"]

    per_region_train = client.post(
        "/api/ml/train",
        json={
            "region_id": "sf_bay_estuary",
            "dataset_id": dataset_id,
            "architecture": "linear_residual",
            "training_scope": "per_region",
            "activate": True,
            "promote_policy": "always_activate",
        },
    )
    assert per_region_train.status_code == 200
    per_region_model_id = per_region_train.json()["model"]["model_id"]

    rerun_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 24,
            "debris_classes": ["low"],
            "seed": 22,
            "source_mode": "sample",
        },
    )
    assert rerun_response.status_code == 200
    rerun_payload = rerun_response.json()
    assert rerun_payload["summary"]["active_model_id"] == per_region_model_id
    assert rerun_payload["summary"]["training_scope"] == "per_region"
    assert rerun_payload["summary"]["active_model_id"] != shared_model_id


def test_auto_promotion_is_blocked_and_evaluate_reports_trust_context(client) -> None:
    backfill_response = client.post(
        "/api/forecast/backfill",
        json={
            "region_id": "sf_bay_estuary",
            "source_mode": "sample",
            "days": 10,
            "debris_classes": ["low", "high"],
        },
    )
    assert backfill_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/export",
        json={
            "region_id": "sf_bay_estuary",
            "max_forecast_runs": 10,
            "lookback_hours": 12,
            "target_horizons": [24, 48, 72],
        },
    )
    assert dataset_response.status_code == 200
    dataset_id = dataset_response.json()["dataset_id"]

    train_response = client.post(
        "/api/ml/train",
        json={
            "region_id": "sf_bay_estuary",
            "dataset_id": dataset_id,
            "architecture": "linear_residual",
            "activate": True,
        },
    )
    assert train_response.status_code == 200
    train_payload = train_response.json()
    model_id = train_payload["model"]["model_id"]
    assert train_payload["model"]["stage"] == "candidate"
    assert train_payload["model"]["is_active"] is False
    assert train_payload["metrics"]["promotion_eligible"] is False
    assert "sample_count<50" in train_payload["metrics"]["promotion_blockers"]
    assert "precision_at_10_defined" in train_payload["metrics"]

    evaluate_response = client.get(f"/api/ml/models/{model_id}/evaluate")
    assert evaluate_response.status_code == 200
    evaluate_payload = evaluate_response.json()
    assert evaluate_payload["stage"] == "candidate"
    assert evaluate_payload["artifact_path"]
    assert evaluate_payload["export_format"] == "json"
    assert evaluate_payload["metrics"]["sample_count"] < 50
    assert evaluate_payload["metrics"]["test_split_count"] < 10
    assert evaluate_payload["metrics"]["promotion_eligible"] is False
    assert "split_counts" in evaluate_payload["metrics"]


def test_explicit_candidate_model_id_runs_forecast_without_promotion(client) -> None:
    backfill_response = client.post(
        "/api/forecast/backfill",
        json={
            "region_id": "sf_bay_estuary",
            "source_mode": "sample",
            "days": 10,
            "debris_classes": ["low", "high"],
        },
    )
    assert backfill_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/build",
        json={"region_id": "sf_bay_estuary", "label_type": "hotspot_presence", "max_forecast_runs": 10},
    )
    dataset_id = dataset_response.json()["dataset_id"]
    train_response = client.post(
        "/api/ml/train",
        json={
            "region_id": "sf_bay_estuary",
            "dataset_id": dataset_id,
            "architecture": "linear_residual",
            "activate": False,
            "promote_policy": "candidate_only",
        },
    )
    assert train_response.status_code == 200
    model_id = train_response.json()["model"]["model_id"]

    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 24,
            "debris_classes": ["low"],
            "seed": 31,
            "source_mode": "sample",
            "model_id": model_id,
        },
    )
    assert forecast_response.status_code == 200
    forecast_payload = forecast_response.json()
    assert forecast_payload["summary"]["requested_model_id"] == model_id
    assert forecast_payload["summary"]["resolved_model_id"] == model_id
    assert forecast_payload["summary"]["model_stage"] == "candidate"
    assert forecast_payload["summary"]["used_candidate_override"] is True
    assert forecast_payload["provenance"]["requested_model_id"] == model_id
    assert forecast_payload["provenance"]["resolved_model_id"] == model_id
    assert forecast_payload["provenance"]["model_stage"] == "candidate"
    assert forecast_payload["provenance"]["used_candidate_override"] is True
    assert forecast_payload["provenance"]["used_inference_fallback"] is False


def test_explicit_deep_candidate_model_resolves_runtime_horizons_against_available_forecast_horizons(
    client,
    db_session,
    monkeypatch,
    tmp_path,
) -> None:
    model_id = "candidate-deep-resolution"
    db_session.add(
        ModelRegistryModel(
            model_id=model_id,
            region_id="sf_bay_estuary",
            created_at=datetime.now(timezone.utc),
            architecture="convlstm",
            status="completed",
            stage="candidate",
            is_active=False,
            training_scope="per_region",
            artifact_path="C:\\candidate\\metadata.json",
            dataset_id="fixture-dataset",
            dataset_version="v2",
            trained_regions=["sf_bay_estuary"],
            compatible_regions=["sf_bay_estuary"],
            horizons=[24, 48, 72],
            input_channels=[
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
            ],
            output_heads=["hotspot_probability", "expected_kg", "uncertainty"],
            normalization_stats_path=None,
            feature_schema_path=None,
            best_checkpoint_path=None,
            checkpoint_path=None,
            export_artifact_path="C:\\candidate\\model.ts",
            evaluation_path=None,
            framework="pytorch_lightning",
            metrics_json={},
        )
    )
    db_session.commit()

    captured: dict[str, object] = {}

    def _fake_predict(request, db):
        captured["target_horizons"] = list(request.target_horizons)
        x_tensor = read_tensor(request.feature_artifact_uri)
        height, width = x_tensor.shape[-2:]
        horizon_count = len(request.target_horizons)
        probability_path = tmp_path / "probability.npy"
        kilograms_path = tmp_path / "kilograms.npy"
        uncertainty_path = tmp_path / "uncertainty.npy"
        np.save(probability_path, np.full((horizon_count, height, width), 0.6, dtype=np.float32))
        np.save(kilograms_path, np.full((horizon_count, height, width), 2.5, dtype=np.float32))
        np.save(uncertainty_path, np.full((horizon_count, height, width), 0.2, dtype=np.float32))
        artifact = PredictionArtifact(
            artifact_id="prediction-artifact",
            forecast_run_id=request.forecast_run_id,
            region_id=request.region_id,
            debris_class=request.debris_class,
            created_at=datetime.now(timezone.utc),
            model_id=model_id,
            model_architecture="convlstm",
            model_dataset_version="v2",
            training_scope="per_region",
            target_horizons=list(request.target_horizons),
            inference_service_version="test",
            feature_artifact_uri=request.feature_artifact_uri,
            hotspot_probability_uri=str(probability_path),
            expected_kg_uri=str(kilograms_path),
            uncertainty_uri=str(uncertainty_path),
            feature_schema_path=None,
            normalization_stats_path=None,
            parquet_index_uri="",
            manifest_uri=str(tmp_path / "prediction.json"),
            metadata={},
        )
        return InferencePredictResponse(
            artifact=artifact,
            loaded_model_id=model_id,
            used_fallback=False,
        )

    monkeypatch.setattr("app.services.forecast_service.predict_with_inference_service", _fake_predict)

    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 24,
            "debris_classes": ["low"],
            "seed": 36,
            "source_mode": "sample",
            "model_id": model_id,
        },
    )
    assert forecast_response.status_code == 200
    assert captured["target_horizons"] == [24]
    forecast_payload = forecast_response.json()
    assert forecast_payload["summary"]["requested_model_id"] == model_id
    assert forecast_payload["summary"]["resolved_model_id"] == model_id
    assert forecast_payload["summary"]["used_candidate_override"] is True


def test_explicit_incompatible_model_id_fails_cleanly(client) -> None:
    backfill_response = client.post(
        "/api/forecast/backfill",
        json={
            "region_id": "sf_bay_estuary",
            "source_mode": "sample",
            "days": 10,
            "debris_classes": ["low", "high"],
        },
    )
    assert backfill_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/build",
        json={"region_id": "sf_bay_estuary", "label_type": "hotspot_presence", "max_forecast_runs": 10},
    )
    dataset_id = dataset_response.json()["dataset_id"]
    train_response = client.post(
        "/api/ml/train",
        json={
            "region_id": "sf_bay_estuary",
            "dataset_id": dataset_id,
            "architecture": "linear_residual",
            "activate": False,
            "promote_policy": "candidate_only",
        },
    )
    assert train_response.status_code == 200
    model_id = train_response.json()["model"]["model_id"]

    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "puget_sound",
            "horizon_hours": 24,
            "debris_classes": ["low"],
            "seed": 32,
            "source_mode": "sample",
            "model_id": model_id,
        },
    )
    assert forecast_response.status_code == 404


def test_explicit_deep_candidate_override_fails_cleanly_when_inference_fails(client, db_session, monkeypatch) -> None:
    model_id = "candidate-deep-model"
    db_session.add(
        ModelRegistryModel(
            model_id=model_id,
            region_id="sf_bay_estuary",
            created_at=datetime.now(timezone.utc),
            architecture="convlstm",
            status="completed",
            stage="candidate",
            is_active=False,
            training_scope="per_region",
            artifact_path="C:\\candidate\\metadata.json",
            dataset_id="fixture-dataset",
            dataset_version="v2",
            trained_regions=["sf_bay_estuary"],
            compatible_regions=["sf_bay_estuary"],
            horizons=[24, 48, 72],
            input_channels=[],
            output_heads=["hotspot_probability", "expected_kg", "uncertainty"],
            normalization_stats_path=None,
            feature_schema_path=None,
            best_checkpoint_path=None,
            checkpoint_path=None,
            export_artifact_path="C:\\candidate\\model.ts",
            evaluation_path=None,
            framework="pytorch_lightning",
            metrics_json={},
        )
    )
    db_session.commit()

    def _boom(*args, **kwargs):
        raise RuntimeError("forced inference failure")

    monkeypatch.setattr("app.services.forecast_service.predict_with_inference_service", _boom)

    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 24,
            "debris_classes": ["low"],
            "seed": 33,
            "source_mode": "sample",
            "model_id": model_id,
        },
    )
    assert forecast_response.status_code == 400


def test_default_champion_path_keeps_forecast_running_when_deep_inference_fails(client, db_session, monkeypatch) -> None:
    model_id = "champion-deep-model"
    db_session.add(
        ModelRegistryModel(
            model_id=model_id,
            region_id="sf_bay_estuary",
            created_at=datetime.now(timezone.utc),
            architecture="convlstm",
            status="completed",
            stage="champion",
            is_active=True,
            training_scope="per_region",
            artifact_path="C:\\champion\\metadata.json",
            dataset_id="fixture-dataset",
            dataset_version="v2",
            trained_regions=["sf_bay_estuary"],
            compatible_regions=["sf_bay_estuary"],
            horizons=[24, 48, 72],
            input_channels=[],
            output_heads=["hotspot_probability", "expected_kg", "uncertainty"],
            normalization_stats_path=None,
            feature_schema_path=None,
            best_checkpoint_path=None,
            checkpoint_path=None,
            export_artifact_path="C:\\champion\\model.ts",
            evaluation_path=None,
            framework="pytorch_lightning",
            metrics_json={},
        )
    )
    db_session.commit()

    def _boom(*args, **kwargs):
        raise RuntimeError("forced inference failure")

    monkeypatch.setattr("app.services.forecast_service.predict_with_inference_service", _boom)

    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 24,
            "debris_classes": ["low"],
            "seed": 34,
            "source_mode": "sample",
        },
    )
    assert forecast_response.status_code == 200
    forecast_payload = forecast_response.json()
    assert forecast_payload["summary"].get("used_candidate_override") is False
    assert forecast_payload["summary"].get("used_inference_fallback") is True
    assert forecast_payload["summary"].get("resolved_model_id") is None

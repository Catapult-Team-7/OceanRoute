from __future__ import annotations

from pathlib import Path


def test_dataset_export_writes_canonical_contract(client) -> None:
    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "sf_bay_estuary",
            "horizon_hours": 48,
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
    assert dataset_payload["horizons"] == [24, 48]
    assert Path(dataset_payload["manifest_path"]).exists()
    assert Path(dataset_payload["splits_path"]).exists()
    assert Path(dataset_payload["feature_stats_path"]).exists()
    assert Path(dataset_payload["metadata_path"]).exists()

    inspect_response = client.get(f"/api/ml/datasets/{dataset_payload['dataset_id']}")
    assert inspect_response.status_code == 200
    inspect_payload = inspect_response.json()
    assert inspect_payload["metadata"]["tensor_shapes"]["X"][2] == 12
    assert inspect_payload["metadata"]["horizons"] == [24, 48]
    assert inspect_payload["sample_index_preview"]
    assert inspect_payload["split_preview"]


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

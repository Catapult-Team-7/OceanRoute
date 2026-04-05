from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_regions_endpoint_and_ml_training_pipeline(client) -> None:
    regions_response = client.get("/api/regions")
    assert regions_response.status_code == 200
    regions_payload = regions_response.json()
    region_ids = {item["id"] for item in regions_payload}
    assert {"sf_bay_estuary", "puget_sound", "long_island_sound"} <= region_ids

    forecast_response = client.post(
        "/api/forecast/run",
        json={
            "region_id": "puget_sound",
            "horizon_hours": 48,
            "debris_classes": ["low", "high"],
            "source_strength": 1.05,
            "seed": 9,
            "source_mode": "sample",
        },
    )
    assert forecast_response.status_code == 200
    forecast_payload = forecast_response.json()
    assert forecast_payload["region"]["id"] == "puget_sound"

    latest_response = client.get("/api/forecast/latest", params={"region_id": "puget_sound", "horizon_hour": 24})
    assert latest_response.status_code == 200
    latest_payload = latest_response.json()
    assert latest_payload["region"]["id"] == "puget_sound"
    assert latest_payload["top_hotspots"]
    assert latest_payload["steps"][0]["baseline_density"] >= 0
    assert latest_payload["steps"][0]["ensemble_spread"] >= 0
    assert "mean_beaching_fraction" in latest_payload["summary"]

    top_hotspot = latest_payload["top_hotspots"][0]
    observation_response = client.post(
        "/api/observations/upload",
        json={
            "observed_at": latest_payload["generated_at"],
            "lat": top_hotspot["lat"],
            "lon": top_hotspot["lon"],
            "found_status": "found",
            "debris_class": top_hotspot["debris_class"],
            "estimated_kg": 4.2,
            "confidence": 0.84,
        },
    )
    assert observation_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/build",
        json={"region_id": "puget_sound", "label_type": "hotspot_presence", "max_forecast_runs": 5},
    )
    assert dataset_response.status_code == 200
    dataset_payload = dataset_response.json()
    assert dataset_payload["region_id"] == "puget_sound"
    assert dataset_payload["sample_count"] > 0
    assert "feature_stats" in dataset_payload["metadata"]

    train_response = client.post(
        "/api/ml/train",
        json={
            "region_id": "puget_sound",
            "dataset_id": dataset_payload["dataset_id"],
            "architecture": "linear_residual",
            "activate": True,
            "promote_policy": "always_activate",
        },
    )
    assert train_response.status_code == 200
    train_payload = train_response.json()
    assert train_payload["model"]["region_id"] == "puget_sound"
    assert train_payload["model"]["is_active"] is True

    models_response = client.get("/api/ml/models", params={"region_id": "puget_sound"})
    assert models_response.status_code == 200
    models_payload = models_response.json()
    assert models_payload
    assert models_payload[0]["model_id"] == train_payload["model"]["model_id"]

    forecast_with_model = client.post(
        "/api/forecast/run",
        json={
            "region_id": "puget_sound",
            "horizon_hours": 24,
            "debris_classes": ["low"],
            "seed": 10,
            "source_mode": "sample",
        },
    )
    assert forecast_with_model.status_code == 200
    rerun_payload = forecast_with_model.json()
    assert rerun_payload["summary"]["active_model_id"] == train_payload["model"]["model_id"]

    benchmark_response = client.get("/api/impact/benchmarks/latest", params={"region_id": "puget_sound"})
    assert benchmark_response.status_code == 200
    benchmark_payload = benchmark_response.json()
    assert benchmark_payload["region_id"] == "puget_sound"
    assert len(benchmark_payload["compared_strategies"]) == 3


def test_small_sequence_training_stays_candidate_when_evaluation_data_is_too_small(client) -> None:
    backfill_response = client.post(
        "/api/forecast/backfill",
        json={
            "region_id": "sf_bay_estuary",
            "source_mode": "sample",
            "days": 8,
            "debris_classes": ["low", "high"],
        },
    )
    assert backfill_response.status_code == 200

    dataset_response = client.post(
        "/api/ml/datasets/build",
        json={"region_id": "sf_bay_estuary", "label_type": "hotspot_presence", "max_forecast_runs": 8},
    )
    dataset_id = dataset_response.json()["dataset_id"]
    train_response = client.post(
        "/api/ml/train",
        json={
            "region_id": "sf_bay_estuary",
            "dataset_id": dataset_id,
            "architecture": "convlstm",
            "activate": True,
            "epochs": 1,
            "batch_size": 2,
        },
    )
    assert train_response.status_code == 200
    train_payload = train_response.json()
    assert train_payload["model"]["stage"] == "candidate"
    assert train_payload["model"]["is_active"] is False
    assert train_payload["metrics"]["promotion_eligible"] is False
    assert "sample_count<50" in train_payload["metrics"]["promotion_blockers"]


test_small_sequence_training_stays_candidate_when_evaluation_data_is_too_small = pytest.mark.slow(
    pytest.mark.ml(
        pytest.mark.backfill(test_small_sequence_training_stays_candidate_when_evaluation_data_is_too_small)
    )
)

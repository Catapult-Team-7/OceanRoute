from __future__ import annotations

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.ml, pytest.mark.backfill, pytest.mark.slow]


def test_stage2_backfill_and_model_admin_endpoints(client) -> None:
    backfill_response = client.post(
        "/api/forecast/backfill",
        json={
            "region_id": "long_island_sound",
            "source_mode": "sample",
            "days": 8,
            "debris_classes": ["low", "high"],
        },
    )
    assert backfill_response.status_code == 200
    backfill_payload = backfill_response.json()
    assert backfill_payload["region_id"] == "long_island_sound"
    assert backfill_payload["mode"] == "dataset_only"
    assert backfill_payload["runs_created"] >= 1
    assert backfill_payload["timestamps_planned"] >= backfill_payload["timestamps_processed"] >= 1
    assert backfill_payload["baseline_artifacts_created"] > 0
    assert backfill_payload["chunk_count"] >= 1
    assert backfill_payload["timings"]
    assert backfill_payload["timings"][0]["baseline_ms"] > 0
    assert backfill_payload["timings"][0]["artifact_write_ms"] > 0

    latest_response = client.get("/api/forecast/latest", params={"region_id": "long_island_sound"})
    assert latest_response.status_code == 404

    dataset_response = client.post(
        "/api/ml/datasets/export",
        json={
            "region_id": "long_island_sound",
            "max_forecast_runs": 8,
            "lookback_hours": 12,
            "target_horizons": [24, 48, 72],
            "label_strategy": "observed_or_proxy",
        },
    )
    assert dataset_response.status_code == 200
    dataset_payload = dataset_response.json()
    assert dataset_payload["sample_count"] > 0
    assert dataset_payload["zarr_uri"]
    assert dataset_payload["parquet_index_uri"]

    inspect_response = client.get(f"/api/ml/datasets/{dataset_payload['dataset_id']}")
    assert inspect_response.status_code == 200
    inspect_payload = inspect_response.json()
    assert inspect_payload["dataset"]["dataset_id"] == dataset_payload["dataset_id"]
    assert "sample_index_preview" in inspect_payload

    train_response = client.post(
        "/api/ml/train",
        json={
            "region_id": "long_island_sound",
            "dataset_id": dataset_payload["dataset_id"],
            "architecture": "linear_residual",
            "activate": False,
        },
    )
    assert train_response.status_code == 200
    train_payload = train_response.json()
    model_id = train_payload["model"]["model_id"]
    assert train_payload["model"]["stage"] == "candidate"

    evaluate_response = client.get(f"/api/ml/models/{model_id}/evaluate")
    assert evaluate_response.status_code == 200
    evaluate_payload = evaluate_response.json()
    assert evaluate_payload["model_id"] == model_id
    assert "route_uplift_pct" in evaluate_payload["metrics"]

    export_response = client.get(f"/api/ml/models/{model_id}/export")
    assert export_response.status_code == 200
    export_payload = export_response.json()
    assert export_payload["model_id"] == model_id
    assert export_payload["export_artifact_path"]

    promote_response = client.post(f"/api/ml/models/{model_id}/promote")
    assert promote_response.status_code == 200
    promote_payload = promote_response.json()
    assert promote_payload["promoted_model_id"] == model_id
    assert promote_payload["promoted"] is True

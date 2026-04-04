from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db import SessionLocal
from app.models import ForecastRunModel, ForecastStepModel, ObservationModel, RoutePlanModel


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_end_to_end_pipeline_persists_forecasts_routes_and_feedback(client) -> None:
    latest_before = client.get("/api/forecast/latest")
    assert latest_before.status_code == 404

    forecast_resp = client.post(
        "/api/forecast/run",
        json={"horizon_hours": 48, "debris_classes": ["low", "high"], "source_strength": 1.15, "seed": 5, "source_mode": "sample"},
    )
    assert forecast_resp.status_code == 200
    forecast_payload = forecast_resp.json()
    assert forecast_payload["steps_generated"] > 0
    assert forecast_payload["source_mode_used"] == "sample"
    assert forecast_payload["source_mode_requested"] == "sample"
    assert forecast_payload["is_fallback"] is False
    assert forecast_payload["top_hotspots"]
    assert forecast_payload["provenance"]["is_stale"] is False

    latest_resp = client.get("/api/forecast/latest", params={"horizon_hour": 24, "class": "low", "min_confidence": 0.2})
    assert latest_resp.status_code == 200
    latest_payload = latest_resp.json()
    assert latest_payload["steps"]
    assert all(item["horizon_hour"] == 24 for item in latest_payload["steps"])
    assert all(item["debris_class"] == "low" for item in latest_payload["steps"])
    assert latest_payload["provenance"]["source_mode_used"] == "sample"
    assert latest_payload["is_stale"] is False

    route_resp = client.post(
        "/api/route/optimize",
        json={
            "depot_lat": 37.8066,
            "depot_lon": -122.4659,
            "mission_hours": 4.0,
            "vessel_speed_kmh": 18.0,
            "fuel_burn_lph": 12.0,
            "target_horizon_hour": 24,
            "min_confidence": 0.2,
        },
    )
    assert route_resp.status_code == 200
    route_payload = route_resp.json()
    assert route_payload["mission_id"]
    assert route_payload["recommended_mode"] in {"collection", "recon"}
    assert route_payload["forecast_run_id"] == latest_payload["run_id"]
    assert route_payload["forecast_provenance"]["source_mode_used"] == "sample"

    route_by_id = client.get(f"/api/route/{route_payload['mission_id']}")
    assert route_by_id.status_code == 200
    assert route_by_id.json()["forecast_run_id"] == latest_payload["run_id"]

    geojson_resp = client.get("/api/export/geojson", params={"horizon_hour": 24})
    assert geojson_resp.status_code == 200
    geojson_payload = geojson_resp.json()
    assert geojson_payload["features"]
    assert geojson_payload["metadata"]["run_id"] == latest_payload["run_id"]
    assert geojson_payload["metadata"]["source_mode_used"] == "sample"

    obs_resp = client.post(
        "/api/observations/upload",
        json={
            "mission_id": route_payload["mission_id"],
            "observed_at": _now_iso(),
            "lat": 37.81,
            "lon": -122.37,
            "found_status": "found",
            "debris_class": "low",
            "estimated_kg": 3.2,
            "confidence": 0.8,
            "photo_url": "https://example.com/debris.jpg",
        },
    )
    assert obs_resp.status_code == 200
    assert obs_resp.json()["found_status"] == "found"

    log_resp = client.post(
        "/api/cleanup/log",
        json={
            "mission_id": route_payload["mission_id"],
            "completed_at": _now_iso(),
            "vessel_id": "sf-bay-pilot-vessel",
            "recommended_mode": route_payload["recommended_mode"],
            "predicted_kg_min": route_payload["expected_kg_min"],
            "predicted_kg_max": route_payload["expected_kg_max"],
            "collected_kg": 12.0,
            "vessel_distance_km": max(route_payload["expected_distance_km"], 1.0),
            "vessel_hours": max(route_payload["expected_duration_min"] / 60, 0.5),
            "hotspot_hits": 3,
            "hotspot_misses": 1,
            "false_search_km": 1.3,
            "fuel_liters": max(route_payload["estimated_fuel_liters"], 4.0),
            "route_deviation_reason": "Debris drifted closer to shoreline than forecast.",
        },
    )
    assert log_resp.status_code == 200

    impact_resp = client.get("/api/impact/dashboard")
    assert impact_resp.status_code == 200
    impact_payload = impact_resp.json()
    assert impact_payload["total_missions"] == 1
    assert impact_payload["kg_per_vessel_km"] > 0
    assert impact_payload["kg_per_hour"] > 0

    pdf_resp = client.get("/api/export/pdf-brief")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert latest_payload["run_id"].encode("utf-8") in pdf_resp.content

    benchmark_resp = client.get("/api/impact/benchmarks/latest")
    assert benchmark_resp.status_code == 200
    benchmark_payload = benchmark_resp.json()
    assert benchmark_payload["forecast_run_id"] == latest_payload["run_id"]
    assert len(benchmark_payload["compared_strategies"]) == 3

    with SessionLocal() as session:
        assert session.query(ForecastRunModel).count() == 1
        assert session.query(RoutePlanModel).count() == 1
        assert session.query(ObservationModel).count() == 1


def test_auto_ingest_failure_is_exposed_as_sample_fallback(client, monkeypatch) -> None:
    from app.services import ingest_service

    monkeypatch.setattr(
        ingest_service,
        "_load_live_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("upstream timeout")),
    )
    response = client.post(
        "/api/forecast/run",
        json={"horizon_hours": 24, "debris_classes": ["low"], "source_mode": "auto", "seed": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["source_mode_requested"] == "auto"
    assert payload["source_mode_used"] == "sample"
    assert payload["is_fallback"] is True
    assert payload["provenance"]["is_fallback"] is True

    latest_payload = client.get("/api/forecast/latest").json()
    assert any("fell back to sample" in note for note in latest_payload["source_notes"])


def test_stale_forecast_and_route_surface_freshness_metadata(client, db_session) -> None:
    generated_at = datetime.now(timezone.utc) - timedelta(hours=6)
    db_session.add(
        ForecastRunModel(
            id="stale-run",
            generated_at=generated_at,
            horizon_hours=24,
            pilot_region="sf_bay_estuary",
            source_mode_requested="auto",
            source_mode_used="sample",
            is_fallback=True,
            source_notes=["Live ingest failed and fell back to sample data: timeout"],
            summary={"step_count": 1},
        )
    )
    db_session.add(
        ForecastStepModel(
            run_id="stale-run",
            valid_at=generated_at + timedelta(hours=24),
            horizon_hour=24,
            cell_id="stale_cell",
            lat=37.81,
            lon=-122.37,
            debris_class="low",
            probability=0.82,
            expected_kg_min=4.4,
            expected_kg_max=7.9,
            uncertainty=0.18,
            confidence=0.82,
            beaching_risk=0.22,
            restricted=False,
        )
    )
    db_session.commit()

    latest_payload = client.get("/api/forecast/latest").json()
    assert latest_payload["run_id"] == "stale-run"
    assert latest_payload["is_stale"] is True
    assert latest_payload["is_fallback"] is True
    assert latest_payload["age_minutes"] >= 360

    route_payload = client.post(
        "/api/route/optimize",
        json={
            "depot_lat": 37.8066,
            "depot_lon": -122.4659,
            "mission_hours": 4.0,
            "vessel_speed_kmh": 18.0,
            "fuel_burn_lph": 12.0,
            "target_horizon_hour": 24,
            "min_confidence": 0.2,
        },
    ).json()
    assert route_payload["forecast_run_id"] == "stale-run"
    assert route_payload["forecast_is_stale"] is True
    assert route_payload["forecast_provenance"]["is_fallback"] is True

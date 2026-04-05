from __future__ import annotations

from app.services import ingest_service


def test_sample_ingest_produces_time_indexed_frames() -> None:
    context = ingest_service.load_operational_context(horizon_hours=30, seed=7, source_mode="sample")
    assert context.source_mode_used == "sample"
    assert context.is_fallback is False
    assert context.frames
    assert context.frames[0].horizon_hour == 1
    assert context.frames[-1].horizon_hour == 30
    assert all(frame.grid for frame in context.frames)


def test_auto_mode_falls_back_to_sample_when_live_ingest_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        ingest_service,
        "_load_live_observations",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    context = ingest_service.load_operational_context(horizon_hours=24, seed=3, source_mode="auto")
    assert context.source_mode_used == "sample"
    assert context.is_fallback is True
    assert any("fell back to sample" in note for note in context.source_notes)


def test_auto_mode_uses_hybrid_when_only_some_live_hours_are_available(monkeypatch) -> None:
    def _stub_live_observations(horizon_hours: int, region, generated_at):  # type: ignore[no-untyped-def]
        hours = [hour for hour in ingest_service._forecast_hours(horizon_hours) if hour in {1, 24}]
        live_vectors = {
            hour: [
                ingest_service.VectorObservation(
                    station_id="live-1",
                    name="Live Station",
                    lat=region.cells[0]["lat"],
                    lon=region.cells[0]["lon"],
                    speed=0.8,
                    direction_deg=45.0,
                )
            ]
            for hour in hours
        }
        return [{"id": "live-1", "name": "Live Station"}], live_vectors, []

    monkeypatch.setattr(ingest_service, "_load_live_observations", _stub_live_observations)
    context = ingest_service.load_operational_context(horizon_hours=24, seed=3, source_mode="auto")
    assert context.source_mode_used == "hybrid"
    assert context.is_fallback is True
    assert any("Live current coverage used" in note for note in context.source_notes)
    assert any("Sample current fill used" in note for note in context.source_notes)
    assert any("Sample wind fill used" in note for note in context.source_notes)

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
    monkeypatch.setattr(ingest_service, "_load_live_context", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    context = ingest_service.load_operational_context(horizon_hours=24, seed=3, source_mode="auto")
    assert context.source_mode_used == "sample"
    assert context.is_fallback is True
    assert any("fell back to sample" in note for note in context.source_notes)

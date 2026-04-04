from __future__ import annotations

from datetime import datetime, timezone

from app.config import settings
from app.schemas import ForecastProvenance


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def build_forecast_provenance(
    *,
    generated_at: datetime,
    horizon_hours: int,
    source_mode_requested: str,
    source_mode_used: str,
    is_fallback: bool,
    source_notes: list[str],
    baseline_engine: str | None = None,
    baseline_artifact_uri: str | None = None,
    model_id: str | None = None,
    model_architecture: str | None = None,
    model_dataset_version: str | None = None,
    training_scope: str | None = None,
    inference_service_version: str | None = None,
    prediction_artifact_uri: str | None = None,
) -> ForecastProvenance:
    generated_at = _normalize_timestamp(generated_at)
    age_minutes = max(0, int((_now() - generated_at).total_seconds() // 60))
    stale_after_minutes = settings.forecast_stale_after_minutes
    return ForecastProvenance(
        generated_at=generated_at,
        horizon_hours=horizon_hours,
        source_mode_requested=source_mode_requested,  # type: ignore[arg-type]
        source_mode_used=source_mode_used,  # type: ignore[arg-type]
        is_fallback=is_fallback,
        is_stale=age_minutes >= stale_after_minutes,
        age_minutes=age_minutes,
        stale_after_minutes=stale_after_minutes,
        source_notes=source_notes,
        baseline_engine=baseline_engine,  # type: ignore[arg-type]
        baseline_artifact_uri=baseline_artifact_uri,
        model_id=model_id,
        model_architecture=model_architecture,  # type: ignore[arg-type]
        model_dataset_version=model_dataset_version,
        training_scope=training_scope,  # type: ignore[arg-type]
        inference_service_version=inference_service_version,
        prediction_artifact_uri=prediction_artifact_uri,
    )

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import BaselineArtifactModel, ForecastRunModel
from app.schemas import ForecastRunRequest, HistoricalBackfillRequest, HistoricalBackfillResponse
from app.services.forecast_service import run_forecast
from app.services.region_service import REGIONS


BACKFILL_STRIDE_DAYS = 7


def _normalized_hour_now() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _backfill_dates(days: int) -> list[datetime]:
    end = _normalized_hour_now()
    dates: list[datetime] = []
    offset = 0
    while offset < days:
        dates.append(end - timedelta(days=offset))
        offset += BACKFILL_STRIDE_DAYS
    return list(reversed(dates))


def _existing_run_timestamps(region_id: str, db: Session) -> set[datetime]:
    rows = db.execute(
        select(ForecastRunModel.generated_at).where(ForecastRunModel.pilot_region == region_id)
    ).scalars().all()
    return {row.replace(minute=0, second=0, microsecond=0) for row in rows}


def run_historical_backfill(request: HistoricalBackfillRequest, db: Session) -> HistoricalBackfillResponse:
    if request.region_id == "all":
        region_ids = list(REGIONS)
    else:
        region_ids = [request.region_id]

    runs_created = 0
    artifact_count_before = db.execute(select(func.count(BaselineArtifactModel.artifact_id))).scalar_one()

    for region_id in region_ids:
        existing = _existing_run_timestamps(region_id, db)
        for index, generated_at in enumerate(_backfill_dates(request.days)):
            generated_at = generated_at.replace(minute=0, second=0, microsecond=0)
            if generated_at in existing:
                continue
            run_forecast(
                ForecastRunRequest(
                    region_id=region_id,
                    horizon_hours=72,
                    debris_classes=request.debris_classes,
                    source_strength=1.0,
                    seed=(index % 31) + 11,
                    source_mode=request.source_mode,
                ),
                db,
                generated_at_override=generated_at,
            )
            runs_created += 1
            existing.add(generated_at)

    artifact_count_after = db.execute(select(func.count(BaselineArtifactModel.artifact_id))).scalar_one()
    ready_run_count = db.execute(
        select(func.count(ForecastRunModel.id)).where(ForecastRunModel.pilot_region.in_(region_ids))
    ).scalar_one()
    return HistoricalBackfillResponse(
        region_id=request.region_id,
        source_mode=request.source_mode,
        days_backfilled=request.days,
        runs_created=runs_created,
        baseline_artifacts_created=int(artifact_count_after - artifact_count_before),
        dataset_ready_run_count=int(ready_run_count),
    )

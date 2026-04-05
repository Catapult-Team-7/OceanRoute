from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import FeedbackEventModel, MissionOutcomeModel, ObservationModel
from app.schemas import ImpactDashboard, MissionOutcome, MissionRecord, ObservationUpload
from app.services.artifact_service import write_debug_json


def _now() -> datetime:
    return datetime.now(timezone.utc)


def add_observation(payload: ObservationUpload, db: Session) -> ObservationUpload:
    db.add(
        ObservationModel(
            mission_id=payload.mission_id,
            observed_at=payload.observed_at,
            lat=payload.lat,
            lon=payload.lon,
            found_status=payload.found_status,
            debris_class=payload.debris_class,
            estimated_kg=payload.estimated_kg,
            confidence=payload.confidence,
            photo_url=payload.photo_url,
            note=payload.note,
            route_deviation_reason=payload.route_deviation_reason,
        )
    )
    if payload.mission_id:
        db.add(
            FeedbackEventModel(
                mission_id=payload.mission_id,
                event_type="observation",
                created_at=_now(),
                payload=payload.model_dump(mode="json"),
            )
        )
    db.commit()
    write_debug_json(
        "interim",
        "latest_observation.json",
        payload.model_dump(mode="json"),
    )
    return payload


def log_mission_outcome(payload: MissionOutcome, db: Session) -> MissionOutcome:
    existing = db.get(MissionOutcomeModel, payload.mission_id)
    if existing is None:
        existing = MissionOutcomeModel(
            mission_id=payload.mission_id,
            completed_at=payload.completed_at,
            vessel_id=payload.vessel_id,
            recommended_mode=payload.recommended_mode,
            predicted_kg_min=payload.predicted_kg_min,
            predicted_kg_max=payload.predicted_kg_max,
            collected_kg=payload.collected_kg,
            vessel_distance_km=payload.vessel_distance_km,
            vessel_hours=payload.vessel_hours,
            hotspot_hits=payload.hotspot_hits,
            hotspot_misses=payload.hotspot_misses,
            false_search_km=payload.false_search_km,
            fuel_liters=payload.fuel_liters,
            route_deviation_reason=payload.route_deviation_reason,
            notes=payload.notes,
        )
        db.add(existing)
    else:
        existing.completed_at = payload.completed_at
        existing.vessel_id = payload.vessel_id
        existing.recommended_mode = payload.recommended_mode
        existing.predicted_kg_min = payload.predicted_kg_min
        existing.predicted_kg_max = payload.predicted_kg_max
        existing.collected_kg = payload.collected_kg
        existing.vessel_distance_km = payload.vessel_distance_km
        existing.vessel_hours = payload.vessel_hours
        existing.hotspot_hits = payload.hotspot_hits
        existing.hotspot_misses = payload.hotspot_misses
        existing.false_search_km = payload.false_search_km
        existing.fuel_liters = payload.fuel_liters
        existing.route_deviation_reason = payload.route_deviation_reason
        existing.notes = payload.notes

    db.add(
        FeedbackEventModel(
            mission_id=payload.mission_id,
            event_type="mission_outcome",
            created_at=_now(),
            payload=payload.model_dump(mode="json"),
        )
    )
    db.commit()
    write_debug_json(
        "interim",
        "latest_mission_outcome.json",
        payload.model_dump(mode="json"),
    )
    return payload


def list_mission_records(db: Session) -> list[MissionRecord]:
    outcomes = db.execute(
        select(MissionOutcomeModel).order_by(MissionOutcomeModel.completed_at.desc())
    ).scalars().all()
    return [
        MissionRecord(
            mission_id=outcome.mission_id,
            date=outcome.completed_at,
            collected_kg=outcome.collected_kg,
            distance_km=outcome.vessel_distance_km,
            hours=outcome.vessel_hours,
            mode=outcome.recommended_mode,  # type: ignore[arg-type]
            notes=outcome.notes,
        )
        for outcome in outcomes
    ]


def save_mission_record(payload: MissionRecord, db: Session) -> MissionRecord:
    collected = max(payload.collected_kg, 0.0)
    mission_outcome = MissionOutcome(
        mission_id=payload.mission_id,
        completed_at=payload.date,
        vessel_id="local-ops-vessel",
        recommended_mode=payload.mode,
        predicted_kg_min=collected,
        predicted_kg_max=collected,
        collected_kg=collected,
        vessel_distance_km=max(payload.distance_km, 0.0),
        vessel_hours=max(payload.hours, 0.0),
        hotspot_hits=1 if collected > 0 else 0,
        hotspot_misses=0 if collected > 0 else 1,
        false_search_km=0.0 if collected > 0 else max(payload.distance_km, 0.0),
        fuel_liters=max(payload.hours, 0.0) * 12.0,
        notes=payload.notes,
    )
    saved = log_mission_outcome(mission_outcome, db)
    return MissionRecord(
        mission_id=saved.mission_id,
        date=saved.completed_at,
        collected_kg=saved.collected_kg,
        distance_km=saved.vessel_distance_km,
        hours=saved.vessel_hours,
        mode=saved.recommended_mode,  # type: ignore[arg-type]
        notes=saved.notes,
    )


def impact_dashboard(db: Session) -> ImpactDashboard:
    outcomes = db.execute(select(MissionOutcomeModel)).scalars().all()
    if not outcomes:
        return ImpactDashboard(
            total_missions=0,
            total_collected_kg=0.0,
            total_distance_km=0.0,
            total_hours=0.0,
            kg_per_vessel_km=0.0,
            kg_per_hour=0.0,
            hotspot_precision=0.0,
            false_search_distance_km=0.0,
            mission_hit_rate=0.0,
            latest_updated_at=None,
        )

    total_collected = sum(item.collected_kg for item in outcomes)
    total_distance = sum(item.vessel_distance_km for item in outcomes)
    total_hours = sum(item.vessel_hours for item in outcomes)
    total_hits = sum(item.hotspot_hits for item in outcomes)
    total_misses = sum(item.hotspot_misses for item in outcomes)
    false_search = sum(item.false_search_km for item in outcomes)
    return ImpactDashboard(
        total_missions=len(outcomes),
        total_collected_kg=round(total_collected, 3),
        total_distance_km=round(total_distance, 3),
        total_hours=round(total_hours, 3),
        kg_per_vessel_km=round(total_collected / max(total_distance, 1e-6), 4),
        kg_per_hour=round(total_collected / max(total_hours, 1e-6), 4),
        hotspot_precision=round(total_hits / max((total_hits + total_misses), 1), 4),
        false_search_distance_km=round(false_search, 3),
        mission_hit_rate=round(sum(1 for item in outcomes if item.hotspot_hits > 0) / len(outcomes), 4),
        latest_updated_at=max(item.completed_at for item in outcomes),
    )

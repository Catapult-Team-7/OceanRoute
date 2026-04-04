from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class ForecastRunModel(Base):
    __tablename__ = "forecast_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    horizon_hours: Mapped[int] = mapped_column(Integer)
    pilot_region: Mapped[str] = mapped_column(String(64), index=True)
    source_mode_requested: Mapped[str] = mapped_column(String(16))
    source_mode_used: Mapped[str] = mapped_column(String(16))
    is_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    source_notes: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)

    steps: Mapped[list["ForecastStepModel"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ForecastStepModel.horizon_hour",
    )


class ForecastStepModel(Base):
    __tablename__ = "forecast_steps"
    __table_args__ = (
        UniqueConstraint("run_id", "horizon_hour", "cell_id", "debris_class", name="uq_forecast_step_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("forecast_runs.id", ondelete="CASCADE"), index=True)
    valid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    horizon_hour: Mapped[int] = mapped_column(Integer, index=True)
    cell_id: Mapped[str] = mapped_column(String(64), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    debris_class: Mapped[str] = mapped_column(String(8), index=True)
    probability: Mapped[float] = mapped_column(Float)
    expected_kg_min: Mapped[float] = mapped_column(Float)
    expected_kg_max: Mapped[float] = mapped_column(Float)
    uncertainty: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    beaching_risk: Mapped[float] = mapped_column(Float)
    restricted: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped[ForecastRunModel] = relationship(back_populates="steps")


class RoutePlanModel(Base):
    __tablename__ = "route_plans"

    mission_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    forecast_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("forecast_runs.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    vessel_id: Mapped[str] = mapped_column(String(64), index=True)
    recommended_mode: Mapped[str] = mapped_column(String(16), index=True)
    target_horizon_hour: Mapped[int] = mapped_column(Integer)
    ordered_cell_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_kg_min: Mapped[float] = mapped_column(Float)
    expected_kg_max: Mapped[float] = mapped_column(Float)
    expected_distance_km: Mapped[float] = mapped_column(Float)
    expected_duration_min: Mapped[float] = mapped_column(Float)
    estimated_fuel_liters: Mapped[float] = mapped_column(Float)
    objective_score: Mapped[float] = mapped_column(Float)
    uncertainty_risk: Mapped[float] = mapped_column(Float)
    alternates: Mapped[list[str]] = mapped_column(JSON, default=list)
    legs: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, object]] = mapped_column("metadata", JSON, default=dict)


class ObservationModel(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mission_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    found_status: Mapped[str] = mapped_column(String(16), index=True)
    debris_class: Mapped[str | None] = mapped_column(String(8), nullable=True)
    estimated_kg: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    photo_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    route_deviation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class MissionOutcomeModel(Base):
    __tablename__ = "mission_outcomes"

    mission_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    vessel_id: Mapped[str] = mapped_column(String(64), index=True)
    recommended_mode: Mapped[str] = mapped_column(String(16), index=True)
    predicted_kg_min: Mapped[float] = mapped_column(Float, default=0.0)
    predicted_kg_max: Mapped[float] = mapped_column(Float, default=0.0)
    collected_kg: Mapped[float] = mapped_column(Float, default=0.0)
    vessel_distance_km: Mapped[float] = mapped_column(Float, default=0.0)
    vessel_hours: Mapped[float] = mapped_column(Float, default=0.0)
    hotspot_hits: Mapped[int] = mapped_column(Integer, default=0)
    hotspot_misses: Mapped[int] = mapped_column(Integer, default=0)
    false_search_km: Mapped[float] = mapped_column(Float, default=0.0)
    fuel_liters: Mapped[float] = mapped_column(Float, default=0.0)
    route_deviation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class FeedbackEventModel(Base):
    __tablename__ = "feedback_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mission_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)

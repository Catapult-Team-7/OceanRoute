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
    baseline_density: Mapped[float] = mapped_column(Float, default=0.0)
    ensemble_spread: Mapped[float] = mapped_column(Float, default=0.0)
    beaching_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    stokes_drift_u: Mapped[float] = mapped_column(Float, default=0.0)
    stokes_drift_v: Mapped[float] = mapped_column(Float, default=0.0)
    windage_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    restricted: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped[ForecastRunModel] = relationship(back_populates="steps")


class RoutePlanModel(Base):
    __tablename__ = "route_plans"

    mission_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    region_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
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


class DatasetArtifactModel(Base):
    __tablename__ = "dataset_artifacts"

    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    region_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    dataset_version: Mapped[str] = mapped_column(String(32), default="v2")
    label_type: Mapped[str] = mapped_column(String(32), index=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    feature_count: Mapped[int] = mapped_column(Integer, default=0)
    feature_names: Mapped[list[str]] = mapped_column(JSON, default=list)
    artifact_path: Mapped[str] = mapped_column(Text)
    zarr_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    parquet_index_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    manifest_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    splits_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_stats_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    split_counts: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    schema_versions: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    region_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    horizons: Mapped[list[int]] = mapped_column(JSON, default=list)
    input_channels: Mapped[list[str]] = mapped_column(JSON, default=list)
    target_channels: Mapped[list[str]] = mapped_column(JSON, default=list)
    tensor_shapes: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict[str, object]] = mapped_column("metadata", JSON, default=dict)


class ModelRegistryModel(Base):
    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    region_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    architecture: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    stage: Mapped[str] = mapped_column(String(16), default="candidate", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    training_scope: Mapped[str] = mapped_column(String(16), default="per_region", index=True)
    artifact_path: Mapped[str] = mapped_column(Text)
    dataset_id: Mapped[str] = mapped_column(String(64), index=True)
    dataset_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    trained_regions: Mapped[list[str]] = mapped_column(JSON, default=list)
    compatible_regions: Mapped[list[str]] = mapped_column(JSON, default=list)
    horizons: Mapped[list[int]] = mapped_column(JSON, default=list)
    input_channels: Mapped[list[str]] = mapped_column(JSON, default=list)
    output_heads: Mapped[list[str]] = mapped_column(JSON, default=list)
    normalization_stats_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_schema_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    best_checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    export_artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluation_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    framework: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metrics_json: Mapped[dict[str, object]] = mapped_column("metrics", JSON, default=dict)


class TrainingRunModel(Base):
    __tablename__ = "training_runs"

    training_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    region_id: Mapped[str] = mapped_column(String(64), index=True)
    dataset_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    architecture: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    training_scope: Mapped[str] = mapped_column(String(16), default="per_region", index=True)
    trained_regions: Mapped[list[str]] = mapped_column(JSON, default=list)
    compatible_regions: Mapped[list[str]] = mapped_column(JSON, default=list)
    horizons: Mapped[list[int]] = mapped_column(JSON, default=list)
    input_channels: Mapped[list[str]] = mapped_column(JSON, default=list)
    output_heads: Mapped[list[str]] = mapped_column(JSON, default=list)
    normalization_stats_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_schema_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    best_checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    export_artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluation_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    framework: Mapped[str | None] = mapped_column(String(32), nullable=True)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    best_val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    log_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[dict[str, object]] = mapped_column("metrics", JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class BaselineArtifactModel(Base):
    __tablename__ = "baseline_artifacts"

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    region_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    debris_class: Mapped[str] = mapped_column(String(8), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    forecast_valid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    horizon_hour: Mapped[int] = mapped_column(Integer, index=True)
    baseline_engine: Mapped[str] = mapped_column(String(32), index=True)
    source_mode_requested: Mapped[str] = mapped_column(String(16))
    source_mode_used: Mapped[str] = mapped_column(String(16))
    is_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    manifest_uri: Mapped[str] = mapped_column(Text)
    parquet_index_uri: Mapped[str] = mapped_column(Text)
    density_uri: Mapped[str] = mapped_column(Text)
    current_u_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_v_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    wind_u_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    wind_v_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    ensemble_spread_uri: Mapped[str] = mapped_column(Text)
    beaching_fraction_uri: Mapped[str] = mapped_column(Text)
    stokes_u_uri: Mapped[str] = mapped_column(Text)
    stokes_v_uri: Mapped[str] = mapped_column(Text)
    stokes_magnitude_uri: Mapped[str] = mapped_column(Text)
    grid_spec_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    forcing_refs_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    source_notes: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, object]] = mapped_column("metadata", JSON, default=dict)


class PredictionArtifactModel(Base):
    __tablename__ = "prediction_artifacts"

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    forecast_run_id: Mapped[str] = mapped_column(String(64), index=True)
    region_id: Mapped[str] = mapped_column(String(64), index=True)
    debris_class: Mapped[str] = mapped_column(String(8), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    model_architecture: Mapped[str] = mapped_column(String(32), index=True)
    model_dataset_version: Mapped[str] = mapped_column(String(32), index=True)
    training_scope: Mapped[str | None] = mapped_column(String(16), nullable=True)
    inference_service_version: Mapped[str] = mapped_column(String(32), default="v1")
    feature_artifact_uri: Mapped[str] = mapped_column(Text)
    hotspot_probability_uri: Mapped[str] = mapped_column(Text)
    expected_kg_uri: Mapped[str] = mapped_column(Text)
    uncertainty_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_schema_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalization_stats_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    parquet_index_uri: Mapped[str] = mapped_column(Text)
    manifest_uri: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, object]] = mapped_column("metadata", JSON, default=dict)

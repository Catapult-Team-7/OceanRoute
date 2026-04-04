"""Initialize OceanRoute operational schema.

Revision ID: 20260404_0001
Revises:
Create Date: 2026-04-04 13:50:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260404_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_hours", sa.Integer(), nullable=False),
        sa.Column("pilot_region", sa.String(length=64), nullable=False),
        sa.Column("source_mode_requested", sa.String(length=16), nullable=False),
        sa.Column("source_mode_used", sa.String(length=16), nullable=False),
        sa.Column("is_fallback", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source_notes", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecast_runs_generated_at", "forecast_runs", ["generated_at"])
    op.create_index("ix_forecast_runs_pilot_region", "forecast_runs", ["pilot_region"])

    op.create_table(
        "forecast_steps",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("valid_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_hour", sa.Integer(), nullable=False),
        sa.Column("cell_id", sa.String(length=64), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("debris_class", sa.String(length=8), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("expected_kg_min", sa.Float(), nullable=False),
        sa.Column("expected_kg_max", sa.Float(), nullable=False),
        sa.Column("uncertainty", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("beaching_risk", sa.Float(), nullable=False),
        sa.Column("restricted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(["run_id"], ["forecast_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "horizon_hour", "cell_id", "debris_class", name="uq_forecast_step_key"),
    )
    op.create_index("ix_forecast_steps_run_id", "forecast_steps", ["run_id"])
    op.create_index("ix_forecast_steps_valid_at", "forecast_steps", ["valid_at"])
    op.create_index("ix_forecast_steps_horizon_hour", "forecast_steps", ["horizon_hour"])
    op.create_index("ix_forecast_steps_cell_id", "forecast_steps", ["cell_id"])
    op.create_index("ix_forecast_steps_debris_class", "forecast_steps", ["debris_class"])

    op.create_table(
        "route_plans",
        sa.Column("mission_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("forecast_run_id", sa.String(length=64), nullable=True),
        sa.Column("vessel_id", sa.String(length=64), nullable=False),
        sa.Column("recommended_mode", sa.String(length=16), nullable=False),
        sa.Column("target_horizon_hour", sa.Integer(), nullable=False),
        sa.Column("ordered_cell_ids", sa.JSON(), nullable=False),
        sa.Column("expected_kg_min", sa.Float(), nullable=False),
        sa.Column("expected_kg_max", sa.Float(), nullable=False),
        sa.Column("expected_distance_km", sa.Float(), nullable=False),
        sa.Column("expected_duration_min", sa.Float(), nullable=False),
        sa.Column("estimated_fuel_liters", sa.Float(), nullable=False),
        sa.Column("objective_score", sa.Float(), nullable=False),
        sa.Column("uncertainty_risk", sa.Float(), nullable=False),
        sa.Column("alternates", sa.JSON(), nullable=False),
        sa.Column("legs", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["forecast_run_id"], ["forecast_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("mission_id"),
    )
    op.create_index("ix_route_plans_created_at", "route_plans", ["created_at"])
    op.create_index("ix_route_plans_forecast_run_id", "route_plans", ["forecast_run_id"])
    op.create_index("ix_route_plans_vessel_id", "route_plans", ["vessel_id"])
    op.create_index("ix_route_plans_recommended_mode", "route_plans", ["recommended_mode"])

    op.create_table(
        "observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("mission_id", sa.String(length=64), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("found_status", sa.String(length=16), nullable=False),
        sa.Column("debris_class", sa.String(length=8), nullable=True),
        sa.Column("estimated_kg", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("photo_url", sa.Text(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("route_deviation_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_observations_mission_id", "observations", ["mission_id"])
    op.create_index("ix_observations_observed_at", "observations", ["observed_at"])
    op.create_index("ix_observations_found_status", "observations", ["found_status"])

    op.create_table(
        "mission_outcomes",
        sa.Column("mission_id", sa.String(length=64), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("vessel_id", sa.String(length=64), nullable=False),
        sa.Column("recommended_mode", sa.String(length=16), nullable=False),
        sa.Column("predicted_kg_min", sa.Float(), nullable=False),
        sa.Column("predicted_kg_max", sa.Float(), nullable=False),
        sa.Column("collected_kg", sa.Float(), nullable=False),
        sa.Column("vessel_distance_km", sa.Float(), nullable=False),
        sa.Column("vessel_hours", sa.Float(), nullable=False),
        sa.Column("hotspot_hits", sa.Integer(), nullable=False),
        sa.Column("hotspot_misses", sa.Integer(), nullable=False),
        sa.Column("false_search_km", sa.Float(), nullable=False),
        sa.Column("fuel_liters", sa.Float(), nullable=False),
        sa.Column("route_deviation_reason", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("mission_id"),
    )
    op.create_index("ix_mission_outcomes_completed_at", "mission_outcomes", ["completed_at"])
    op.create_index("ix_mission_outcomes_vessel_id", "mission_outcomes", ["vessel_id"])
    op.create_index("ix_mission_outcomes_recommended_mode", "mission_outcomes", ["recommended_mode"])

    op.create_table(
        "feedback_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("mission_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_events_mission_id", "feedback_events", ["mission_id"])
    op.create_index("ix_feedback_events_event_type", "feedback_events", ["event_type"])
    op.create_index("ix_feedback_events_created_at", "feedback_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_feedback_events_created_at", table_name="feedback_events")
    op.drop_index("ix_feedback_events_event_type", table_name="feedback_events")
    op.drop_index("ix_feedback_events_mission_id", table_name="feedback_events")
    op.drop_table("feedback_events")

    op.drop_index("ix_mission_outcomes_recommended_mode", table_name="mission_outcomes")
    op.drop_index("ix_mission_outcomes_vessel_id", table_name="mission_outcomes")
    op.drop_index("ix_mission_outcomes_completed_at", table_name="mission_outcomes")
    op.drop_table("mission_outcomes")

    op.drop_index("ix_observations_found_status", table_name="observations")
    op.drop_index("ix_observations_observed_at", table_name="observations")
    op.drop_index("ix_observations_mission_id", table_name="observations")
    op.drop_table("observations")

    op.drop_index("ix_route_plans_recommended_mode", table_name="route_plans")
    op.drop_index("ix_route_plans_vessel_id", table_name="route_plans")
    op.drop_index("ix_route_plans_forecast_run_id", table_name="route_plans")
    op.drop_index("ix_route_plans_created_at", table_name="route_plans")
    op.drop_table("route_plans")

    op.drop_index("ix_forecast_steps_debris_class", table_name="forecast_steps")
    op.drop_index("ix_forecast_steps_cell_id", table_name="forecast_steps")
    op.drop_index("ix_forecast_steps_horizon_hour", table_name="forecast_steps")
    op.drop_index("ix_forecast_steps_valid_at", table_name="forecast_steps")
    op.drop_index("ix_forecast_steps_run_id", table_name="forecast_steps")
    op.drop_table("forecast_steps")

    op.drop_index("ix_forecast_runs_pilot_region", table_name="forecast_runs")
    op.drop_index("ix_forecast_runs_generated_at", table_name="forecast_runs")
    op.drop_table("forecast_runs")

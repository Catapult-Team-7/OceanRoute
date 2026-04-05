"""Add canonical data lake and inference runtime tables.

Revision ID: 20260404_0004
Revises: 20260404_0003
Create Date: 2026-04-04 21:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260404_0004"
down_revision = "20260404_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("dataset_artifacts", sa.Column("dataset_version", sa.String(length=32), nullable=False, server_default="v2"))
    op.add_column("dataset_artifacts", sa.Column("zarr_uri", sa.Text(), nullable=True))
    op.add_column("dataset_artifacts", sa.Column("parquet_index_uri", sa.Text(), nullable=True))
    op.add_column("dataset_artifacts", sa.Column("metadata_path", sa.Text(), nullable=True))
    op.add_column("dataset_artifacts", sa.Column("split_counts", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column("dataset_artifacts", sa.Column("schema_versions", sa.JSON(), nullable=False, server_default="{}"))

    op.add_column("model_registry", sa.Column("stage", sa.String(length=16), nullable=False, server_default="candidate"))
    op.add_column("model_registry", sa.Column("dataset_version", sa.String(length=32), nullable=True))
    op.add_column("model_registry", sa.Column("checkpoint_path", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("export_artifact_path", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("evaluation_path", sa.Text(), nullable=True))
    op.create_index("ix_model_registry_stage", "model_registry", ["stage"])

    op.add_column("training_runs", sa.Column("checkpoint_path", sa.Text(), nullable=True))
    op.add_column("training_runs", sa.Column("export_artifact_path", sa.Text(), nullable=True))
    op.add_column("training_runs", sa.Column("evaluation_path", sa.Text(), nullable=True))

    op.create_table(
        "baseline_artifacts",
        sa.Column("artifact_id", sa.String(length=64), nullable=False),
        sa.Column("region_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("debris_class", sa.String(length=8), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("forecast_valid_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_hour", sa.Integer(), nullable=False),
        sa.Column("baseline_engine", sa.String(length=32), nullable=False),
        sa.Column("source_mode_requested", sa.String(length=16), nullable=False),
        sa.Column("source_mode_used", sa.String(length=16), nullable=False),
        sa.Column("is_fallback", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("manifest_uri", sa.Text(), nullable=False),
        sa.Column("parquet_index_uri", sa.Text(), nullable=False),
        sa.Column("density_uri", sa.Text(), nullable=False),
        sa.Column("ensemble_spread_uri", sa.Text(), nullable=False),
        sa.Column("beaching_fraction_uri", sa.Text(), nullable=False),
        sa.Column("stokes_u_uri", sa.Text(), nullable=False),
        sa.Column("stokes_v_uri", sa.Text(), nullable=False),
        sa.Column("stokes_magnitude_uri", sa.Text(), nullable=False),
        sa.Column("grid_spec_json", sa.JSON(), nullable=False),
        sa.Column("forcing_refs_json", sa.JSON(), nullable=False),
        sa.Column("source_notes", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("artifact_id"),
    )
    op.create_index("ix_baseline_artifacts_region_id", "baseline_artifacts", ["region_id"])
    op.create_index("ix_baseline_artifacts_run_id", "baseline_artifacts", ["run_id"])
    op.create_index("ix_baseline_artifacts_debris_class", "baseline_artifacts", ["debris_class"])
    op.create_index("ix_baseline_artifacts_generated_at", "baseline_artifacts", ["generated_at"])
    op.create_index("ix_baseline_artifacts_forecast_valid_at", "baseline_artifacts", ["forecast_valid_at"])
    op.create_index("ix_baseline_artifacts_horizon_hour", "baseline_artifacts", ["horizon_hour"])
    op.create_index("ix_baseline_artifacts_baseline_engine", "baseline_artifacts", ["baseline_engine"])

    op.create_table(
        "prediction_artifacts",
        sa.Column("artifact_id", sa.String(length=64), nullable=False),
        sa.Column("forecast_run_id", sa.String(length=64), nullable=False),
        sa.Column("region_id", sa.String(length=64), nullable=False),
        sa.Column("debris_class", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("model_architecture", sa.String(length=32), nullable=False),
        sa.Column("model_dataset_version", sa.String(length=32), nullable=False),
        sa.Column("inference_service_version", sa.String(length=32), nullable=False, server_default="v1"),
        sa.Column("feature_artifact_uri", sa.Text(), nullable=False),
        sa.Column("hotspot_probability_uri", sa.Text(), nullable=False),
        sa.Column("expected_kg_uri", sa.Text(), nullable=False),
        sa.Column("parquet_index_uri", sa.Text(), nullable=False),
        sa.Column("manifest_uri", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("artifact_id"),
    )
    op.create_index("ix_prediction_artifacts_forecast_run_id", "prediction_artifacts", ["forecast_run_id"])
    op.create_index("ix_prediction_artifacts_region_id", "prediction_artifacts", ["region_id"])
    op.create_index("ix_prediction_artifacts_debris_class", "prediction_artifacts", ["debris_class"])
    op.create_index("ix_prediction_artifacts_created_at", "prediction_artifacts", ["created_at"])
    op.create_index("ix_prediction_artifacts_model_id", "prediction_artifacts", ["model_id"])
    op.create_index("ix_prediction_artifacts_model_architecture", "prediction_artifacts", ["model_architecture"])
    op.create_index("ix_prediction_artifacts_model_dataset_version", "prediction_artifacts", ["model_dataset_version"])


def downgrade() -> None:
    op.drop_index("ix_prediction_artifacts_model_dataset_version", table_name="prediction_artifacts")
    op.drop_index("ix_prediction_artifacts_model_architecture", table_name="prediction_artifacts")
    op.drop_index("ix_prediction_artifacts_model_id", table_name="prediction_artifacts")
    op.drop_index("ix_prediction_artifacts_created_at", table_name="prediction_artifacts")
    op.drop_index("ix_prediction_artifacts_debris_class", table_name="prediction_artifacts")
    op.drop_index("ix_prediction_artifacts_region_id", table_name="prediction_artifacts")
    op.drop_index("ix_prediction_artifacts_forecast_run_id", table_name="prediction_artifacts")
    op.drop_table("prediction_artifacts")

    op.drop_index("ix_baseline_artifacts_baseline_engine", table_name="baseline_artifacts")
    op.drop_index("ix_baseline_artifacts_horizon_hour", table_name="baseline_artifacts")
    op.drop_index("ix_baseline_artifacts_forecast_valid_at", table_name="baseline_artifacts")
    op.drop_index("ix_baseline_artifacts_generated_at", table_name="baseline_artifacts")
    op.drop_index("ix_baseline_artifacts_debris_class", table_name="baseline_artifacts")
    op.drop_index("ix_baseline_artifacts_run_id", table_name="baseline_artifacts")
    op.drop_index("ix_baseline_artifacts_region_id", table_name="baseline_artifacts")
    op.drop_table("baseline_artifacts")

    op.drop_column("training_runs", "evaluation_path")
    op.drop_column("training_runs", "export_artifact_path")
    op.drop_column("training_runs", "checkpoint_path")

    op.drop_index("ix_model_registry_stage", table_name="model_registry")
    op.drop_column("model_registry", "evaluation_path")
    op.drop_column("model_registry", "export_artifact_path")
    op.drop_column("model_registry", "checkpoint_path")
    op.drop_column("model_registry", "dataset_version")
    op.drop_column("model_registry", "stage")

    op.drop_column("dataset_artifacts", "schema_versions")
    op.drop_column("dataset_artifacts", "split_counts")
    op.drop_column("dataset_artifacts", "metadata_path")
    op.drop_column("dataset_artifacts", "parquet_index_uri")
    op.drop_column("dataset_artifacts", "zarr_uri")
    op.drop_column("dataset_artifacts", "dataset_version")

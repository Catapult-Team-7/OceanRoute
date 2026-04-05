"""Add multi-region route metadata and ML registry tables.

Revision ID: 20260404_0002
Revises: 20260404_0001
Create Date: 2026-04-04 18:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260404_0002"
down_revision = "20260404_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("route_plans", sa.Column("region_id", sa.String(length=64), nullable=True))
    op.create_index("ix_route_plans_region_id", "route_plans", ["region_id"])

    op.create_table(
        "dataset_artifacts",
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("region_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label_type", sa.String(length=32), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.Column("feature_names", sa.JSON(), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("dataset_id"),
    )
    op.create_index("ix_dataset_artifacts_region_id", "dataset_artifacts", ["region_id"])
    op.create_index("ix_dataset_artifacts_created_at", "dataset_artifacts", ["created_at"])
    op.create_index("ix_dataset_artifacts_label_type", "dataset_artifacts", ["label_type"])

    op.create_table(
        "model_registry",
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("region_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("architecture", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("model_id"),
    )
    op.create_index("ix_model_registry_region_id", "model_registry", ["region_id"])
    op.create_index("ix_model_registry_created_at", "model_registry", ["created_at"])
    op.create_index("ix_model_registry_architecture", "model_registry", ["architecture"])
    op.create_index("ix_model_registry_status", "model_registry", ["status"])
    op.create_index("ix_model_registry_is_active", "model_registry", ["is_active"])
    op.create_index("ix_model_registry_dataset_id", "model_registry", ["dataset_id"])

    op.create_table(
        "training_runs",
        sa.Column("training_run_id", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=True),
        sa.Column("region_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("architecture", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("training_run_id"),
    )
    op.create_index("ix_training_runs_model_id", "training_runs", ["model_id"])
    op.create_index("ix_training_runs_region_id", "training_runs", ["region_id"])
    op.create_index("ix_training_runs_dataset_id", "training_runs", ["dataset_id"])
    op.create_index("ix_training_runs_created_at", "training_runs", ["created_at"])
    op.create_index("ix_training_runs_architecture", "training_runs", ["architecture"])
    op.create_index("ix_training_runs_status", "training_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_training_runs_status", table_name="training_runs")
    op.drop_index("ix_training_runs_architecture", table_name="training_runs")
    op.drop_index("ix_training_runs_created_at", table_name="training_runs")
    op.drop_index("ix_training_runs_dataset_id", table_name="training_runs")
    op.drop_index("ix_training_runs_region_id", table_name="training_runs")
    op.drop_index("ix_training_runs_model_id", table_name="training_runs")
    op.drop_table("training_runs")

    op.drop_index("ix_model_registry_dataset_id", table_name="model_registry")
    op.drop_index("ix_model_registry_is_active", table_name="model_registry")
    op.drop_index("ix_model_registry_status", table_name="model_registry")
    op.drop_index("ix_model_registry_architecture", table_name="model_registry")
    op.drop_index("ix_model_registry_created_at", table_name="model_registry")
    op.drop_index("ix_model_registry_region_id", table_name="model_registry")
    op.drop_table("model_registry")

    op.drop_index("ix_dataset_artifacts_label_type", table_name="dataset_artifacts")
    op.drop_index("ix_dataset_artifacts_created_at", table_name="dataset_artifacts")
    op.drop_index("ix_dataset_artifacts_region_id", table_name="dataset_artifacts")
    op.drop_table("dataset_artifacts")

    op.drop_index("ix_route_plans_region_id", table_name="route_plans")
    op.drop_column("route_plans", "region_id")

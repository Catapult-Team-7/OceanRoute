"""Add deep training and runtime contract columns.

Revision ID: 20260404_0006
Revises: 20260404_0005
Create Date: 2026-04-04 23:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260404_0006"
down_revision = "20260404_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("dataset_artifacts", sa.Column("manifest_path", sa.Text(), nullable=True))
    op.add_column("dataset_artifacts", sa.Column("splits_path", sa.Text(), nullable=True))
    op.add_column("dataset_artifacts", sa.Column("feature_stats_path", sa.Text(), nullable=True))
    op.add_column("dataset_artifacts", sa.Column("region_ids", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("dataset_artifacts", sa.Column("horizons", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("dataset_artifacts", sa.Column("input_channels", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("dataset_artifacts", sa.Column("target_channels", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("dataset_artifacts", sa.Column("tensor_shapes", sa.JSON(), nullable=False, server_default="{}"))

    op.add_column("model_registry", sa.Column("training_scope", sa.String(length=16), nullable=False, server_default="per_region"))
    op.add_column("model_registry", sa.Column("trained_regions", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("model_registry", sa.Column("compatible_regions", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("model_registry", sa.Column("horizons", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("model_registry", sa.Column("input_channels", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("model_registry", sa.Column("output_heads", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("model_registry", sa.Column("normalization_stats_path", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("feature_schema_path", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("best_checkpoint_path", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("framework", sa.String(length=32), nullable=True))
    op.create_index("ix_model_registry_training_scope", "model_registry", ["training_scope"])

    op.add_column("training_runs", sa.Column("training_scope", sa.String(length=16), nullable=False, server_default="per_region"))
    op.add_column("training_runs", sa.Column("trained_regions", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("training_runs", sa.Column("compatible_regions", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("training_runs", sa.Column("horizons", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("training_runs", sa.Column("input_channels", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("training_runs", sa.Column("output_heads", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("training_runs", sa.Column("normalization_stats_path", sa.Text(), nullable=True))
    op.add_column("training_runs", sa.Column("feature_schema_path", sa.Text(), nullable=True))
    op.add_column("training_runs", sa.Column("best_checkpoint_path", sa.Text(), nullable=True))
    op.add_column("training_runs", sa.Column("framework", sa.String(length=32), nullable=True))
    op.add_column("training_runs", sa.Column("current_epoch", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("training_runs", sa.Column("best_val_loss", sa.Float(), nullable=True))
    op.add_column("training_runs", sa.Column("log_dir", sa.Text(), nullable=True))
    op.add_column("training_runs", sa.Column("status_message", sa.Text(), nullable=True))
    op.create_index("ix_training_runs_training_scope", "training_runs", ["training_scope"])

    op.add_column("prediction_artifacts", sa.Column("training_scope", sa.String(length=16), nullable=True))
    op.add_column("prediction_artifacts", sa.Column("uncertainty_uri", sa.Text(), nullable=True))
    op.add_column("prediction_artifacts", sa.Column("feature_schema_path", sa.Text(), nullable=True))
    op.add_column("prediction_artifacts", sa.Column("normalization_stats_path", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("prediction_artifacts", "normalization_stats_path")
    op.drop_column("prediction_artifacts", "feature_schema_path")
    op.drop_column("prediction_artifacts", "uncertainty_uri")
    op.drop_column("prediction_artifacts", "training_scope")

    op.drop_index("ix_training_runs_training_scope", table_name="training_runs")
    op.drop_column("training_runs", "status_message")
    op.drop_column("training_runs", "log_dir")
    op.drop_column("training_runs", "best_val_loss")
    op.drop_column("training_runs", "current_epoch")
    op.drop_column("training_runs", "framework")
    op.drop_column("training_runs", "best_checkpoint_path")
    op.drop_column("training_runs", "feature_schema_path")
    op.drop_column("training_runs", "normalization_stats_path")
    op.drop_column("training_runs", "output_heads")
    op.drop_column("training_runs", "input_channels")
    op.drop_column("training_runs", "horizons")
    op.drop_column("training_runs", "compatible_regions")
    op.drop_column("training_runs", "trained_regions")
    op.drop_column("training_runs", "training_scope")

    op.drop_index("ix_model_registry_training_scope", table_name="model_registry")
    op.drop_column("model_registry", "framework")
    op.drop_column("model_registry", "best_checkpoint_path")
    op.drop_column("model_registry", "feature_schema_path")
    op.drop_column("model_registry", "normalization_stats_path")
    op.drop_column("model_registry", "output_heads")
    op.drop_column("model_registry", "input_channels")
    op.drop_column("model_registry", "horizons")
    op.drop_column("model_registry", "compatible_regions")
    op.drop_column("model_registry", "trained_regions")
    op.drop_column("model_registry", "training_scope")

    op.drop_column("dataset_artifacts", "tensor_shapes")
    op.drop_column("dataset_artifacts", "target_channels")
    op.drop_column("dataset_artifacts", "input_channels")
    op.drop_column("dataset_artifacts", "horizons")
    op.drop_column("dataset_artifacts", "region_ids")
    op.drop_column("dataset_artifacts", "feature_stats_path")
    op.drop_column("dataset_artifacts", "splits_path")
    op.drop_column("dataset_artifacts", "manifest_path")

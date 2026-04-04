"""Add particle-baseline diagnostics to forecast steps.

Revision ID: 20260404_0003
Revises: 20260404_0002
Create Date: 2026-04-04 20:05:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260404_0003"
down_revision = "20260404_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("forecast_steps", sa.Column("baseline_density", sa.Float(), nullable=False, server_default="0"))
    op.add_column("forecast_steps", sa.Column("ensemble_spread", sa.Float(), nullable=False, server_default="0"))
    op.add_column("forecast_steps", sa.Column("beaching_fraction", sa.Float(), nullable=False, server_default="0"))
    op.add_column("forecast_steps", sa.Column("stokes_drift_u", sa.Float(), nullable=False, server_default="0"))
    op.add_column("forecast_steps", sa.Column("stokes_drift_v", sa.Float(), nullable=False, server_default="0"))
    op.add_column("forecast_steps", sa.Column("windage_fraction", sa.Float(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("forecast_steps", "windage_fraction")
    op.drop_column("forecast_steps", "stokes_drift_v")
    op.drop_column("forecast_steps", "stokes_drift_u")
    op.drop_column("forecast_steps", "beaching_fraction")
    op.drop_column("forecast_steps", "ensemble_spread")
    op.drop_column("forecast_steps", "baseline_density")

from __future__ import annotations

from datetime import datetime, timezone

from app.schemas import DriftBaselineInput, GridPoint, ResidualModelInput
from app.services.physics_service import run_drift_baseline
from app.services.residual_model_service import apply_residual_correction
from app.services.uncertainty_service import build_uncertainty


def build_grid() -> list[GridPoint]:
    return [
        GridPoint(
            cell_id="cell_1",
            lat=37.8,
            lon=-122.4,
            current_u=0.4,
            current_v=0.2,
            wind_u=0.3,
            wind_v=0.2,
            shoreline_proximity=0.3,
            water_temperature_c=14.2,
        ),
        GridPoint(
            cell_id="cell_2",
            lat=37.82,
            lon=-122.35,
            current_u=0.2,
            current_v=0.1,
            wind_u=0.25,
            wind_v=0.12,
            shoreline_proximity=0.6,
            water_temperature_c=13.6,
        ),
    ]


def test_baseline_produces_non_negative_density_for_low_and_high_classes() -> None:
    for debris_class in ["low", "high"]:
        output = run_drift_baseline(
            DriftBaselineInput(
                run_id="run",
                generated_at=datetime.now(timezone.utc),
                valid_at=datetime.now(timezone.utc),
                horizon_hour=24,
                debris_class=debris_class,  # type: ignore[arg-type]
                grid=build_grid(),
            )
        )
        assert output["cell_1"] >= 0
        assert output["cell_2"] >= 0


def test_residual_model_outputs_non_negative_values() -> None:
    corrected = apply_residual_correction(
        ResidualModelInput(
            run_id="run",
            debris_class="low",
            horizon_hour=24,
            baseline_density={"cell_1": 0.01, "cell_2": 0.2},
            currents={"cell_1": (1.0, -0.4), "cell_2": (0.3, 0.2)},
            winds={"cell_1": (-1.0, -1.0), "cell_2": (0.4, 0.1)},
            history_bias={"cell_1": 0.2, "cell_2": -0.1},
            shoreline={"cell_1": 0.3, "cell_2": 0.6},
        )
    )
    assert corrected["cell_1"] >= 0
    assert corrected["cell_2"] >= 0


def test_uncertainty_stays_bounded_and_confidence_is_inverse() -> None:
    uncertainty, confidence = build_uncertainty(
        build_grid(),
        {"cell_1": 1.2, "cell_2": 0.4},
        horizon_hour=48,
        source_mode_used="sample",
    )
    assert 0.0 <= uncertainty["cell_1"] <= 1.0
    assert 0.0 <= uncertainty["cell_2"] <= 1.0
    assert confidence["cell_1"] == round(1.0 - uncertainty["cell_1"], 4)

from __future__ import annotations

import numpy as np

from app.ml.training.metrics import compute_eval_metrics


def test_compute_eval_metrics_normalizes_singleton_channel_grids() -> None:
    batch = 2
    horizons = [24, 48, 72]
    grid = (8, 8)

    probabilities = np.full((batch, len(horizons), *grid), 0.5, dtype=np.float32)
    predicted_kg = np.full((batch, len(horizons), *grid), 2.0, dtype=np.float32)
    predicted_uncertainty = np.full((batch, len(horizons), *grid), 0.2, dtype=np.float32)
    baseline_density = np.full((batch, len(horizons), *grid), 1.8, dtype=np.float32)

    target_probability = np.full((batch, len(horizons), 1, *grid), 0.5, dtype=np.float32)
    target_expected_kg = np.full((batch, len(horizons), 1, *grid), 2.0, dtype=np.float32)
    target_uncertainty = np.full((batch, len(horizons), 1, *grid), 0.2, dtype=np.float32)

    metrics = compute_eval_metrics(
        probabilities=probabilities,
        predicted_kg=predicted_kg,
        predicted_uncertainty=predicted_uncertainty,
        baseline_density=baseline_density,
        target_probability=target_probability,
        target_expected_kg=target_expected_kg,
        target_uncertainty=target_uncertainty,
        horizons=horizons,
    )

    assert metrics["brier_score"] == 0.0
    assert metrics["kg_mae"] == 0.0
    assert metrics["kg_rmse"] == 0.0
    assert metrics["uncertainty_mae"] == 0.0
    assert "positive_probability_mean" in metrics
    assert "top10_predicted_scores" in metrics
    assert "top10_true_hit_count" in metrics


def test_compute_eval_metrics_broadcasts_single_horizon_baseline_density() -> None:
    batch = 2
    horizons = [24, 48, 72]
    grid = (4, 4)

    probabilities = np.full((batch, len(horizons), *grid), 0.6, dtype=np.float32)
    predicted_kg = np.full((batch, len(horizons), *grid), 1.5, dtype=np.float32)
    predicted_uncertainty = np.full((batch, len(horizons), *grid), 0.3, dtype=np.float32)
    baseline_density = np.full((batch, 1, *grid), 1.25, dtype=np.float32)

    target_probability = np.full((batch, len(horizons), *grid), 0.6, dtype=np.float32)
    target_expected_kg = np.full((batch, len(horizons), *grid), 1.5, dtype=np.float32)
    target_uncertainty = np.full((batch, len(horizons), *grid), 0.3, dtype=np.float32)

    metrics = compute_eval_metrics(
        probabilities=probabilities,
        predicted_kg=predicted_kg,
        predicted_uncertainty=predicted_uncertainty,
        baseline_density=baseline_density,
        target_probability=target_probability,
        target_expected_kg=target_expected_kg,
        target_uncertainty=target_uncertainty,
        horizons=horizons,
    )

    assert metrics["kg_mae"] == 0.0
    assert isinstance(metrics["route_uplift_pct"], float)

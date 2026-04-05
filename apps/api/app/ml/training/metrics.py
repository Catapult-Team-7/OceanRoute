from __future__ import annotations

import math

import numpy as np


def _normalize_grid_tensor(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 5 and arr.shape[2] == 1:
        arr = np.squeeze(arr, axis=2)
    return arr


def _align_horizon_axis(tensor: np.ndarray, *, target_horizons: int, name: str) -> np.ndarray:
    if tensor.ndim != 4:
        raise AssertionError(f"Expected {name} to normalize to [B,H,Y,X], got {tensor.shape}")
    if tensor.shape[1] == target_horizons:
        return tensor
    if tensor.shape[1] == 1 and target_horizons > 1:
        return np.repeat(tensor, target_horizons, axis=1)
    raise AssertionError(
        f"{name} horizon mismatch after normalization: {tensor.shape[1]} vs expected {target_horizons}"
    )


def precision_at_k(probabilities: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    flat_probs = probabilities.reshape(-1)
    flat_targets = targets.reshape(-1)
    k = min(k, flat_probs.size)
    if k <= 0:
        return 0.0
    top_idx = np.argpartition(flat_probs, -k)[-k:]
    return float(flat_targets[top_idx].mean())


def recall_at_k(probabilities: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    positives = float(np.sum(targets > 0.5))
    if positives <= 0:
        return 0.0
    flat_probs = probabilities.reshape(-1)
    flat_targets = targets.reshape(-1)
    k = min(k, flat_probs.size)
    top_idx = np.argpartition(flat_probs, -k)[-k:]
    return float(np.sum(flat_targets[top_idx] > 0.5) / positives)


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probabilities - targets) ** 2))


def kg_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def kg_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(math.sqrt(np.mean((prediction - target) ** 2)))


def calibration_summary(probabilities: np.ndarray, targets: np.ndarray, horizons: list[int]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for index, horizon in enumerate(horizons):
        summary[f"h{horizon}"] = round(float(np.mean(probabilities[:, index] - targets[:, index])), 4)
    return summary


def route_uplift_pct(predicted_kg: np.ndarray, baseline_density: np.ndarray, targets: np.ndarray) -> float:
    predicted_top = float(np.sum(np.sort(predicted_kg.reshape(-1))[-10:]))
    baseline_top = float(np.sum(np.sort(baseline_density.reshape(-1))[-10:]))
    actual_top = float(np.sum(np.sort(targets.reshape(-1))[-10:]))
    predicted_error = predicted_top - actual_top
    baseline_error = baseline_top - actual_top
    if abs(baseline_error) < 1e-6:
        return 0.0
    return float(((baseline_error - predicted_error) / abs(baseline_error)) * 100.0)


def compute_eval_metrics(
    *,
    probabilities: np.ndarray,
    predicted_kg: np.ndarray,
    predicted_uncertainty: np.ndarray,
    baseline_density: np.ndarray,
    target_probability: np.ndarray,
    target_expected_kg: np.ndarray,
    target_uncertainty: np.ndarray,
    horizons: list[int],
) -> dict[str, float | str]:
    probabilities = _normalize_grid_tensor(probabilities)
    predicted_kg = _normalize_grid_tensor(predicted_kg)
    predicted_uncertainty = _normalize_grid_tensor(predicted_uncertainty)
    baseline_density = _normalize_grid_tensor(baseline_density)
    target_probability = _normalize_grid_tensor(target_probability)
    target_expected_kg = _normalize_grid_tensor(target_expected_kg)
    target_uncertainty = _normalize_grid_tensor(target_uncertainty)

    target_horizons = target_expected_kg.shape[1] if target_expected_kg.ndim == 4 else -1

    assert probabilities.ndim == 4, f"Expected probabilities to normalize to [B,H,Y,X], got {probabilities.shape}"
    assert predicted_kg.ndim == 4, f"Expected predicted_kg to normalize to [B,H,Y,X], got {predicted_kg.shape}"
    assert predicted_uncertainty.ndim == 4, (
        f"Expected predicted_uncertainty to normalize to [B,H,Y,X], got {predicted_uncertainty.shape}"
    )
    assert baseline_density.ndim == 4, f"Expected baseline_density to normalize to [B,H,Y,X], got {baseline_density.shape}"
    assert target_probability.ndim == 4, (
        f"Expected target_probability to normalize to [B,H,Y,X], got {target_probability.shape}"
    )
    assert target_expected_kg.ndim == 4, (
        f"Expected target_expected_kg to normalize to [B,H,Y,X], got {target_expected_kg.shape}"
    )
    assert target_uncertainty.ndim == 4, (
        f"Expected target_uncertainty to normalize to [B,H,Y,X], got {target_uncertainty.shape}"
    )

    probabilities = _align_horizon_axis(probabilities, target_horizons=target_horizons, name="probabilities")
    predicted_kg = _align_horizon_axis(predicted_kg, target_horizons=target_horizons, name="predicted_kg")
    predicted_uncertainty = _align_horizon_axis(
        predicted_uncertainty,
        target_horizons=target_horizons,
        name="predicted_uncertainty",
    )
    baseline_density = _align_horizon_axis(baseline_density, target_horizons=target_horizons, name="baseline_density")
    target_probability = _align_horizon_axis(target_probability, target_horizons=target_horizons, name="target_probability")
    target_expected_kg = _align_horizon_axis(
        target_expected_kg,
        target_horizons=target_horizons,
        name="target_expected_kg",
    )
    target_uncertainty = _align_horizon_axis(
        target_uncertainty,
        target_horizons=target_horizons,
        name="target_uncertainty",
    )

    assert probabilities.shape == target_probability.shape, (
        f"Probability shape mismatch after normalization: {probabilities.shape} vs {target_probability.shape}"
    )
    assert predicted_kg.shape == target_expected_kg.shape, (
        f"Yield shape mismatch after normalization: {predicted_kg.shape} vs {target_expected_kg.shape}"
    )
    assert predicted_uncertainty.shape == target_uncertainty.shape, (
        f"Uncertainty shape mismatch after normalization: {predicted_uncertainty.shape} vs {target_uncertainty.shape}"
    )
    assert baseline_density.shape == target_expected_kg.shape, (
        f"Baseline density shape mismatch after normalization: {baseline_density.shape} vs {target_expected_kg.shape}"
    )

    calibration = calibration_summary(probabilities, target_probability, horizons)
    uncertainty_error = float(np.mean(np.abs(predicted_uncertainty - target_uncertainty)))
    return {
        "precision_at_10": round(precision_at_k(probabilities, target_probability, k=10), 4),
        "recall_at_10": round(recall_at_k(probabilities, target_probability, k=10), 4),
        "brier_score": round(brier_score(probabilities, target_probability), 4),
        "kg_mae": round(kg_mae(predicted_kg, target_expected_kg), 4),
        "kg_rmse": round(kg_rmse(predicted_kg, target_expected_kg), 4),
        "uncertainty_mae": round(uncertainty_error, 4),
        "route_uplift_pct": round(route_uplift_pct(predicted_kg, baseline_density, target_expected_kg), 4),
        "calibration_summary": str(calibration),
    }

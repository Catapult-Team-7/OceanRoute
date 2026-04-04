from __future__ import annotations

import math

import numpy as np


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

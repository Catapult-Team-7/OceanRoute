from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


TORCH_AVAILABLE = torch is not None and F is not None


def focal_bce_loss(prediction, target, gamma: float = 2.0):  # pragma: no cover - deep path
    base = F.binary_cross_entropy(prediction, target, reduction="none")
    modulating = (1.0 - torch.exp(-base)) ** gamma
    return torch.mean(modulating * base)


def multitask_loss(
    *,
    prediction_probability,
    prediction_expected_kg,
    prediction_uncertainty,
    target_probability,
    target_expected_kg,
    target_uncertainty,
    hotspot_loss: str = "focal",
    focal_gamma: float = 2.0,
    positive_class_weight: float | None = None,
):  # pragma: no cover - deep path
    if not TORCH_AVAILABLE:
        raise ValueError("torch is required to compute training losses.")
    if hotspot_loss == "weighted_bce":
        positive_weight = float(positive_class_weight or 2.0)
        weights = torch.where(
            target_probability > 0.5,
            torch.full_like(target_probability, positive_weight),
            torch.ones_like(target_probability),
        )
        hotspot_value = F.binary_cross_entropy(prediction_probability, target_probability, weight=weights)
    elif hotspot_loss == "bce":
        hotspot_value = F.binary_cross_entropy(prediction_probability, target_probability)
    else:
        hotspot_value = focal_bce_loss(prediction_probability, target_probability, gamma=focal_gamma)
    yield_loss = F.huber_loss(prediction_expected_kg, target_expected_kg)
    uncertainty_loss = F.mse_loss(prediction_uncertainty, target_uncertainty)
    total = hotspot_value + yield_loss + (0.25 * uncertainty_loss)
    return total, {
        "hotspot_loss": float(hotspot_value.detach().cpu().item()),
        "yield_loss": float(yield_loss.detach().cpu().item()),
        "uncertainty_loss": float(uncertainty_loss.detach().cpu().item()),
    }

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
    use_focal: bool = False,
):  # pragma: no cover - deep path
    if not TORCH_AVAILABLE:
        raise ValueError("torch is required to compute training losses.")
    hotspot_loss = focal_bce_loss(prediction_probability, target_probability) if use_focal else F.binary_cross_entropy(
        prediction_probability,
        target_probability,
    )
    yield_loss = F.huber_loss(prediction_expected_kg, target_expected_kg)
    uncertainty_loss = F.mse_loss(prediction_uncertainty, target_uncertainty)
    total = hotspot_loss + yield_loss + (0.25 * uncertainty_loss)
    return total, {
        "hotspot_loss": float(hotspot_loss.detach().cpu().item()),
        "yield_loss": float(yield_loss.detach().cpu().item()),
        "uncertainty_loss": float(uncertainty_loss.detach().cpu().item()),
    }

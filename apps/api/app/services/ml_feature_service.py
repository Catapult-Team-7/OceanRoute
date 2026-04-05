from __future__ import annotations

from app.schemas import ForecastStep


FEATURE_NAMES = [
    "probability",
    "baseline_density",
    "uncertainty",
    "confidence",
    "beaching_risk",
    "beaching_fraction",
    "ensemble_spread",
    "expected_mid_kg",
    "horizon_norm",
    "is_high_windage",
    "stokes_drift_mag",
    "windage_fraction",
]


def build_step_features(step: ForecastStep) -> dict[str, float]:
    expected_mid_kg = (step.expected_kg_min + step.expected_kg_max) / 2
    stokes_drift_mag = ((step.stokes_drift_u**2) + (step.stokes_drift_v**2)) ** 0.5
    return {
        "probability": step.probability,
        "baseline_density": step.baseline_density,
        "uncertainty": step.uncertainty,
        "confidence": step.confidence,
        "beaching_risk": step.beaching_risk,
        "beaching_fraction": step.beaching_fraction,
        "ensemble_spread": step.ensemble_spread,
        "expected_mid_kg": expected_mid_kg,
        "horizon_norm": step.horizon_hour / 72,
        "is_high_windage": 1.0 if step.debris_class == "high" else 0.0,
        "stokes_drift_mag": stokes_drift_mag,
        "windage_fraction": step.windage_fraction,
    }


def ordered_feature_vector(feature_map: dict[str, float]) -> list[float]:
    return [float(feature_map[name]) for name in FEATURE_NAMES]

from __future__ import annotations

from random import Random

from app.schemas import DriftBaselineInput


CLASS_WINDAGE = {
    "low": 0.05,
    "high": 0.14,
}


def run_drift_baseline(payload: DriftBaselineInput) -> dict[str, float]:
    rng = Random(1000 + payload.horizon_hour + len(payload.grid) + int(payload.source_strength * 100))
    windage = payload.windage_factor + CLASS_WINDAGE[payload.debris_class]
    horizon_decay = max(0.45, 1.0 - (payload.horizon_hour / 110))
    density: dict[str, float] = {}
    for point in payload.grid:
        advection = (point.current_u * 0.68) + (point.current_v * 0.34)
        wind_push = (point.wind_u * 0.07) + (point.wind_v * 0.05 * windage)
        shoreline_accumulation = point.shoreline_proximity * (0.16 if payload.debris_class == "low" else 0.22)
        temperature_bonus = (point.water_temperature_c or 13.5) * 0.002
        diffusion = rng.uniform(-payload.diffusion_sigma, payload.diffusion_sigma) * horizon_decay
        value = (payload.source_strength * horizon_decay) + advection + wind_push + shoreline_accumulation + temperature_bonus + diffusion
        density[point.cell_id] = max(0.0, round(value, 4))
    return density

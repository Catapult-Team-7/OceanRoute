from __future__ import annotations

from app.schemas import ResidualModelInput


def apply_residual_correction(payload: ResidualModelInput) -> dict[str, float]:
    corrected: dict[str, float] = {}
    horizon_penalty = max(0.0, payload.horizon_hour - 24) * 0.0025
    for cell_id, base in payload.baseline_density.items():
        current_u, current_v = payload.currents[cell_id]
        wind_u, wind_v = payload.winds[cell_id]
        history = payload.history_bias.get(cell_id, 0.0)
        shoreline = payload.shoreline.get(cell_id, 0.0)
        residual = (
            (wind_u * 0.065)
            + (wind_v * 0.055)
            - (current_u * 0.025)
            + (current_v * 0.03)
            + (history * 0.28)
            + (shoreline * 0.06)
            - horizon_penalty
        )
        corrected[cell_id] = max(0.0, round(base + residual, 4))
    return corrected

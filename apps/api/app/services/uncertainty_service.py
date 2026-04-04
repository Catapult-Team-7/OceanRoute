from __future__ import annotations

from app.schemas import GridPoint


def build_uncertainty(
    grid: list[GridPoint],
    corrected_density: dict[str, float],
    *,
    horizon_hour: int,
    source_mode_used: str,
) -> tuple[dict[str, float], dict[str, float]]:
    uncertainty: dict[str, float] = {}
    confidence: dict[str, float] = {}
    max_density = max(corrected_density.values()) if corrected_density else 1.0
    horizon_factor = min(0.22, horizon_hour * 0.0045)
    source_penalty = 0.09 if source_mode_used == "sample" else 0.03
    for point in grid:
        density_ratio = corrected_density[point.cell_id] / max_density if max_density > 0 else 0.0
        sigma = 0.11 + (point.shoreline_proximity * 0.24) + ((1 - density_ratio) * 0.18) + horizon_factor + source_penalty
        sigma = max(0.06, min(0.95, sigma))
        conf = max(0.05, min(0.99, 1.0 - sigma))
        uncertainty[point.cell_id] = round(sigma, 4)
        confidence[point.cell_id] = round(conf, 4)
    return uncertainty, confidence

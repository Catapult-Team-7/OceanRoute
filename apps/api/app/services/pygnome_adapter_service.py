from __future__ import annotations

from app.schemas import DriftBaselineInput
from app.services.physics_service import DriftBaselineDiagnostics, run_drift_baseline_ensemble

try:  # pragma: no cover - optional dependency
    import gnome  # type: ignore  # noqa: F401

    PYGNOME_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PYGNOME_AVAILABLE = False


def run_pygnome_compatible_baseline(payload: DriftBaselineInput) -> DriftBaselineDiagnostics:
    # Local development falls back to the custom particle integrator when PyGNOME is unavailable.
    # The return contract is kept identical so historical backfill, dataset export, and runtime
    # inference can consume the same artifact shape regardless of baseline engine.
    return run_drift_baseline_ensemble(payload)

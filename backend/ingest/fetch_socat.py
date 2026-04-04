from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_socat_observations(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(
            [
                {"year": 2025, "month": 10, "day": 15, "latitude": 33.5, "longitude": -147.0, "fCO2rec": 398.2},
                {"year": 2025, "month": 10, "day": 16, "latitude": -22.0, "longitude": -10.0, "fCO2rec": 421.3},
            ]
        )
    return pd.read_csv(path)

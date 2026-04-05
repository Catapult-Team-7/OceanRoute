from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from app.services.data_lake_service import read_tensor, write_baseline_tensors
from app.services.region_service import get_region_definition


def test_read_tensor_supports_legacy_dataset_layout() -> None:
    dataset_root = Path(__file__).resolve().parent / "test-data-debug3" / "legacy-dataset-loader"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    group_path = dataset_root / "tensors.zarr"
    group_path.mkdir(parents=True)

    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(dataset_root / "X.npy", expected)

    loaded = read_tensor(f"{group_path}::X")
    assert np.array_equal(loaded, expected)


def test_write_baseline_tensors_supports_compact_group_uris() -> None:
    region = get_region_definition("sf_bay_estuary")
    values = {str(cell["cell_id"]): float(index + 1) for index, cell in enumerate(region.cells)}
    grid_spec, tensor_uris, _ = write_baseline_tensors(
        region_id="sf_bay_estuary",
        generated_at=datetime.now(timezone.utc),
        run_id="compact-run",
        debris_class="low",
        horizon_hour=24,
        density_by_cell=values,
        current_u_by_cell=values,
        current_v_by_cell=values,
        wind_u_by_cell=values,
        wind_v_by_cell=values,
        ensemble_spread_by_cell=values,
        beaching_fraction_by_cell=values,
        stokes_u_by_cell=values,
        stokes_v_by_cell=values,
        compact_group=True,
    )

    assert tensor_uris["density_uri"].endswith("::density")
    assert tensor_uris["stokes_magnitude_uri"].endswith("::stokes_magnitude")
    density = read_tensor(tensor_uris["density_uri"])
    assert density.shape == (grid_spec.height, grid_spec.width)

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from app.services.data_lake_service import read_tensor


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

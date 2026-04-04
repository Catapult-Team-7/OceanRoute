from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.data_lake_service import read_table_rows, read_tensor


@dataclass(frozen=True)
class DatasetBundle:
    root_path: Path
    metadata: dict[str, Any]
    feature_stats: dict[str, dict[str, float]]
    manifest_rows: list[dict[str, Any]]
    split_rows: list[dict[str, Any]]
    x_tensor: np.ndarray
    y_probability: np.ndarray
    y_expected_kg: np.ndarray
    y_uncertainty: np.ndarray


def load_dataset_bundle(dataset_root: str | Path) -> DatasetBundle:
    root_path = Path(dataset_root)
    metadata = json.loads((root_path / "metadata.json").read_text(encoding="utf-8"))
    feature_stats = json.loads(Path(metadata["feature_stats_path"]).read_text(encoding="utf-8"))
    manifest_rows = read_table_rows(str(metadata["manifest_path"]))
    split_rows = read_table_rows(str(metadata["splits_path"]))
    tensor_uris = metadata["tensor_uris"]
    return DatasetBundle(
        root_path=root_path,
        metadata=metadata,
        feature_stats=feature_stats,
        manifest_rows=manifest_rows,
        split_rows=split_rows,
        x_tensor=read_tensor(tensor_uris["X"]).astype(np.float32),
        y_probability=read_tensor(tensor_uris["Y_hotspot_probability"]).astype(np.float32),
        y_expected_kg=read_tensor(tensor_uris["Y_expected_kg"]).astype(np.float32),
        y_uncertainty=read_tensor(tensor_uris["Y_uncertainty"]).astype(np.float32),
    )


class SequenceDatasetReader:
    def __init__(
        self,
        bundle: DatasetBundle,
        *,
        split: str,
        region_ids: list[str] | None = None,
        horizons: list[int] | None = None,
    ) -> None:
        split_by_sample_id = {str(row["sample_id"]): row for row in bundle.split_rows if row.get("split") == split}
        requested_regions = set(region_ids or [])
        dataset_horizons = list(bundle.metadata.get("horizons", [24, 48, 72]))
        selected_horizons = horizons or dataset_horizons
        self.horizon_indexes = [dataset_horizons.index(horizon) for horizon in selected_horizons]
        self.bundle = bundle
        self.sample_rows = []
        for row in bundle.manifest_rows:
            sample_id = str(row["sample_id"])
            if sample_id not in split_by_sample_id:
                continue
            if requested_regions and str(row["region_id"]) not in requested_regions:
                continue
            merged = {**row, **split_by_sample_id[sample_id]}
            self.sample_rows.append(merged)
        self.selected_horizons = selected_horizons

    def __len__(self) -> int:
        return len(self.sample_rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.sample_rows[index]
        sample_index = int(row["sample_index"])
        return {
            "sample_id": str(row["sample_id"]),
            "sample_index": sample_index,
            "region_id": str(row["region_id"]),
            "run_id": str(row["run_id"]),
            "debris_class": str(row["debris_class"]),
            "inputs": self.bundle.x_tensor[sample_index],
            "target_probability": self.bundle.y_probability[sample_index][self.horizon_indexes],
            "target_expected_kg": self.bundle.y_expected_kg[sample_index][self.horizon_indexes],
            "target_uncertainty": self.bundle.y_uncertainty[sample_index][self.horizon_indexes],
        }

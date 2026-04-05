from __future__ import annotations

from typing import Any

import numpy as np

from app.ml.training.dataset_reader import DatasetBundle, SequenceDatasetReader

try:  # pragma: no cover - optional deep-learning dependency
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - exercised when torch missing
    torch = None
    DataLoader = None
    Dataset = object

try:  # pragma: no cover - optional dependency
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    try:
        import pytorch_lightning as pl  # type: ignore[no-redef]
    except Exception:  # pragma: no cover
        pl = None


TORCH_AVAILABLE = torch is not None and DataLoader is not None
LIGHTNING_AVAILABLE = pl is not None


def _channel_stats(feature_stats: dict[str, dict[str, float]], channel_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray([feature_stats[name]["mean"] for name in channel_names], dtype=np.float32)
    stds = np.asarray([max(feature_stats[name]["std"], 1e-6) for name in channel_names], dtype=np.float32)
    return means, stds


if TORCH_AVAILABLE:  # pragma: no cover - only exercised with torch installed
    class SequenceTorchDataset(Dataset):
        def __init__(self, reader: SequenceDatasetReader, *, channel_names: list[str], feature_stats: dict[str, dict[str, float]]) -> None:
            self.reader = reader
            self.means, self.stds = _channel_stats(feature_stats, channel_names)

        def __len__(self) -> int:
            return len(self.reader)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = self.reader[index]
            inputs = sample["inputs"].astype(np.float32)
            inputs = (inputs - self.means[None, :, None, None]) / self.stds[None, :, None, None]
            return {
                "sample_id": sample["sample_id"],
                "region_id": sample["region_id"],
                "run_id": sample["run_id"],
                "debris_class": sample["debris_class"],
                "inputs": torch.from_numpy(inputs),
                "target_probability": torch.from_numpy(sample["target_probability"].astype(np.float32)),
                "target_expected_kg": torch.from_numpy(sample["target_expected_kg"].astype(np.float32)),
                "target_uncertainty": torch.from_numpy(sample["target_uncertainty"].astype(np.float32)),
            }


BaseDataModule = pl.LightningDataModule if LIGHTNING_AVAILABLE else object


class OceanRouteDataModule(BaseDataModule):
    def __init__(
        self,
        bundle: DatasetBundle,
        *,
        channel_names: list[str],
        region_ids: list[str] | None,
        horizons: list[int] | None,
        batch_size: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ValueError("torch is required to build training dataloaders.")
        super().__init__()
        self.bundle = bundle
        self.channel_names = channel_names
        self.region_ids = region_ids
        self.horizons = horizons
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.train_dataset: SequenceTorchDataset | None = None
        self.val_dataset: SequenceTorchDataset | None = None
        self.test_dataset: SequenceTorchDataset | None = None

    def _loader_kwargs(self, *, shuffle: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def setup(self, stage: str | None = None) -> None:  # pragma: no cover - thin wrapper
        self.train_dataset = SequenceTorchDataset(
            SequenceDatasetReader(self.bundle, split="train", region_ids=self.region_ids, horizons=self.horizons),
            channel_names=self.channel_names,
            feature_stats=self.bundle.feature_stats,
        )
        self.val_dataset = SequenceTorchDataset(
            SequenceDatasetReader(self.bundle, split="val", region_ids=self.region_ids, horizons=self.horizons),
            channel_names=self.channel_names,
            feature_stats=self.bundle.feature_stats,
        )
        self.test_dataset = SequenceTorchDataset(
            SequenceDatasetReader(self.bundle, split="test", region_ids=self.region_ids, horizons=self.horizons),
            channel_names=self.channel_names,
            feature_stats=self.bundle.feature_stats,
        )

    def train_dataloader(self):  # pragma: no cover - thin wrapper
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self):  # pragma: no cover - thin wrapper
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self):  # pragma: no cover - thin wrapper
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))

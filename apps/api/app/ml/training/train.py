from __future__ import annotations

import json
from pathlib import Path

from app.ml.training.config import TrainingConfig
from app.ml.training.trainer_service import train_sequence_model


def run_training(config: TrainingConfig, dataset_root: str | Path) -> dict[str, object]:
    return train_sequence_model(config, Path(dataset_root))


def print_training_result(config: TrainingConfig, dataset_root: str | Path) -> None:
    print(json.dumps(run_training(config, dataset_root), indent=2, default=str))

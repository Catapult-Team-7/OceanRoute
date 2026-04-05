from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


@dataclass(frozen=True)
class TrainingConfig:
    run_id: str
    model_id: str
    dataset_id: str
    architecture: str
    training_scope: str
    region_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    device: str = "auto"
    epochs: int = 2
    batch_size: int = 4
    num_workers: int = 4
    prefetch_factor: int = 2
    learning_rate: float = 1e-3
    promote_policy: str = "auto"
    training_command: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_root: Path = field(default_factory=lambda: settings.data_root)
    output_heads: tuple[str, ...] = ("hotspot_probability", "expected_kg", "uncertainty")
    framework: str = "pytorch_lightning"
    hotspot_loss: str = "focal"
    focal_gamma: float = 2.0

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["data_root"] = str(self.data_root)
        return payload

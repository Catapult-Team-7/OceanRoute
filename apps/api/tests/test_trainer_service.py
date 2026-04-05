from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "nn"):
    pytest.skip("real torch is required for trainer service tests", allow_module_level=True)

from app.ml.training.datamodule import OceanRouteDataModule
from app.ml.training.trainer_service import _evaluate_model


class TrackedInput:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.to_calls: list[tuple[torch.device, bool]] = []

    def to(self, device, non_blocking: bool = False):
        resolved = torch.device(device)
        self.to_calls.append((resolved, non_blocking))
        return self.tensor.to(resolved)


class DeviceAwareModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen_device: torch.device | None = None

    def forward(self, inputs):  # type: ignore[override]
        self.seen_device = inputs.device
        assert inputs.device == self.weight.device
        probability = inputs[:, -1:, 4:5]
        expected_kg = inputs[:, -1:, 4:5]
        uncertainty = inputs[:, -1:, 5:6]
        return probability, expected_kg, uncertainty


class FakeDataModule:
    def __init__(self, batch) -> None:
        self.batch = batch

    def test_dataloader(self):
        return [self.batch]


def test_evaluate_model_moves_inputs_to_module_device(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = torch.zeros((2, 12, 8, 4, 4), dtype=torch.float32)
    inputs[:, -1, 4:5] = 0.25
    inputs[:, -1, 5:6] = 0.1
    tracked_input = TrackedInput(inputs)
    batch = {
        "inputs": tracked_input,
        "target_probability": torch.zeros((2, 1, 1, 4, 4), dtype=torch.float32),
        "target_expected_kg": torch.zeros((2, 1, 1, 4, 4), dtype=torch.float32),
        "target_uncertainty": torch.zeros((2, 1, 1, 4, 4), dtype=torch.float32),
    }
    module = DeviceAwareModule()
    datamodule = FakeDataModule(batch)

    monkeypatch.setattr(
        "app.ml.training.trainer_service.compute_eval_metrics",
        lambda **kwargs: {"probability_shape": tuple(kwargs["probabilities"].shape)},
    )

    metrics = _evaluate_model(module, datamodule, [24])

    assert tracked_input.to_calls == [(module.weight.device, True)]
    assert module.seen_device == module.weight.device
    assert metrics["probability_shape"] == (2, 1, 4, 4)


def test_dataloader_kwargs_enable_workers_and_pin_memory() -> None:
    datamodule = OceanRouteDataModule(
        object(),
        channel_names=["current_u"],
        region_ids=None,
        horizons=None,
        batch_size=8,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )

    worker_kwargs = datamodule._loader_kwargs(shuffle=True)
    assert worker_kwargs["num_workers"] == 4
    assert worker_kwargs["pin_memory"] is True
    assert worker_kwargs["persistent_workers"] is True
    assert worker_kwargs["prefetch_factor"] == 2

    zero_worker_datamodule = OceanRouteDataModule(
        object(),
        channel_names=["current_u"],
        region_ids=None,
        horizons=None,
        batch_size=8,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=False,
    )
    zero_worker_kwargs = zero_worker_datamodule._loader_kwargs(shuffle=False)
    assert zero_worker_kwargs["num_workers"] == 0
    assert zero_worker_kwargs["pin_memory"] is False
    assert "persistent_workers" not in zero_worker_kwargs
    assert "prefetch_factor" not in zero_worker_kwargs

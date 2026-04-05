from __future__ import annotations

import pytest

from app.ml.sequence_model_architectures import TORCH_AVAILABLE, build_model


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed in the base test environment")
pytestmark = [pytestmark, pytest.mark.ml]


def test_sequence_models_emit_three_multi_horizon_heads() -> None:
    import torch

    inputs = torch.randn(2, 12, 12, 8, 8)
    for architecture in ("convlstm", "temporal_unet"):
        model = build_model(architecture, input_channels=12, num_horizons=3)
        probability, expected_kg, uncertainty = model(inputs)
        assert tuple(probability.shape) == (2, 3, 1, 8, 8)
        assert tuple(expected_kg.shape) == (2, 3, 1, 8, 8)
        assert tuple(uncertainty.shape) == (2, 3, 1, 8, 8)

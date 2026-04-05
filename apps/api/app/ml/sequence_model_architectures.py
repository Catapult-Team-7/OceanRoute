from __future__ import annotations

try:  # pragma: no cover - optional dependency scaffold
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency scaffold
    torch = None
    nn = None


TORCH_AVAILABLE = torch is not None and nn is not None


if TORCH_AVAILABLE:  # pragma: no cover - exercised only when torch is installed
    class ConvBlock2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        def forward(self, x):  # type: ignore[override]
            return self.net(x)


    class ConvBlock3d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        def forward(self, x):  # type: ignore[override]
            return self.net(x)


    class ConvLSTMCell(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.hidden_dim = hidden_dim
            self.gates = nn.Conv2d(
                input_dim + hidden_dim,
                4 * hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
            )

        def forward(self, x, state):  # type: ignore[override]
            h_cur, c_cur = state
            combined = torch.cat([x, h_cur], dim=1)
            i, f, o, g = torch.chunk(self.gates(combined), 4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)
            c_next = (f * c_cur) + (i * g)
            h_next = o * torch.tanh(c_next)
            return h_next, c_next


    class ConvLSTMForecaster(nn.Module):
        def __init__(self, input_channels: int, num_horizons: int, base_channels: int = 32) -> None:
            super().__init__()
            self.num_horizons = num_horizons
            self.base_channels = base_channels
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.temporal_layers = nn.ModuleList(
                [
                    ConvLSTMCell(base_channels, base_channels),
                    ConvLSTMCell(base_channels, base_channels),
                ]
            )
            self.decoder = ConvBlock2d(base_channels, base_channels)
            self.prob_head = nn.Sequential(
                nn.Conv2d(base_channels, num_horizons, kernel_size=1),
                nn.Sigmoid(),
            )
            self.kg_head = nn.Sequential(
                nn.Conv2d(base_channels, num_horizons, kernel_size=1),
                nn.ReLU(),
            )
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(base_channels, num_horizons, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):  # type: ignore[override]
            batch_size, _, _, height, width = x.shape
            hidden_states = [
                (
                    torch.zeros(batch_size, self.base_channels, height, width, device=x.device),
                    torch.zeros(batch_size, self.base_channels, height, width, device=x.device),
                )
                for _ in self.temporal_layers
            ]
            for time_index in range(x.shape[1]):
                encoded = self.encoder(x[:, time_index])
                next_input = encoded
                new_states = []
                for layer, state in zip(self.temporal_layers, hidden_states, strict=False):
                    h, c = layer(next_input, state)
                    new_states.append((h, c))
                    next_input = h
                hidden_states = new_states
            decoded = self.decoder(hidden_states[-1][0])
            probability = self.prob_head(decoded).unsqueeze(2)
            expected_kg = self.kg_head(decoded).unsqueeze(2)
            uncertainty = self.uncertainty_head(decoded).unsqueeze(2)
            return probability, expected_kg, uncertainty


    class TemporalUNetForecaster(nn.Module):
        def __init__(self, input_channels: int, num_horizons: int, base_channels: int = 32) -> None:
            super().__init__()
            self.encoder1 = ConvBlock3d(input_channels, base_channels)
            self.encoder2 = ConvBlock3d(base_channels, base_channels * 2)
            self.bottleneck = ConvBlock3d(base_channels * 2, base_channels * 2)
            self.decoder = ConvBlock2d(base_channels * 2, base_channels)
            self.prob_head = nn.Sequential(
                nn.Conv2d(base_channels, num_horizons, kernel_size=1),
                nn.Sigmoid(),
            )
            self.kg_head = nn.Sequential(
                nn.Conv2d(base_channels, num_horizons, kernel_size=1),
                nn.ReLU(),
            )
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(base_channels, num_horizons, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):  # type: ignore[override]
            x = x.permute(0, 2, 1, 3, 4)
            encoded1 = self.encoder1(x)
            encoded2 = self.encoder2(encoded1)
            bottleneck = self.bottleneck(encoded2)
            decoded = self.decoder(bottleneck.mean(dim=2))
            probability = self.prob_head(decoded).unsqueeze(2)
            expected_kg = self.kg_head(decoded).unsqueeze(2)
            uncertainty = self.uncertainty_head(decoded).unsqueeze(2)
            return probability, expected_kg, uncertainty


def architecture_ready(architecture: str) -> bool:
    return architecture == "linear_residual" or TORCH_AVAILABLE


def build_model(architecture: str, input_channels: int, num_horizons: int = 3):
    if not TORCH_AVAILABLE:
        raise ValueError("torch is not installed, so deep sequence architectures are unavailable.")
    if architecture == "temporal_unet":
        return TemporalUNetForecaster(input_channels=input_channels, num_horizons=num_horizons)
    if architecture == "convlstm":
        return ConvLSTMForecaster(input_channels=input_channels, num_horizons=num_horizons)
    raise ValueError(f"Unsupported architecture '{architecture}'.")

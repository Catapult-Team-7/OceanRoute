from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OceanPulseLSTM(nn.Module):
    def __init__(self, in_channels: int = 13, hidden_dim: int = 96, pooled_size: tuple[int, int] = (6, 12)):
        super().__init__()
        self.pooled_size = pooled_size
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(self.pooled_size),
        )
        self.lstm = nn.LSTM(
            input_size=96 * self.pooled_size[0] * self.pooled_size[1],
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
        )
        self.atm_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_projector = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x_spatial: torch.Tensor, atm_co2: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, height, width = x_spatial.shape
        encoded_steps = []
        for timestep in range(seq_len):
            encoded = self.spatial_encoder(x_spatial[:, timestep])
            encoded_steps.append(encoded.reshape(batch_size, -1))
        encoded_sequence = torch.stack(encoded_steps, dim=1)
        lstm_out, _ = self.lstm(encoded_sequence)
        latent = lstm_out[:, -1, :] + self.atm_projection(atm_co2)
        latent = latent.view(batch_size, -1, 1, 1).expand(-1, -1, self.pooled_size[0], self.pooled_size[1])
        decoded = self.latent_projector(latent)
        decoded = F.interpolate(decoded, size=(height, width), mode="bilinear", align_corners=False)
        return self.decoder(decoded)


class AnomalyDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, 24),
        )
        self.decoder = nn.Sequential(
            nn.Linear(24, 96),
            nn.GELU(),
            nn.Linear(96, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.forward(x)
        return torch.mean((x - reconstruction) ** 2, dim=-1)


def masked_huber_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.75) -> torch.Tensor:
    weighted = F.huber_loss(prediction * mask, target * mask, reduction="sum", delta=delta)
    denom = torch.clamp(mask.sum(), min=1.0)
    return weighted / denom


def physics_consistency_penalty(prediction: torch.Tensor, temperature_channel: torch.Tensor) -> torch.Tensor:
    predicted_mean = prediction.mean(dim=(-1, -2))
    temperature_mean = temperature_channel.mean(dim=(-1, -2))
    # Warmer conditions should generally weaken sink strength.
    return torch.mean(torch.relu((-predicted_mean) * (temperature_mean - 0.5)))

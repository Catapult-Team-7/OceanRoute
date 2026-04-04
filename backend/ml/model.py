import torch
import torch.nn as nn


class OceanPulseLSTM(nn.Module):
    def __init__(self, in_channels: int = 9, hidden_dim: int = 128):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.lstm = nn.LSTM(input_size=64 * 8 * 8, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.atm_projection = nn.Linear(1, hidden_dim)

    def forward(self, x_spatial: torch.Tensor, atm_co2: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _, _ = x_spatial.shape
        encoded = []
        for t in range(seq_len):
            features = self.spatial_encoder(x_spatial[:, t])
            encoded.append(features.reshape(batch_size, -1))
        sequence = torch.stack(encoded, dim=1)
        lstm_out, _ = self.lstm(sequence)
        state = lstm_out[:, -1, :] + self.atm_projection(atm_co2)
        state = state.view(batch_size, -1, 1, 1).expand(-1, -1, 8, 8)
        return self.decoder(state)


class AnomalyDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.forward(x)
        return torch.mean((x - reconstruction) ** 2, dim=-1)

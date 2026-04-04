import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.model import OceanPulseLSTM


class SyntheticOceanDataset(Dataset):
    def __init__(self, sample_count: int = 32):
        self.x = torch.randn(sample_count, 12, 9, 24, 48)
        self.y = torch.randn(sample_count, 1, 32, 32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OceanPulseLSTM().to(device)
    loader = DataLoader(SyntheticOceanDataset(), batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(3):
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            atm_co2 = torch.ones(features.shape[0], 1, device=device) * 420
            predictions = model(features, atm_co2)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch} loss={loss.item():.4f}")

    torch.save(model.state_dict(), "backend/ml/checkpoints/oceanpulse_demo.pt")


if __name__ == "__main__":
    train()

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import CNN_Autoencoder


DATA_PATH = Path("data/processed/fan_train_mfcc.npy")
MODEL_DIR = Path("models/mfcc_section_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = np.load(DATA_PATH, mmap_mode="r")
files = sorted(Path("data/raw/fan/train").glob("*.wav"))

assert len(features) == len(files), "Feature and filename counts differ!"
assert features.shape[1] == 40, "Expected 40 MFCC coefficients!"

print("Device:", device)
print("Total recordings:", len(files))
print("MFCC feature shape:", features.shape)


class FanDataset(Dataset):
    def __init__(self, indices, mean, std):
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        sample = features[self.indices[position]].astype(np.float32)
        sample = (sample - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(sample).unsqueeze(0)


for section in ["section_00", "section_01", "section_02"]:

    print(f"\n{'=' * 50}")
    print("Training MFCC model:", section)
    print("=" * 50)

    indices = [
        i for i, file in enumerate(files)
        if section in file.name
    ]

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=SEED
    )

    print("Training recordings:", len(train_indices))
    print("Validation recordings:", len(val_indices))

    # Calculate normalization statistics using training data only
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for idx in train_indices:
        sample = features[idx].astype(np.float64)
        total_sum += sample.sum()
        total_sq_sum += np.square(sample).sum()
        total_count += sample.size

    mean = total_sum / total_count
    std = np.sqrt(total_sq_sum / total_count - mean ** 2)

    print("Mean:", mean)
    print("Std:", std)

    train_loader = DataLoader(
        FanDataset(train_indices, mean, std),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        FanDataset(val_indices, mean, std),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = CNN_Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True
            )

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True
                )

                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item() * batch.size(0)

        val_loss /= len(val_loader.dataset)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"{section} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Train: {train_loss:.6f} "
            f"| Val: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "mean": mean,
                    "std": std
                },
                MODEL_DIR / f"{section}_autoencoder.pth"
            )

            print("Best checkpoint saved!")

    with open(MODEL_DIR / f"{section}_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print(
        f"{section} complete. "
        f"Best validation loss: {best_val_loss:.6f}"
    )

print("\nAll three MFCC section models trained!")
print("Saved in:", MODEL_DIR)

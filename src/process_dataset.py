from pathlib import Path
import numpy as np
from preprocessing import extract_mel_spectrogram

DATA_DIR = Path("data/raw/fan/train")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob("*.wav"))
features = []

for i, file in enumerate(files, 1):
    mel = extract_mel_spectrogram(file)
    features.append(mel.astype(np.float32))

    if i % 250 == 0:
        print(f"Processed {i}/{len(files)} recordings")

features = np.stack(features)

np.savez_compressed(
    OUTPUT_DIR / "fan_train_mel.npz",
    features=features
)

print("\nProcessing complete!")
print("Feature shape:", features.shape)
print("Saved:", OUTPUT_DIR / "fan_train_mel.npz")

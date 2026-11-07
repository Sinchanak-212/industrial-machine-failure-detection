import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/fan_train_mfcc.npy")
OUTPUT_PATH = Path("data/processed/fan_train_mfcc_stats.npy")

mfcc = np.load(DATA_PATH, mmap_mode="r")

print("MFCC shape:", mfcc.shape)

features = []

for i in range(mfcc.shape[0]):
    recording = mfcc[i]

    recording_features = np.concatenate([
        np.mean(recording, axis=1),
        np.std(recording, axis=1),
        np.percentile(recording, 10, axis=1),
        np.percentile(recording, 90, axis=1)
    ])

    features.append(recording_features)

features = np.asarray(features, dtype=np.float32)

np.save(OUTPUT_PATH, features)

print("Feature shape:", features.shape)
print("Saved to:", OUTPUT_PATH)
print("Feature extraction complete.")

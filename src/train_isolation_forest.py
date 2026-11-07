import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

FEATURE_PATH = Path("data/processed/fan_train_mfcc_stats.npy")
TRAIN_DIR = Path("data/raw/fan/train")
MODEL_DIR = Path("models/isolation_forest")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

features = np.load(FEATURE_PATH)

# Keep the same sorted file order used during MFCC extraction.
files = sorted(TRAIN_DIR.rglob("*.wav"))

if len(files) != len(features):
    raise ValueError(
        f"File/feature mismatch: {len(files)} files, {len(features)} feature rows"
    )

sections = ["section_00", "section_01", "section_02"]

for section in sections:
    indices = np.array([
        i for i, path in enumerate(files)
        if section in path.name
    ])

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=42
    )

    X_train = features[train_idx]
    X_val = features[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled)

    # Lower decision_function scores indicate more anomalous recordings.
    val_scores = model.decision_function(X_val_scaled)
    threshold = np.percentile(val_scores, 5)

    model_path = MODEL_DIR / f"{section}_model.joblib"
    scaler_path = MODEL_DIR / f"{section}_scaler.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    np.save(MODEL_DIR / f"{section}_threshold.npy", threshold)

    print(f"\n{section}")
    print("Training recordings:", len(train_idx))
    print("Validation recordings:", len(val_idx))
    print("Feature dimensions:", X_train.shape[1])
    print("Validation threshold:", round(float(threshold), 6))
    print("Model saved:", model_path)

print("\nAll Isolation Forest models trained successfully.")

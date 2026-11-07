from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import extract_mel_spectrogram
from model import CNN_Autoencoder


TRAIN_DIR = Path("data/raw/fan/train")
TEST_DIR = Path("data/raw/fan/source_test")
MODEL_DIR = Path("models/section_models")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = np.load("data/processed/fan_train_mel.npy", mmap_mode="r")
train_files = sorted(TRAIN_DIR.glob("*.wav"))

all_results = []
summary = []

print("Evaluation device:", device)

for section in ["section_00", "section_01", "section_02"]:

    print(f"\n{'=' * 45}")
    print("Evaluating:", section)
    print("=" * 45)

    # Reproduce the original training-validation split
    indices = [
        i for i, file in enumerate(train_files)
        if section in file.name
    ]

    _, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=SEED
    )

    # Load section-specific checkpoint
    checkpoint_path = MODEL_DIR / f"{section}_autoencoder.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    model = CNN_Autoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Calculate threshold using normal validation recordings only
    val_errors = []

    with torch.no_grad():
        for idx in val_indices:

            sample = features[idx].astype(np.float32)
            sample = (sample - mean) / (std + 1e-8)

            x = torch.from_numpy(sample).unsqueeze(0).unsqueeze(0)
            x = x.to(device=device, dtype=torch.float32)

            reconstructed = model(x)
            error = torch.mean((reconstructed - x) ** 2).item()

            val_errors.append(error)

    threshold = float(np.percentile(val_errors, 95))

    print("Validation threshold:", round(threshold, 6))

    # Evaluate corresponding test section
    test_files = sorted([
        file for file in TEST_DIR.glob("*.wav")
        if section in file.name
    ])

    section_results = []

    with torch.no_grad():
        for file in test_files:

            mel = extract_mel_spectrogram(file)
            mel = (mel.astype(np.float32) - mean) / (std + 1e-8)

            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
            x = x.to(device=device, dtype=torch.float32)

            reconstructed = model(x)
            error = torch.mean((reconstructed - x) ** 2).item()

            actual = 1 if "anomaly" in file.name else 0
            predicted = 1 if error > threshold else 0

            row = {
                "section": section,
                "filename": file.name,
                "actual": actual,
                "predicted": predicted,
                "reconstruction_error": error,
                "threshold": threshold
            }

            section_results.append(row)
            all_results.append(row)

    df = pd.DataFrame(section_results)

    if len(df) != 200 or df["actual"].nunique() != 2:
        raise ValueError(
            f"Unexpected test data for {section}: {len(df)} recordings"
        )

    y_true = df["actual"]
    y_pred = df["predicted"]
    scores = df["reconstruction_error"]

    auc = roc_auc_score(y_true, scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Test recordings:", len(df))
    print("ROC-AUC:", round(auc, 4))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    summary.append({
        "section": section,
        "threshold": threshold,
        "ROC-AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })


# Save detailed results and comparison
results_df = pd.DataFrame(all_results)
summary_df = pd.DataFrame(summary)

results_df.to_csv(
    OUTPUT_DIR / "section_test_predictions.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "section_model_comparison.csv",
    index=False
)

print("\n========== FINAL COMPARISON ==========")
print(summary_df.round(4).to_string(index=False))

print("\nSaved:")
print("reports/section_test_predictions.csv")
print("reports/section_model_comparison.csv")

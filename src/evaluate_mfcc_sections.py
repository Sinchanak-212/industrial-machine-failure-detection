from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
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

from model import CNN_Autoencoder


TRAIN_DIR = Path("data/raw/fan/train")
TEST_DIR = Path("data/raw/fan/source_test")
MODEL_DIR = Path("models/mfcc_section_models")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 40
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = np.load(
    "data/processed/fan_train_mfcc.npy",
    mmap_mode="r"
)
train_files = sorted(TRAIN_DIR.glob("*.wav"))

assert len(features) == len(train_files), "Feature and filename counts differ!"

print("Evaluation device:", device)
print("Training feature shape:", features.shape)


def extract_mfcc(audio_path):
    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    return mfcc


all_results = []
summary = []

for section in ["section_00", "section_01", "section_02"]:

    print(f"\n{'=' * 45}")
    print("Evaluating MFCC:", section)
    print("=" * 45)

    indices = [
        i for i, file in enumerate(train_files)
        if section in file.name
    ]

    _, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=SEED
    )

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

    # Threshold from normal validation recordings only
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

    test_files = sorted([
        file for file in TEST_DIR.glob("*.wav")
        if section in file.name
    ])

    section_results = []

    with torch.no_grad():
        for file in test_files:

            mfcc = extract_mfcc(file)
            mfcc = (mfcc.astype(np.float32) - mean) / (std + 1e-8)

            x = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0)
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


results_df = pd.DataFrame(all_results)
summary_df = pd.DataFrame(summary)

results_df.to_csv(
    OUTPUT_DIR / "mfcc_section_test_predictions.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "mfcc_section_model_comparison.csv",
    index=False
)

print("\n========== MFCC FINAL COMPARISON ==========")
print(summary_df.round(4).to_string(index=False))

print("\nSaved:")
print("reports/mfcc_section_test_predictions.csv")
print("reports/mfcc_section_model_comparison.csv")

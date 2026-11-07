import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

FEATURE_PATH = Path("data/processed/fan_train_mfcc_stats.npy")
TRAIN_DIR = Path("data/raw/fan/train")
TEST_DIR = Path("data/raw/fan/source_test")
MODEL_DIR = Path("models/isolation_forest")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

features = np.load(FEATURE_PATH)
train_files = sorted(TRAIN_DIR.rglob("*.wav"))

if len(train_files) != len(features):
    raise ValueError("Training file and feature counts do not match.")

results = []
all_predictions = []

for section in ["section_00", "section_01", "section_02"]:

    model = joblib.load(MODEL_DIR / f"{section}_model.joblib")
    scaler = joblib.load(MODEL_DIR / f"{section}_scaler.joblib")
    threshold = float(np.load(MODEL_DIR / f"{section}_threshold.npy"))

    # Source-test recordings are evaluated, never used for fitting.
    test_files = sorted([
        path for path in TEST_DIR.rglob("*.wav")
        if section in path.name
    ])

    X_test = []

    for path in test_files:
        # Features must use the same MFCC extraction as training.
        import librosa
        import soundfile as sf

        audio, sr = sf.read(path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sr,
                target_sr=16000
            )

        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=16000,
            n_mfcc=40,
            n_fft=1024,
            hop_length=256
        )

        row = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.percentile(mfcc, 10, axis=1),
            np.percentile(mfcc, 90, axis=1)
        ])

        X_test.append(row)

    X_test = np.asarray(X_test, dtype=np.float32)
    X_test_scaled = scaler.transform(X_test)

    # Lower decision scores mean more anomalous.
    scores = model.decision_function(X_test_scaled)

    y_true = np.array([
        1 if "anomaly" in path.name else 0
        for path in test_files
    ])

    y_pred = (scores < threshold).astype(int)

    auc = roc_auc_score(y_true, -scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    results.append({
        "Section": section,
        "ROC_AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Threshold": threshold
    })

    for path, actual, score, predicted in zip(
        test_files, y_true, scores, y_pred
    ):
        all_predictions.append({
            "Section": section,
            "Filename": path.name,
            "Actual": actual,
            "Anomaly_Score": -score,
            "Predicted": predicted
        })

    print(f"\n{section}")
    print("Test recordings:", len(test_files))
    print("ROC-AUC:", round(auc, 4))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("Confusion matrix:\n", cm)

results_df = pd.DataFrame(results)
predictions_df = pd.DataFrame(all_predictions)

results_df.to_csv(
    OUTPUT_DIR / "isolation_forest_comparison.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR / "isolation_forest_predictions.csv",
    index=False
)

print("\nOverall average ROC-AUC:",
      round(results_df["ROC_AUC"].mean(), 4))

print("\nSaved evaluation reports.")

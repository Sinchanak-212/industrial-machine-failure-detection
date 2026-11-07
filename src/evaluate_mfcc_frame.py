from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from model import CNN_Autoencoder


TRAIN_DIR = Path("data/raw/fan/train")
TEST_DIR = Path("data/raw/fan/source_test")
FEATURE_PATH = Path("data/processed/fan_train_mfcc.npy")
MODEL_DIR = Path("models/mfcc_section_models")
OUTPUT_DIR = Path("reports")

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = np.load(FEATURE_PATH, mmap_mode="r")
train_files = sorted(TRAIN_DIR.glob("*.wav"))

all_results = []
summary = []

print("Device:", device)


def get_frame_errors(model, sample, mean, std):
    sample = sample.astype(np.float32)
    sample = (sample - mean) / (std + 1e-8)

    x = torch.from_numpy(sample).unsqueeze(0).unsqueeze(0)
    x = x.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(x)

        # Mean squared reconstruction error for each time frame
        frame_errors = torch.mean(
            (reconstructed - x) ** 2,
            dim=(0, 1, 2)
        )

    return frame_errors.cpu().numpy()


for section in ["section_00", "section_01", "section_02"]:

    print(f"\n{'=' * 45}")
    print("Evaluating:", section)
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

    checkpoint = torch.load(
        MODEL_DIR / f"{section}_autoencoder.pth",
        map_location=device,
        weights_only=False
    )

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    model = CNN_Autoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Calculate both thresholds from normal validation recordings
    val_mean_scores = []
    val_p95_scores = []

    for idx in val_indices:
        frame_errors = get_frame_errors(
            model, features[idx], mean, std
        )

        val_mean_scores.append(np.mean(frame_errors))
        val_p95_scores.append(np.percentile(frame_errors, 95))

    mean_threshold = np.percentile(val_mean_scores, 95)
    p95_threshold = np.percentile(val_p95_scores, 95)

    print("Mean-error threshold:", round(mean_threshold, 6))
    print("P95-frame threshold:", round(p95_threshold, 6))

    test_files = sorted([
        file for file in TEST_DIR.glob("*.wav")
        if section in file.name
    ])

    for file in test_files:

        # Extracted MFCC features use the same parameters as training
        import librosa
        import soundfile as sf

        audio, sr = sf.read(file)

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

        frame_errors = get_frame_errors(
            model, mfcc, mean, std
        )

        mean_score = float(np.mean(frame_errors))
        p95_score = float(np.percentile(frame_errors, 95))

        actual = 1 if "anomaly" in file.name else 0

        all_results.append({
            "section": section,
            "filename": file.name,
            "actual": actual,
            "mean_score": mean_score,
            "p95_score": p95_score
        })

    section_df = pd.DataFrame([
        row for row in all_results if row["section"] == section
    ])

    y_true = section_df["actual"]

    for method, score_column, threshold in [
        ("Mean", "mean_score", mean_threshold),
        ("Frame_P95", "p95_score", p95_threshold)
    ]:

        scores = section_df[score_column]
        predictions = (scores > threshold).astype(int)

        auc = roc_auc_score(y_true, scores)
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        summary.append({
            "section": section,
            "method": method,
            "ROC-AUC": auc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

        print(f"\n{method} scoring:")
        print("ROC-AUC:", round(auc, 4))
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1-score:", round(f1, 4))


pd.DataFrame(all_results).to_csv(
    OUTPUT_DIR / "mfcc_frame_test_predictions.csv",
    index=False
)

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_DIR / "mfcc_frame_comparison.csv",
    index=False
)

print("\n========== FRAME SCORING COMPARISON ==========")
print(summary_df.round(4).to_string(index=False))

print("\nSaved:")
print("reports/mfcc_frame_test_predictions.csv")
print("reports/mfcc_frame_comparison.csv")

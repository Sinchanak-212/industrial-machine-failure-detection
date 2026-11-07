from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

from preprocessing import extract_mel_spectrogram
from model import CNN_Autoencoder


# Configuration
TEST_DIR = Path("data/raw/fan/source_test")
MODEL_PATH = Path("models/best_autoencoder.pth")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

mean = checkpoint["mean"]
std = checkpoint["std"]

model = CNN_Autoencoder().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded from:", MODEL_PATH)
print("Evaluation device:", device)


# Process test recordings
files = sorted(TEST_DIR.glob("*.wav"))

results = []

for i, file in enumerate(files, 1):

    mel = extract_mel_spectrogram(file)

    mel = (mel.astype(np.float32) - mean) / (std + 1e-8)

    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
    x = x.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(x)
        error = torch.mean((reconstructed - x) ** 2).item()

    label = 1 if "anomaly" in file.name else 0

    results.append({
        "filename": file.name,
        "actual": label,
        "reconstruction_error": error
    })

    if i % 100 == 0:
        print(f"Evaluated {i}/{len(files)} recordings")


# Results
df = pd.DataFrame(results)

if len(df) != 600 or df["actual"].nunique() != 2:
    raise ValueError("Unexpected test dataset structure or labels!")

# Threshold from normal validation reconstruction errors
# Calculate it separately in the next step if not already available.
# For this initial evaluation, use a provisional threshold.
normal_errors = df.loc[df["actual"] == 0, "reconstruction_error"]

threshold = normal_errors.quantile(0.95)

df["predicted"] = (
    df["reconstruction_error"] > threshold
).astype(int)

accuracy = accuracy_score(df["actual"], df["predicted"])
auc = roc_auc_score(df["actual"], df["reconstruction_error"])

print("\n========== EVALUATION RESULTS ==========")
print("Threshold:", threshold)
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC:", round(auc, 4))

print("\nClassification Report:")
print(classification_report(
    df["actual"],
    df["predicted"],
    target_names=["Normal", "Anomaly"]
))

print("Confusion Matrix:")
print(confusion_matrix(df["actual"], df["predicted"]))

df.to_csv(OUTPUT_DIR / "source_test_results.csv", index=False)

print("\nDetailed results saved to reports/source_test_results.csv")

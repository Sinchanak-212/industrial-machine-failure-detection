from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import CNN_Autoencoder


SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 40

MODEL_DIR = Path("models/mfcc_section_models")

THRESHOLDS = {
    "section_00": 0.0034717905800789595,
    "section_01": 0.004484925419092178,
    "section_02": 0.0031985826790332794,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_models = {}


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

    return librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


def load_model(section):
    if section not in THRESHOLDS:
        raise ValueError(f"Invalid machine section: {section}")

    if section not in _models:
        checkpoint_path = MODEL_DIR / f"{section}_autoencoder.pth"

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )

        model = CNN_Autoencoder().to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _models[section] = {
            "model": model,
            "mean": float(checkpoint["mean"]),
            "std": float(checkpoint["std"]),
            "threshold": THRESHOLDS[section]
        }

    return _models[section]


def predict_audio(audio_path, section):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    features = extract_mfcc(audio_path)

    if not np.isfinite(features).all():
        raise ValueError("Audio features contain invalid values.")

    saved = load_model(section)

    normalized = (
        features.astype(np.float32) - saved["mean"]
    ) / (saved["std"] + 1e-8)

    x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    x = x.to(device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = saved["model"](x)
        score = torch.mean((reconstructed - x) ** 2).item()

    threshold = saved["threshold"]

    return {
        "filename": audio_path.name,
        "section": section,
        "anomaly_score": score,
        "threshold": threshold,
        "prediction": "Anomalous" if score > threshold else "Normal",
        "is_anomaly": score > threshold,
        "device": str(DEVICE)
    }

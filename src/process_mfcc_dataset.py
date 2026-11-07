from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 40

TRAIN_DIR = Path("data/raw/fan/train")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(TRAIN_DIR.glob("*.wav"))

features = np.lib.format.open_memmap(
    OUTPUT_DIR / "fan_train_mfcc.npy",
    mode="w+",
    dtype=np.float32,
    shape=(len(files), N_MFCC, 626)
)

for i, file in enumerate(files):
    audio, sr = sf.read(file)

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

    if mfcc.shape != (N_MFCC, 626):
        raise ValueError(f"Unexpected shape {mfcc.shape}: {file.name}")

    features[i] = mfcc

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(files)}")

features.flush()

print("\nMFCC preprocessing completed!")
print("Dataset shape:", features.shape)
print("Saved: data/processed/fan_train_mfcc.npy")

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

def extract_mel_spectrogram(audio_path):
    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=1.0)

    return mel_db


if __name__ == "__main__":
    sample = next(Path("data/raw/fan/train").glob("*.wav"))
    features = extract_mel_spectrogram(sample)

    print("Sample:", sample.name)
    print("Mel spectrogram shape:", features.shape)
    print("Minimum value:", features.min())
    print("Maximum value:", features.max())

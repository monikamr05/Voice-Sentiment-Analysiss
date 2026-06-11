"""
Audio feature extraction using Librosa.
Extracts MFCC, Chroma, Mel Spectrogram, ZCR, and RMS Energy.
"""

import os
import numpy as np
import librosa

from config import (
    SAMPLE_RATE,
    DURATION_SEC,
    MAX_FRAMES,
    N_MFCC,
    N_CHROMA,
    N_MEL,
    RAVDESS_EMOTION_MAP,
)


def parse_ravdess_emotion(filename):
    """Extract emotion label from RAVDESS filename (e.g. 03-01-06-02-01-02-11.wav)."""
    base = os.path.basename(filename).replace(".wav", "").replace(".mp3", "")
    parts = base.split("-")
    if len(parts) >= 3:
        code = parts[2]
        return RAVDESS_EMOTION_MAP.get(code, None)
    return None


def load_audio_file(filepath):
    """Load and trim/pad audio to fixed duration."""
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION_SEC)
    target_len = int(SAMPLE_RATE * DURATION_SEC)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def extract_features(y, sr):
    """
    Extract combined feature matrix (frames x features).
    Features: MFCC, Chroma, Mel (pooled), ZCR, RMS.
    """
    hop_length = 512
    n_fft = 2048

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length, n_fft=n_fft)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length, n_fft=n_fft)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pool mel bands to N_MEL dimensions
    mel_pooled = np.zeros((N_MEL, mel_db.shape[1]))
    step = max(1, mel_db.shape[0] // N_MEL)
    for i in range(N_MEL):
        start = i * step
        end = min((i + 1) * step, mel_db.shape[0])
        mel_pooled[i] = np.mean(mel_db[start:end], axis=0)

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)

    # Align frame counts
    n_frames = min(
        mfcc.shape[1],
        chroma.shape[1],
        mel_pooled.shape[1],
        zcr.shape[1],
        rms.shape[1],
    )
    mfcc = mfcc[:, :n_frames]
    chroma = chroma[:, :n_frames]
    mel_pooled = mel_pooled[:, :n_frames]
    zcr = zcr[:, :n_frames]
    rms = rms[:, :n_frames]

    # Stack: (n_features, n_frames) -> transpose to (n_frames, n_features)
    features = np.vstack([mfcc, chroma, mel_pooled, zcr, rms]).T
    return features


def pad_or_truncate(features, max_frames=MAX_FRAMES):
    """Pad or truncate feature matrix to fixed frame count."""
    n_frames, n_feat = features.shape
    if n_frames >= max_frames:
        return features[:max_frames, :]
    pad = np.zeros((max_frames - n_frames, n_feat))
    return np.vstack([features, pad])


def extract_from_array(y, sr):
    """Return fixed-size feature matrix from loaded audio."""
    features = extract_features(y, sr)
    return pad_or_truncate(features)


def extract_from_file(filepath):
    """Load audio file and return fixed-size feature matrix."""
    y, sr = load_audio_file(filepath)
    return extract_from_array(y, sr)


def get_feature_shape():
    """Return shape of a single feature sample (frames, features)."""
    dummy = np.zeros(int(SAMPLE_RATE * DURATION_SEC))
    feat = extract_features(dummy, SAMPLE_RATE)
    feat = pad_or_truncate(feat)
    return feat.shape


def normalize_features(X, mean=None, std=None):
    """Normalize feature array. Returns normalized X and (mean, std)."""
    if mean is None or std is None:
        mean = np.mean(X, axis=(0, 1), keepdims=True)
        std = np.std(X, axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
    X_norm = (X - mean) / std
    return X_norm, mean, std


def scan_dataset(dataset_dir):
    """
    Scan dataset folder recursively for audio files.
    Returns list of (filepath, emotion_label).
    """
    samples = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                path = os.path.join(root, f)
                emotion = parse_ravdess_emotion(f)
                if emotion:
                    samples.append((path, emotion))
    return samples

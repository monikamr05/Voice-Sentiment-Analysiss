"""
Prediction utilities for uploaded audio files.
"""

import os
import pickle
import numpy as np
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    MODEL_PATH,
    NORMALIZER_PATH,
    LABEL_ENCODER_PATH,
    SENTIMENT_MAP,
    SENTIMENT_COLORS,
    SAMPLE_RATE,
    DURATION_SEC,
    CONFIDENCE_THRESHOLD,
)
from audio_features import extract_from_array, extract_from_file, normalize_features, load_audio_file


_model = None
_normalizer = None
_label_encoder = None
_use_numpy_backend = None


def _get_keras():
    import importlib

    tf = importlib.import_module("tensorflow")
    return tf.keras


def _use_numpy():
    """
    Use NumPy inference from model.h5 (no TensorFlow).
    Set USE_TENSORFLOW_INFERENCE=1 to force Keras/TensorFlow instead.
    """
    global _use_numpy_backend
    if _use_numpy_backend is None:
        flag = os.environ.get("USE_TENSORFLOW_INFERENCE", "").lower()
        _use_numpy_backend = flag not in ("1", "true", "yes")
    return _use_numpy_backend


def load_model():
    """Load trained model and preprocessing artifacts (cached)."""
    global _model, _normalizer, _label_encoder
    if _normalizer is None:
        if not os.path.exists(NORMALIZER_PATH):
            raise FileNotFoundError("Normalizer not found. Train the model first.")
        data = np.load(NORMALIZER_PATH)
        _normalizer = (data["mean"], data["std"])
    if _label_encoder is None:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)

    if _use_numpy():
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python train_model.py"
            )
        return None, _normalizer, _label_encoder

    keras = _get_keras()
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python train_model.py"
            )
        _model = keras.models.load_model(MODEL_PATH)
    return _model, _normalizer, _label_encoder


def _run_inference(model, X_norm):
    if _use_numpy():
        from numpy_inference import predict_proba

        return predict_proba(X_norm)
    return model.predict(X_norm, verbose=0)


def _build_result(probs, le, duration_sec):
    idx = int(np.argmax(probs))
    emotion = le.inverse_transform([idx])[0]
    confidence = float(probs[idx] * 100)
    sentiment = SENTIMENT_MAP.get(emotion, "Neutral")
    prob_dict = {
        le.inverse_transform([i])[0]: round(float(probs[i]) * 100, 2)
        for i in range(len(probs))
    }
    conf_round = round(confidence, 2)
    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "confidence": conf_round,
        "probabilities": prob_dict,
        "sentiment_color": SENTIMENT_COLORS.get(sentiment, "#6c757d"),
        "low_confidence": conf_round < CONFIDENCE_THRESHOLD,
        "duration_sec": duration_sec,
    }


def predict_emotion_from_audio(y, sr):
    """Predict emotion from already-loaded audio (avoids re-reading the file)."""
    model, (mean, std), le = load_model()
    features = extract_from_array(y, sr)
    X = np.expand_dims(features, axis=0).astype(np.float32)
    X_norm, _, _ = normalize_features(X, mean=mean, std=std)
    probs = _run_inference(model, X_norm)[0]
    duration_sec = round(len(y) / sr, 2)
    return _build_result(probs, le, duration_sec)


def predict_emotion(filepath):
    """
    Predict emotion and sentiment from audio file.
    Returns dict with emotion, sentiment, confidence, probabilities.
    """
    y, sr = load_audio_file(filepath)
    return predict_emotion_from_audio(y, sr)


def analyze_audio(filepath):
    """Load audio once, then predict. Returns (result, y, sr)."""
    y, sr = load_audio_file(filepath)
    result = predict_emotion_from_audio(y, sr)
    return result, y, sr


def warmup_model():
    """Pre-load model weights and JIT-compile audio processing at startup."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(NORMALIZER_PATH):
        return False

    load_model()
    from audio_features import get_feature_shape

    frames, n_feat = get_feature_shape()
    dummy = np.zeros((1, frames, n_feat), dtype=np.float32)
    _run_inference(None, dummy)

    silent = np.zeros(int(SAMPLE_RATE * DURATION_SEC), dtype=np.float32)
    extract_from_array(silent, SAMPLE_RATE)
    return True


def get_audio_duration(filepath):
    """Return audio duration in seconds."""
    try:
        return round(float(librosa.get_duration(path=filepath)), 2)
    except Exception:
        return None


def generate_waveform_plot(filepath, output_path, y=None, sr=None):
    """Save waveform image for display."""
    if y is None or sr is None:
        y, sr = load_audio_file(filepath)
    fig, ax = plt.subplots(figsize=(7, 2))
    times = np.arange(len(y)) / sr
    ax.plot(times, y, color="#0d6efd", linewidth=0.6)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, times[-1] if len(times) else 0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=72, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_spectrogram_plot(filepath, output_path, y=None, sr=None):
    """Save mel spectrogram image for display."""
    if y is None or sr is None:
        y, sr = load_audio_file(filepath)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(7, 2.5))
    img = ax.imshow(S_db, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Mel band")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=72, bbox_inches="tight", facecolor="white")
    plt.close()

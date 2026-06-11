"""
Application configuration and emotion/sentiment mappings.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.h5")
NORMALIZER_PATH = os.path.join(SAVED_MODEL_DIR, "normalizer.npz")
LABEL_ENCODER_PATH = os.path.join(SAVED_MODEL_DIR, "label_encoder.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
UPLOADS_DIR = os.path.join(BASE_DIR, "static", "uploads")
DATABASE_PATH = os.path.join(BASE_DIR, "predictions.db")

# Audio processing
SAMPLE_RATE = 22050
DURATION_SEC = 3.0
MAX_FRAMES = 128
N_MFCC = 40
N_CHROMA = 12
N_MEL = 40

# RAVDESS emotion codes (3rd field in filename)
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

# Emotion -> sentiment mapping
SENTIMENT_MAP = {
    "happy": "Positive",
    "calm": "Positive",
    "neutral": "Neutral",
    "sad": "Negative",
    "angry": "Negative",
    "fearful": "Negative",
    "disgust": "Negative",
    "surprised": "Positive",
}

SENTIMENT_COLORS = {
    "Positive": "#28a745",
    "Neutral": "#ffc107",
    "Negative": "#dc3545",
}

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}
MAX_UPLOAD_SIZE_MB = 16

# Below this confidence (%), show "manual review recommended"
CONFIDENCE_THRESHOLD = 60.0
MAX_BATCH_FILES = 10

# Default admin (change in production)
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"

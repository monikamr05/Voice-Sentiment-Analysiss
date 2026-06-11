# AI Voice Call Sentiment Analysis

A complete web application for analyzing emotions and sentiment in voice/audio recordings using deep learning. Built with **Python**, **Flask**, **TensorFlow**, **Librosa**, and **Bootstrap 5**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)

## Features

- **RAVDESS dataset** — automatic loading, emotion parsing from filenames, train/test split
- **Audio features** — MFCC, Chroma, Mel Spectrogram, ZCR, RMS Energy (Librosa)
- **CNN model** — TensorFlow/Keras speech emotion recognition
- **Sentiment mapping** — 8 emotions → Positive / Neutral / Negative
- **Web UI** — drag-and-drop upload, audio preview, waveform & spectrogram plots
- **SQLite history** — prediction storage with admin dashboard
- **PDF reports** — downloadable prediction reports
- **REST API** — `POST /api/predict` for real-time inference

## Project Structure

```
voice_sentiment_analysis/
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── predict.py             # Prediction utilities
├── audio_features.py      # Feature extraction
├── config.py              # Configuration
├── database.py            # SQLite integration
├── requirements.txt
├── saved_model/
│   └── model.h5           # (created after training)
├── dataset/               # RAVDESS .wav files
├── static/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── plots/
└── templates/
    ├── index.html
    ├── upload.html
    ├── result.html
    ├── dashboard.html
    ├── about.html
    ├── training.html
    └── results.html
```

## Quick Start

### 1. Install dependencies

```bash
cd voice_sentiment_analysis
pip install -r requirements.txt
```

### 2. Download RAVDESS dataset

1. Go to https://zenodo.org/record/1188976
2. Download **Audio_Speech_Actors_01-24** (~215 MB)
3. Extract `.wav` files into the `dataset/` folder

See `dataset/README.md` for details.

### 3. Train the model

```bash
python train_model.py
```

This will:
- Scan all audio in `dataset/`
- Extract features and normalize
- Train a 1D CNN (80/20 train/test split)
- Save `saved_model/model.h5`, normalizer, and label encoder
- Generate accuracy/loss plots and confusion matrix in `static/plots/`

### 4. Run the web app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Usage

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Overview and sentiment mapping |
| Upload | `/upload` | Drag-and-drop audio analysis |
| Training | `/training` | Model metrics and plots |
| Results | `/results` | Prediction history |
| Dashboard | `/dashboard` | Admin stats and charts |
| About | `/about` | Project documentation |

### API — Real-time prediction

```bash
curl -X POST -F "audio=@your_file.wav" http://localhost:5000/api/predict
```

Response:

```json
{
  "emotion": "happy",
  "sentiment": "Positive",
  "confidence": 87.5,
  "probabilities": { "happy": 87.5, "sad": 3.2, ... },
  "sentiment_color": "#28a745",
  "prediction_id": 1
}
```

## Sentiment Mapping

| Emotion | Sentiment |
|---------|-----------|
| Happy | Positive |
| Calm | Positive |
| Surprised | Positive |
| Neutral | Neutral |
| Sad | Negative |
| Angry | Negative |
| Fearful | Negative |
| Disgust | Negative |

## Model Architecture

- **Input**: 128 time frames × 94 features (MFCC + Chroma + Mel + ZCR + RMS)
- **Architecture**: Conv1D(64) → Conv1D(128) → Conv1D(256) → GlobalAveragePooling → Dense(128) → Softmax(8)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Sparse categorical crossentropy

## Requirements

- **Python 3.9–3.12** (required for TensorFlow; Python 3.13+ is not supported yet)
- ~4 GB RAM recommended for training
- FFmpeg not required (Librosa + soundfile handle WAV)

> **Note:** If you only have Python 3.14 installed, install Python 3.12 from [python.org](https://www.python.org/downloads/) and use `py -3.12` to create a virtual environment.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No files in dataset | Download RAVDESS and extract to `dataset/` |
| Model not found | Run `python train_model.py` first |
| Librosa load error | Install `soundfile`: `pip install soundfile` |
| Slow training | Use GPU TensorFlow or reduce dataset for testing |

## License

Educational / academic use. RAVDESS dataset has its own license — see Zenodo record.

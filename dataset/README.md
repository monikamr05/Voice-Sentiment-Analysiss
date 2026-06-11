# RAVDESS Dataset

Place RAVDESS audio files in this folder (subfolders are supported).

## Download

1. Visit: https://zenodo.org/record/1188976
2. Download **Audio_Speech_Actors_01-24** (or full dataset)
3. Extract all `.wav` files into this `dataset/` directory

## Filename Format

RAVDESS files follow this pattern:

```
MM-VC-EM-IN-ST-RE-AC.wav
```

- **EM** (3rd field) = Emotion code:
  - 01 = neutral, 02 = calm, 03 = happy, 04 = sad
  - 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised

Example: `03-01-05-02-01-02-01.wav` → angry (speech, audio-only)

## Training

After adding files, run from project root:

```bash
python train_model.py
```

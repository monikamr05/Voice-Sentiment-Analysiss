"""
Optional: Generate minimal demo WAV files with RAVDESS naming for smoke testing.
Not a substitute for real RAVDESS data — use for pipeline verification only.

Usage: python generate_demo_data.py
"""

import os
import numpy as np
import soundfile as sf

from config import DATASET_DIR, SAMPLE_RATE

# One file per emotion (3rd field = emotion code)
DEMO_FILES = [
    ("03-01-01-01-01-01-01.wav", 220),   # neutral
    ("03-01-02-01-01-01-01.wav", 261),   # calm
    ("03-01-03-01-01-01-01.wav", 330),   # happy
    ("03-01-04-01-01-01-01.wav", 196),   # sad
    ("03-01-05-01-01-01-01.wav", 150),   # angry
    ("03-01-06-01-01-01-01.wav", 300),   # fearful
    ("03-01-07-01-01-01-01.wav", 180),   # disgust
    ("03-01-08-01-01-01-01.wav", 400),   # surprised
]


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    for name, freq in DEMO_FILES:
        # Simple tone + noise — demo only
        y = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(len(t))
        path = os.path.join(DATASET_DIR, name)
        sf.write(path, y.astype(np.float32), SAMPLE_RATE)
        print(f"Created {path}")

    print(f"\nGenerated {len(DEMO_FILES)} demo files in {DATASET_DIR}")
    print("For real accuracy, use the full RAVDESS dataset.")


if __name__ == "__main__":
    main()

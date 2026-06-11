"""
Train CNN emotion recognition model on RAVDESS dataset.
Saves model.h5, normalizer, plots, and classification metrics.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATASET_DIR,
    SAVED_MODEL_DIR,
    MODEL_PATH,
    NORMALIZER_PATH,
    LABEL_ENCODER_PATH,
    PLOTS_DIR,
    EMOTIONS,
)
from audio_features import scan_dataset, extract_from_file, normalize_features


def _load_tensorflow():
    """Import TensorFlow lazily to avoid editor 'module not found' noise."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        from tensorflow.keras import layers

        return keras, layers
    except ImportError as e:
        raise SystemExit(
            "TensorFlow is required for training.\n"
            "Use Python 3.9–3.12 (TensorFlow is not available on Python 3.13+).\n"
            f"Original error: {e}"
        ) from e


def build_cnn_model(input_shape, num_classes, keras, layers):
    """1D CNN for speech emotion recognition."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Conv1D(128, kernel_size=5, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="emotion_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_dataset():
    """Load all RAVDESS samples and extract features."""
    samples = scan_dataset(DATASET_DIR)
    if not samples:
        raise FileNotFoundError(
            f"No RAVDESS audio files found in '{DATASET_DIR}'.\n"
            "Download RAVDESS from: https://zenodo.org/record/1188976\n"
            "Extract .wav files into the dataset/ folder (subfolders OK)."
        )

    print(f"Found {len(samples)} audio files. Extracting features...")
    X_list, y_list = [], []
    for i, (path, emotion) in enumerate(samples):
        try:
            feat = extract_from_file(path)
            X_list.append(feat)
            y_list.append(emotion)
        except Exception as e:
            print(f"  Skip {path}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    print(f"Feature matrix shape: {X.shape}")
    return X, y


def plot_training_history(history, save_dir):
    """Save accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_dir):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(
        y_true, y_pred, labels=np.arange(len(labels))
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    keras, layers = _load_tensorflow()

    X, y = load_dataset()

    # Encode labels
    le = LabelEncoder()
    le.fit(EMOTIONS)
    y_encoded = le.transform(y)

    # Normalize features
    X_norm, mean, std = normalize_features(X)
    np.savez(NORMALIZER_PATH, mean=mean, std=std)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    # Train/test split (stratify only if each emotion has >= 2 samples)
    split_kwargs = {"test_size": 0.2, "random_state": 42}
    _, class_counts = np.unique(y_encoded, return_counts=True)
    if len(X_norm) >= 10 and class_counts.min() >= 2:
        split_kwargs["stratify"] = y_encoded
    else:
        print(
            "Note: Random train/test split (some emotions have only 1 file — "
            "add more RAVDESS files per emotion for better accuracy)."
        )
        if len(X_norm) < 10:
            split_kwargs["test_size"] = max(1, int(len(X_norm) * 0.2)) / len(X_norm)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_encoded, **split_kwargs)

    input_shape = X_train.shape[1:]
    num_classes = len(le.classes_)
    model = build_cnn_model(input_shape, num_classes, keras, layers)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    batch_size = min(32, max(4, len(X_train) // 4))
    epochs = 50 if len(X_train) >= 100 else 30

    print("\nTraining model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test) if len(X_test) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    label_ids = np.arange(len(le.classes_))
    report = classification_report(
        y_test,
        y_pred,
        labels=label_ids,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report:\n", report)

    report_path = os.path.join(SAVED_MODEL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Saved: {report_path}")

    # Save model and plots
    model.save(MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

    plot_training_history(history, PLOTS_DIR)
    plot_confusion_matrix(y_test, y_pred, le.classes_, PLOTS_DIR)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

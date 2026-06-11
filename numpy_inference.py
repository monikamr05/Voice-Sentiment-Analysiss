"""
Run CNN inference from Keras model.h5 without TensorFlow.
Used when Windows Application Control blocks TensorFlow DLLs.
"""

import os
import h5py
import numpy as np

from config import MODEL_PATH

_WEIGHTS = None


def _read_dataset(group, path):
    return np.array(group[path])


def _load_weights(model_path=MODEL_PATH):
    """Load layer weights from Keras HDF5 model file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    weights = {}
    with h5py.File(model_path, "r") as f:
        root = f["model_weights"]

        def layer_group(layer_name):
            return root[layer_name]["emotion_cnn"][layer_name]

        def bn(layer_name):
            g = layer_group(layer_name)
            return {
                "gamma": _read_dataset(g, "gamma"),
                "beta": _read_dataset(g, "beta"),
                "mean": _read_dataset(g, "moving_mean"),
                "var": _read_dataset(g, "moving_variance"),
            }

        def conv(layer_name):
            g = layer_group(layer_name)
            return {
                "kernel": _read_dataset(g, "kernel"),
                "bias": _read_dataset(g, "bias"),
            }

        weights["conv1"] = conv("conv1d")
        weights["bn1"] = bn("batch_normalization")
        weights["conv2"] = conv("conv1d_1")
        weights["bn2"] = bn("batch_normalization_1")
        weights["conv3"] = conv("conv1d_2")
        weights["bn3"] = bn("batch_normalization_2")
        g_dense = layer_group("dense")
        weights["dense"] = {
            "kernel": _read_dataset(g_dense, "kernel"),
            "bias": _read_dataset(g_dense, "bias"),
        }
        g_out = layer_group("dense_1")
        weights["dense_out"] = {
            "kernel": _read_dataset(g_out, "kernel"),
            "bias": _read_dataset(g_out, "bias"),
        }
    return weights


def _conv1d_same(x, kernel, bias):
    """x: (batch, length, in_ch), kernel: (k, in_ch, out_ch)."""
    batch, length, _ = x.shape
    k, _, out_ch = kernel.shape
    pad = k // 2
    x_pad = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="constant")
    out = np.zeros((batch, length, out_ch), dtype=np.float32)
    for i in range(length):
        seg = x_pad[:, i : i + k, :]
        out[:, i, :] = np.einsum("bki,kio->bo", seg, kernel, optimize=True) + bias
    return out


def _batch_norm(x, bn, eps=1e-3):
    mean = bn["mean"]
    var = bn["var"]
    gamma = bn["gamma"]
    beta = bn["beta"]
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _max_pool1d(x, pool_size=2):
    batch, length, ch = x.shape
    new_len = length // pool_size
    trimmed = x[:, : new_len * pool_size, :]
    return np.max(
        trimmed.reshape(batch, new_len, pool_size, ch), axis=2
    )


def _relu(x):
    return np.maximum(x, 0)


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def predict_proba(X_norm):
    """
    Run forward pass. X_norm shape: (batch, frames, features) e.g. (1, 128, 94).
    Returns probabilities (batch, num_classes).
    """
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = _load_weights()

    w = _WEIGHTS
    x = np.asarray(X_norm, dtype=np.float32)

    x = _relu(_batch_norm(_conv1d_same(x, w["conv1"]["kernel"], w["conv1"]["bias"]), w["bn1"]))
    x = _max_pool1d(x, 2)

    x = _relu(_batch_norm(_conv1d_same(x, w["conv2"]["kernel"], w["conv2"]["bias"]), w["bn2"]))
    x = _max_pool1d(x, 2)

    x = _relu(_batch_norm(_conv1d_same(x, w["conv3"]["kernel"], w["conv3"]["bias"]), w["bn3"]))
    x = np.mean(x, axis=1)

    x = _relu(x @ w["dense"]["kernel"] + w["dense"]["bias"])
    logits = x @ w["dense_out"]["kernel"] + w["dense_out"]["bias"]
    return _softmax(logits)

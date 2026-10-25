"""
Evaluation metrics for regression and classification experiments.

All regression metrics are computed from the same target/prediction vectors.
The returned population target variance makes the identity
R2 = 1 - MSE / Var(y_true) auditable across methods evaluated on the same test
set. This is useful for detecting tables assembled from inconsistent splits or
scales.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


EPSILON = 1e-12


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute consistent regression metrics from one shared evaluation vector."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("y_true and y_pred must contain only finite values.")

    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    target_variance = float(np.mean((y_true - np.mean(y_true)) ** 2))

    if target_variance <= EPSILON:
        r2 = float("nan")
    else:
        # Equivalent to 1 - SSE/SST, expressed via the population variance.
        r2 = float(1.0 - mse / target_variance)

    return {
        "n": int(y_true.shape[0]),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "target_variance": target_variance,
    }


def softmax(logits) -> np.ndarray:
    """Convert a matrix of logits into row-wise probabilities."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError("logits must be a two-dimensional array.")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have at least one row and one class.")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must contain only finite values.")
    centered = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(centered, -700.0, 700.0))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def one_hot(labels, n_classes: Optional[int] = None) -> np.ndarray:
    """Create a one-hot label matrix from integer class labels."""
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.size == 0:
        raise ValueError("labels must not be empty.")
    if n_classes is None:
        n_classes = int(np.max(labels)) + 1
    if n_classes < 1 or np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels must be between 0 and n_classes - 1.")

    out = np.zeros((labels.shape[0], n_classes), dtype=float)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def cross_entropy_from_probabilities(y_true, probabilities) -> float:
    """Compute multiclass cross entropy from probability estimates."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array.")

    labels = np.asarray(y_true, dtype=int).reshape(-1)
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have the same number of rows.")
    if labels.size == 0:
        raise ValueError("y_true must not be empty.")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("Labels must index the columns of probabilities.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(probabilities < 0):
        raise ValueError("probabilities must be nonnegative.")

    clipped = np.clip(probabilities, EPSILON, 1.0)
    clipped = clipped / np.sum(clipped, axis=1, keepdims=True)
    return float(-np.mean(np.log(clipped[np.arange(labels.shape[0]), labels])))


def classification_metrics(y_true, probabilities=None, predicted_labels=None) -> Dict[str, float]:
    """Compute accuracy and, when available, cross entropy."""
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    if labels.size == 0:
        raise ValueError("y_true must not be empty.")

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2:
            raise ValueError("probabilities must be a two-dimensional array.")
        predicted = np.argmax(probabilities, axis=1)
        ce = cross_entropy_from_probabilities(labels, probabilities)
    elif predicted_labels is not None:
        predicted = np.asarray(predicted_labels, dtype=int).reshape(-1)
        ce = float("nan")
    else:
        raise ValueError("Provide probabilities or predicted_labels.")

    if predicted.shape[0] != labels.shape[0]:
        raise ValueError("Predicted labels must have the same length as y_true.")

    accuracy = float(np.mean(predicted == labels))
    return {
        "n": int(labels.shape[0]),
        "accuracy": accuracy,
        "cross_entropy": ce,
    }


def summarize_elapsed_seconds(start_time: float, end_time: float) -> float:
    """Return elapsed runtime in seconds with stable numeric formatting."""
    return float(max(0.0, end_time - start_time))

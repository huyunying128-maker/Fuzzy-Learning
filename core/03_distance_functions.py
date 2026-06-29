"""
Distance-table utilities for crisp, fuzzy, and modified k-means partitions.

The partition layer uses a full table of distances between observations and
centroids. The same table supports nearest-centroid labels, fuzzy membership
construction, and distance-based truncation rules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


EPSILON = 1e-12


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def minkowski_distance_matrix(X, centroids, p: float = 2.0, chunk_size: Optional[int] = None) -> np.ndarray:
    """Compute the pairwise Minkowski distance table between X and centroids."""
    X = _as_2d_float(X, "X")
    centroids = _as_2d_float(centroids, "centroids")
    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of columns.")
    if p < 1.0:
        raise ValueError("Minkowski distance order p must be at least 1.")

    if chunk_size is None or chunk_size >= X.shape[0]:
        diff = np.abs(X[:, None, :] - centroids[None, :, :])
        return np.sum(diff ** p, axis=2) ** (1.0 / p)

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive when it is provided.")

    distances = np.empty((X.shape[0], centroids.shape[0]), dtype=float)
    for start in range(0, X.shape[0], chunk_size):
        stop = min(start + chunk_size, X.shape[0])
        diff = np.abs(X[start:stop, None, :] - centroids[None, :, :])
        distances[start:stop] = np.sum(diff ** p, axis=2) ** (1.0 / p)
    return distances


def nearest_centroid_labels(distance_table) -> np.ndarray:
    """Return the nearest-centroid label for each row of a distance table."""
    distance_table = _as_2d_float(distance_table, "distance_table")
    return np.argmin(distance_table, axis=1).astype(int)


def average_distance_table_change(current_distance_table, previous_distance_table) -> float:
    """Compute the mean absolute movement between two distance tables."""
    current = _as_2d_float(current_distance_table, "current_distance_table")
    previous = _as_2d_float(previous_distance_table, "previous_distance_table")
    if current.shape != previous.shape:
        raise ValueError("Distance tables must have the same shape.")
    return float(np.mean(np.abs(current - previous)))


def harmonic_distance_change(current_distance_table, previous_distance_table, epsilon: float = EPSILON) -> float:
    """Compute a harmonic summary of absolute distance-table movement."""
    current = _as_2d_float(current_distance_table, "current_distance_table")
    previous = _as_2d_float(previous_distance_table, "previous_distance_table")
    if current.shape != previous.shape:
        raise ValueError("Distance tables must have the same shape.")

    change = np.abs(current - previous)
    return float(change.size / np.sum(1.0 / (change + epsilon)))


def distance_objective(distance_table, labels=None) -> float:
    """Compute the nearest-centroid distance objective for a fitted partition."""
    distance_table = _as_2d_float(distance_table, "distance_table")
    if labels is None:
        selected = np.min(distance_table, axis=1)
    else:
        labels = np.asarray(labels, dtype=int).reshape(-1)
        if labels.shape[0] != distance_table.shape[0]:
            raise ValueError("labels must have one entry for each distance-table row.")
        selected = distance_table[np.arange(distance_table.shape[0]), labels]
    return float(np.mean(selected ** 2))


def replace_zero_distances(distance_table, epsilon: float = EPSILON) -> np.ndarray:
    """Return a safe copy of the distance table for reciprocal calculations."""
    distance_table = _as_2d_float(distance_table, "distance_table")
    safe = np.array(distance_table, copy=True, dtype=float)
    safe[safe < epsilon] = epsilon
    return safe

"""
Membership utilities for hard and soft partition representations.

Crisp memberships assign each observation to one centroid. Fuzzy memberships
keep a graded relationship to every centroid and are used both for local input
weighting and for output aggregation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


EPSILON = 1e-12


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def hard_membership_from_labels(labels, n_clusters: int) -> np.ndarray:
    """Convert nearest-centroid labels into a hard membership table."""
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive.")
    if np.any(labels < 0) or np.any(labels >= n_clusters):
        raise ValueError("labels must be between 0 and n_clusters - 1.")

    membership = np.zeros((labels.shape[0], n_clusters), dtype=float)
    membership[np.arange(labels.shape[0]), labels] = 1.0
    return membership


def hard_membership_from_distances(distance_table) -> Tuple[np.ndarray, np.ndarray]:
    """Build hard memberships by assigning each row to its nearest centroid."""
    distance_table = _as_2d_float(distance_table, "distance_table")
    labels = np.argmin(distance_table, axis=1).astype(int)
    membership = hard_membership_from_labels(labels, distance_table.shape[1])
    return membership, labels


def fuzzy_membership_from_distances(distance_table, f: float = 2.0, epsilon: float = EPSILON) -> np.ndarray:
    """Convert a distance table into normalized fuzzy memberships.

    When f equals 1.00, the usual reciprocal exponent is singular; this value is
    treated as the hard-limit membership induced by the nearest centroid.
    """
    distance_table = _as_2d_float(distance_table, "distance_table")
    if f < 1.0:
        raise ValueError("The fuzzy degree f must be at least 1.00.")
    if f <= 1.0 + epsilon:
        membership, _ = hard_membership_from_distances(distance_table)
        return membership

    n_samples, n_clusters = distance_table.shape
    membership = np.zeros((n_samples, n_clusters), dtype=float)
    zero_mask = distance_table <= epsilon
    rows_with_zero = np.any(zero_mask, axis=1)

    if np.any(rows_with_zero):
        zero_counts = np.sum(zero_mask[rows_with_zero], axis=1, keepdims=True)
        membership[rows_with_zero] = zero_mask[rows_with_zero] / zero_counts

    rows_without_zero = ~rows_with_zero
    if np.any(rows_without_zero):
        exponent = 2.0 / (f - 1.0)
        safe_distances = np.maximum(distance_table[rows_without_zero], epsilon)
        reciprocal_scores = 1.0 / (safe_distances ** exponent)
        membership[rows_without_zero] = reciprocal_scores / np.sum(
            reciprocal_scores, axis=1, keepdims=True
        )

    return membership


def aggregation_weights(membership, f: float = 2.0, epsilon: float = EPSILON) -> np.ndarray:
    """Compute normalized output-aggregation weights from memberships."""
    membership = _as_2d_float(membership, "membership")
    if f < 1.0:
        raise ValueError("The fuzzy degree f must be at least 1.00.")

    powered = np.maximum(membership, 0.0) ** f
    row_sum = np.sum(powered, axis=1, keepdims=True)
    return powered / np.maximum(row_sum, epsilon)


def membership_entropy(membership, epsilon: float = EPSILON) -> float:
    """Compute average Shannon entropy across membership rows."""
    membership = _as_2d_float(membership, "membership")
    safe = np.clip(membership, epsilon, 1.0)
    return float(-np.mean(np.sum(safe * np.log(safe), axis=1)))


def square_probability_change(current_membership, previous_membership) -> float:
    """Compute squared-probability movement between two membership tables."""
    current = _as_2d_float(current_membership, "current_membership")
    previous = _as_2d_float(previous_membership, "previous_membership")
    if current.shape != previous.shape:
        raise ValueError("Membership tables must have the same shape.")
    return float(np.mean((current ** 2 - previous ** 2) ** 2))


def local_weighted_inputs(X, membership) -> np.ndarray:
    """Create the local gated inputs u_ij x_i for all observations and groups."""
    X = _as_2d_float(X, "X")
    membership = _as_2d_float(membership, "membership")
    if X.shape[0] != membership.shape[0]:
        raise ValueError("X and membership must have the same number of rows.")
    return membership[:, :, None] * X[:, None, :]


def dominant_membership_labels(membership) -> np.ndarray:
    """Return the highest-membership group for every observation."""
    membership = _as_2d_float(membership, "membership")
    return np.argmax(membership, axis=1).astype(int)

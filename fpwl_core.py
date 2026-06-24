"""
Core partition routines for fuzzy partition-weighted local learning.

The module contains the distance table, crisp and fuzzy membership updates,
centroid updates, and the reusable partition-learning loop used by regression
and classification experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from truncation_rules import check_truncation
except ImportError:  # Allows local imports when this file is used inside a package.
    from .truncation_rules import check_truncation

ArrayLike = Union[np.ndarray, List[List[float]], List[float]]


@dataclass
class PartitionResult:
    """Container for the learned partition and its convergence history."""

    centroids: np.ndarray
    membership: np.ndarray
    labels: np.ndarray
    distance_table: np.ndarray
    n_iter: int
    converged: bool
    history: List[Dict[str, float]] = field(default_factory=list)


def as_2d_float_array(x: ArrayLike, name: str = "array") -> np.ndarray:
    """Return input data as a finite two-dimensional float array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def minkowski_distance_table(
    x: ArrayLike,
    centroids: ArrayLike,
    p: float = 2.0,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Compute the Minkowski distance table between observations and centroids."""
    x_arr = as_2d_float_array(x, "x")
    c_arr = as_2d_float_array(centroids, "centroids")

    if x_arr.shape[1] != c_arr.shape[1]:
        raise ValueError("x and centroids must have the same number of columns.")
    if p <= 0:
        raise ValueError("p must be positive.")

    diff = np.abs(x_arr[:, None, :] - c_arr[None, :, :])
    if np.isinf(p):
        distances = diff.max(axis=2)
    else:
        distances = np.power(np.sum(np.power(diff, p), axis=2), 1.0 / p)
    return np.maximum(distances, eps)


def initialize_centroids(
    x: ArrayLike,
    k: int,
    random_state: Optional[int] = None,
    method: str = "kmeans++",
) -> np.ndarray:
    """Initialize centroids by random sampling or a k-means++ style rule."""
    x_arr = as_2d_float_array(x, "x")
    n_samples = x_arr.shape[0]
    if k < 1:
        raise ValueError("k must be at least 1.")
    if k > n_samples:
        raise ValueError("k cannot exceed the number of observations.")

    rng = np.random.default_rng(random_state)

    if method == "random":
        indices = rng.choice(n_samples, size=k, replace=False)
        return x_arr[indices].copy()

    if method != "kmeans++":
        raise ValueError("method must be 'kmeans++' or 'random'.")

    centroids = np.empty((k, x_arr.shape[1]), dtype=float)
    first_index = int(rng.integers(0, n_samples))
    centroids[0] = x_arr[first_index]

    closest_sq = np.sum((x_arr - centroids[0]) ** 2, axis=1)
    for j in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 0.0:
            candidates = rng.choice(n_samples, size=k - j, replace=False)
            centroids[j:] = x_arr[candidates]
            break
        probabilities = closest_sq / total
        next_index = int(rng.choice(n_samples, p=probabilities))
        centroids[j] = x_arr[next_index]
        new_sq = np.sum((x_arr - centroids[j]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)

    return centroids


def crisp_membership(distance_table: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a distance table into hard labels and one-hot memberships."""
    distances = as_2d_float_array(distance_table, "distance_table")
    labels = np.argmin(distances, axis=1)
    membership = np.zeros_like(distances, dtype=float)
    membership[np.arange(distances.shape[0]), labels] = 1.0
    return labels, membership


def fuzzy_membership(
    distance_table: np.ndarray,
    f: float = 2.0,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Convert distances into fuzzy memberships."""
    distances = as_2d_float_array(distance_table, "distance_table")
    if f <= 1.0:
        raise ValueError("f must be greater than 1 for fuzzy membership.")

    distances = np.maximum(distances, eps)
    exponent = 2.0 / (f - 1.0)
    ratios = (distances[:, :, None] / distances[:, None, :]) ** exponent
    denominator = np.sum(ratios, axis=2)
    membership = 1.0 / np.maximum(denominator, eps)
    membership = membership / np.maximum(membership.sum(axis=1, keepdims=True), eps)
    return membership


def labels_from_membership(membership: np.ndarray) -> np.ndarray:
    """Return the dominant group label for each membership row."""
    u = as_2d_float_array(membership, "membership")
    return np.argmax(u, axis=1)


def update_crisp_centroids(
    x: ArrayLike,
    labels: np.ndarray,
    k: int,
    old_centroids: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Update centroids by ordinary group means."""
    x_arr = as_2d_float_array(x, "x")
    labels = np.asarray(labels, dtype=int)
    if labels.shape[0] != x_arr.shape[0]:
        raise ValueError("labels must have one entry per observation.")

    rng = np.random.default_rng(random_state)
    centroids = np.zeros((k, x_arr.shape[1]), dtype=float)
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            centroids[j] = x_arr[mask].mean(axis=0)
        elif old_centroids is not None:
            centroids[j] = old_centroids[j]
        else:
            centroids[j] = x_arr[int(rng.integers(0, x_arr.shape[0]))]
    return centroids


def update_fuzzy_centroids(
    x: ArrayLike,
    membership: np.ndarray,
    f: float = 2.0,
    old_centroids: Optional[np.ndarray] = None,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Update centroids by membership-weighted means."""
    x_arr = as_2d_float_array(x, "x")
    u = as_2d_float_array(membership, "membership")
    if u.shape[0] != x_arr.shape[0]:
        raise ValueError("membership must have one row per observation.")
    if f <= 1.0:
        raise ValueError("f must be greater than 1 for fuzzy centroid updates.")

    weights = np.power(u, f)
    denominator = weights.sum(axis=0)
    centroids = np.zeros((u.shape[1], x_arr.shape[1]), dtype=float)
    for j in range(u.shape[1]):
        if denominator[j] > eps:
            centroids[j] = weights[:, j] @ x_arr / denominator[j]
        elif old_centroids is not None:
            centroids[j] = old_centroids[j]
        else:
            centroids[j] = x_arr.mean(axis=0)
    return centroids


def learn_partition(
    x: ArrayLike,
    k: int,
    partition: str = "fuzzy",
    f: float = 2.0,
    p: float = 2.0,
    truncation: str = "hpd",
    tolerance: float = 1.0e-5,
    max_iter: int = 300,
    random_state: Optional[int] = 42,
    init: str = "kmeans++",
    eps: float = 1.0e-12,
) -> PartitionResult:
    """Learn a crisp or fuzzy partition for a numerical dataset."""
    x_arr = as_2d_float_array(x, "x")
    if partition not in {"crisp", "fuzzy"}:
        raise ValueError("partition must be 'crisp' or 'fuzzy'.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    centroids = initialize_centroids(x_arr, k=k, random_state=random_state, method=init)
    previous_distances: Optional[np.ndarray] = None
    previous_membership: Optional[np.ndarray] = None
    previous_labels: Optional[np.ndarray] = None
    history: List[Dict[str, float]] = []
    converged = False

    labels = np.zeros(x_arr.shape[0], dtype=int)
    membership = np.zeros((x_arr.shape[0], k), dtype=float)
    distances = np.zeros((x_arr.shape[0], k), dtype=float)

    for iteration in range(1, max_iter + 1):
        distances = minkowski_distance_table(x_arr, centroids, p=p, eps=eps)

        if partition == "crisp":
            labels, membership = crisp_membership(distances)
            next_centroids = update_crisp_centroids(
                x_arr,
                labels,
                k=k,
                old_centroids=centroids,
                random_state=None if random_state is None else random_state + iteration,
            )
        else:
            membership = fuzzy_membership(distances, f=f, eps=eps)
            labels = labels_from_membership(membership)
            next_centroids = update_fuzzy_centroids(
                x_arr,
                membership,
                f=f,
                old_centroids=centroids,
                eps=eps,
            )

        decision = check_truncation(
            method=truncation,
            current_distances=distances,
            previous_distances=previous_distances,
            current_membership=membership,
            previous_membership=previous_membership,
            current_labels=labels,
            previous_labels=previous_labels,
            tolerance=tolerance,
            eps=eps,
        )
        history.append(
            {
                "iteration": float(iteration),
                "truncation_value": float(decision.value),
                "centroid_shift": float(np.linalg.norm(next_centroids - centroids)),
            }
        )

        centroids = next_centroids
        if decision.converged:
            converged = True
            break

        previous_distances = distances.copy()
        previous_membership = membership.copy()
        previous_labels = labels.copy()

    final_distances = minkowski_distance_table(x_arr, centroids, p=p, eps=eps)
    if partition == "crisp":
        labels, membership = crisp_membership(final_distances)
    else:
        membership = fuzzy_membership(final_distances, f=f, eps=eps)
        labels = labels_from_membership(membership)

    return PartitionResult(
        centroids=centroids,
        membership=membership,
        labels=labels,
        distance_table=final_distances,
        n_iter=len(history),
        converged=converged,
        history=history,
    )


def membership_for_new_points(
    x_new: ArrayLike,
    centroids: ArrayLike,
    partition: str = "fuzzy",
    f: float = 2.0,
    p: float = 2.0,
    eps: float = 1.0e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute memberships for new observations using fixed centroids."""
    distances = minkowski_distance_table(x_new, centroids, p=p, eps=eps)
    if partition == "crisp":
        labels, membership = crisp_membership(distances)
    elif partition == "fuzzy":
        membership = fuzzy_membership(distances, f=f, eps=eps)
        labels = labels_from_membership(membership)
    else:
        raise ValueError("partition must be 'crisp' or 'fuzzy'.")
    return membership, labels, distances


__all__ = [
    "PartitionResult",
    "as_2d_float_array",
    "minkowski_distance_table",
    "initialize_centroids",
    "crisp_membership",
    "fuzzy_membership",
    "labels_from_membership",
    "update_crisp_centroids",
    "update_fuzzy_centroids",
    "learn_partition",
    "membership_for_new_points",
]

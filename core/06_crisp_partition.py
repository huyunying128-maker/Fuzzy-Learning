"""
Crisp nearest-centroid partition learning.

This module fits the hard partition used by the crisp local learner and by the
modified k-means reference. Each observation is assigned to its nearest centroid,
and each centroid is updated by the ordinary group mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util
import sys

import numpy as np


EPSILON = 1e-12


def _load_core_module(file_name: str, alias: str):
    path = Path(__file__).resolve().with_name(file_name)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_dist = _load_core_module("03_distance_functions.py", "fpw_distance_functions")
_membership = _load_core_module("04_membership_functions.py", "fpw_membership_functions")
_truncation = _load_core_module("05_truncation_rules.py", "fpw_truncation_rules")


@dataclass
class CrispPartitionResult:
    """Fitted hard partition and its iteration record."""

    centroids: np.ndarray
    labels: np.ndarray
    membership: np.ndarray
    distance_table: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    truncation_rule: str
    truncation_value: float
    history: List[Dict[str, float]]


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def initialize_centroids(X, k: int, random_state: int = 42) -> np.ndarray:
    """Initialize centroids by selecting observations without replacement."""
    X = _as_2d_float(X, "X")
    if k < 1 or k > X.shape[0]:
        raise ValueError("k must be between 1 and the number of observations.")
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], size=k, replace=False)
    return np.array(X[indices], copy=True, dtype=float)


def update_crisp_centroids(X, labels, k: int, previous_centroids=None) -> np.ndarray:
    """Update centroids by ordinary group means."""
    X = _as_2d_float(X, "X")
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.shape[0] != X.shape[0]:
        raise ValueError("labels must have one entry for each observation.")

    centroids = np.zeros((k, X.shape[1]), dtype=float)
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            centroids[j] = np.mean(X[mask], axis=0)
        elif previous_centroids is not None:
            centroids[j] = previous_centroids[j]
        else:
            centroids[j] = X[np.argmax(np.sum((X - np.mean(X, axis=0)) ** 2, axis=1))]
    return centroids


def predict_crisp_partition(X, centroids, p: float = 2.0):
    """Assign observations to the nearest fitted centroid."""
    distance_table = _dist.minkowski_distance_matrix(X, centroids, p=p)
    labels = _dist.nearest_centroid_labels(distance_table)
    membership = _membership.hard_membership_from_labels(labels, centroids.shape[0])
    return membership, labels, distance_table


def fit_crisp_partition(
    X,
    k: int,
    p: float = 2.0,
    max_iter: int = 300,
    tolerance: float = 1e-6,
    truncation_rule: str = "hpd",
    random_state: int = 42,
    initial_centroids: Optional[np.ndarray] = None,
) -> CrispPartitionResult:
    """Fit a crisp nearest-centroid partition."""
    X = _as_2d_float(X, "X")
    if k < 2 or k > X.shape[0]:
        raise ValueError("k must be at least 2 and no larger than the number of observations.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")

    centroids = (
        np.array(initial_centroids, copy=True, dtype=float)
        if initial_centroids is not None
        else initialize_centroids(X, k, random_state=random_state)
    )
    if centroids.shape != (k, X.shape[1]):
        raise ValueError("initial_centroids must have shape (k, n_features).")

    history: List[Dict[str, float]] = []
    previous_state = None
    converged = False
    current_value = float("inf")

    for iteration in range(1, max_iter + 1):
        distance_table = _dist.minkowski_distance_matrix(X, centroids, p=p)
        labels = _dist.nearest_centroid_labels(distance_table)
        membership = _membership.hard_membership_from_labels(labels, k)
        objective = _dist.distance_objective(distance_table, labels)

        current_state = _truncation.TruncationState(
            distance_table=distance_table,
            membership=membership,
            labels=labels,
        )
        current_value = float("inf")
        if previous_state is not None:
            current_value = _truncation.truncation_value(truncation_rule, current_state, previous_state)
            history.append(
                {
                    "iteration": float(iteration),
                    "objective": float(objective),
                    "truncation_value": float(current_value),
                }
            )
            if _truncation.should_stop(current_value, tolerance):
                converged = True
                break
        else:
            history.append(
                {
                    "iteration": float(iteration),
                    "objective": float(objective),
                    "truncation_value": float("nan"),
                }
            )

        new_centroids = update_crisp_centroids(X, labels, k, previous_centroids=centroids)
        if np.max(np.abs(new_centroids - centroids)) <= tolerance:
            centroids = new_centroids
            converged = True
            break

        centroids = new_centroids
        previous_state = current_state

    distance_table = _dist.minkowski_distance_matrix(X, centroids, p=p)
    labels = _dist.nearest_centroid_labels(distance_table)
    membership = _membership.hard_membership_from_labels(labels, k)
    objective = _dist.distance_objective(distance_table, labels)

    return CrispPartitionResult(
        centroids=centroids,
        labels=labels,
        membership=membership,
        distance_table=distance_table,
        objective=float(objective),
        n_iter=int(iteration),
        converged=bool(converged),
        truncation_rule=truncation_rule,
        truncation_value=float(current_value),
        history=history,
    )


if __name__ == "__main__":
    sample = np.array([[1.0, 1.0], [1.2, 0.9], [5.0, 5.1], [5.2, 4.8]])
    result = fit_crisp_partition(sample, k=2, p=2.0)
    print("labels:", result.labels.tolist())

"""
Fuzzy partition learning with membership-weighted centroid updates.

The fitted partition keeps a graded relation between every observation and every
centroid. These memberships are later reused as local input gates and as
aggregation weights in the supervised learning layer.
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
_crisp = _load_core_module("06_crisp_partition.py", "fpw_crisp_partition")


@dataclass
class FuzzyPartitionResult:
    """Fitted fuzzy partition and its iteration record."""

    centroids: np.ndarray
    labels: np.ndarray
    membership: np.ndarray
    distance_table: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    f: float
    p: float
    truncation_rule: str
    truncation_value: float
    history: List[Dict[str, float]]


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def update_fuzzy_centroids(X, membership, f: float, previous_centroids=None) -> np.ndarray:
    """Update centroids by fuzzy membership-weighted means."""
    X = _as_2d_float(X, "X")
    membership = _as_2d_float(membership, "membership")
    if X.shape[0] != membership.shape[0]:
        raise ValueError("X and membership must have the same number of rows.")
    if f < 1.0:
        raise ValueError("f must be at least 1.00.")

    weights = np.maximum(membership, 0.0) ** f
    denominators = np.sum(weights, axis=0)
    centroids = np.zeros((membership.shape[1], X.shape[1]), dtype=float)
    for j in range(membership.shape[1]):
        if denominators[j] > EPSILON:
            centroids[j] = weights[:, j] @ X / denominators[j]
        elif previous_centroids is not None:
            centroids[j] = previous_centroids[j]
        else:
            centroids[j] = np.mean(X, axis=0)
    return centroids


def fuzzy_objective(distance_table, membership, f: float) -> float:
    """Compute the fuzzy partition objective based on distances and memberships."""
    distance_table = _as_2d_float(distance_table, "distance_table")
    membership = _as_2d_float(membership, "membership")
    if distance_table.shape != membership.shape:
        raise ValueError("distance_table and membership must have the same shape.")
    return float(np.mean((membership ** f) * (distance_table ** 2)))


def predict_fuzzy_partition(X, centroids, f: float = 2.0, p: float = 2.0):
    """Compute fuzzy memberships for new observations from fitted centroids."""
    distance_table = _dist.minkowski_distance_matrix(X, centroids, p=p)
    membership = _membership.fuzzy_membership_from_distances(distance_table, f=f)
    labels = _membership.dominant_membership_labels(membership)
    return membership, labels, distance_table


def fit_fuzzy_partition(
    X,
    k: int,
    f: float = 2.0,
    p: float = 2.0,
    max_iter: int = 300,
    tolerance: float = 1e-6,
    truncation_rule: str = "hpd",
    random_state: int = 42,
    initial_centroids: Optional[np.ndarray] = None,
) -> FuzzyPartitionResult:
    """Fit a fuzzy partition using reciprocal-distance memberships."""
    X = _as_2d_float(X, "X")
    if k < 2 or k > X.shape[0]:
        raise ValueError("k must be at least 2 and no larger than the number of observations.")
    if f < 1.0:
        raise ValueError("f must be at least 1.00.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")

    centroids = (
        np.array(initial_centroids, copy=True, dtype=float)
        if initial_centroids is not None
        else _crisp.initialize_centroids(X, k, random_state=random_state)
    )
    if centroids.shape != (k, X.shape[1]):
        raise ValueError("initial_centroids must have shape (k, n_features).")

    history: List[Dict[str, float]] = []
    previous_state = None
    converged = False
    current_value = float("inf")

    for iteration in range(1, max_iter + 1):
        distance_table = _dist.minkowski_distance_matrix(X, centroids, p=p)
        membership = _membership.fuzzy_membership_from_distances(distance_table, f=f)
        labels = _membership.dominant_membership_labels(membership)
        objective = fuzzy_objective(distance_table, membership, f=f)

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

        new_centroids = update_fuzzy_centroids(X, membership, f=f, previous_centroids=centroids)
        if np.max(np.abs(new_centroids - centroids)) <= tolerance:
            centroids = new_centroids
            converged = True
            break

        centroids = new_centroids
        previous_state = current_state

    distance_table = _dist.minkowski_distance_matrix(X, centroids, p=p)
    membership = _membership.fuzzy_membership_from_distances(distance_table, f=f)
    labels = _membership.dominant_membership_labels(membership)
    objective = fuzzy_objective(distance_table, membership, f=f)

    return FuzzyPartitionResult(
        centroids=centroids,
        labels=labels,
        membership=membership,
        distance_table=distance_table,
        objective=float(objective),
        n_iter=int(iteration),
        converged=bool(converged),
        f=float(f),
        p=float(p),
        truncation_rule=truncation_rule,
        truncation_value=float(current_value),
        history=history,
    )


if __name__ == "__main__":
    sample = np.array([[1.0, 1.0], [1.2, 0.9], [5.0, 5.1], [5.2, 4.8]])
    result = fit_fuzzy_partition(sample, k=2, f=2.0, p=2.0)
    print("memberships:\n", np.round(result.membership, 3))

"""
Truncation rules for iterative partition learning.

The module implements distance-table difference, entropy, harmonic distance
change, square-probability change, and hereditary partition distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TruncationDecision:
    """Value and convergence status of a truncation rule."""

    value: float
    converged: bool
    method: str


def _as_float_matrix(value: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_label_vector(value: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=int).reshape(-1)
    return arr


def distance_table_difference(
    current_distances: np.ndarray,
    previous_distances: Optional[np.ndarray],
) -> float:
    """Average absolute change in the distance table."""
    current = _as_float_matrix(current_distances, "current_distances")
    previous = _as_float_matrix(previous_distances, "previous_distances")
    if previous is None:
        return float("inf")
    if current.shape != previous.shape:
        raise ValueError("current_distances and previous_distances must have the same shape.")
    return float(np.mean(np.abs(current - previous)))


def harmonic_distance_change(
    current_distances: np.ndarray,
    previous_distances: Optional[np.ndarray],
    eps: float = 1.0e-12,
) -> float:
    """Harmonic mean of absolute distance-table changes."""
    current = _as_float_matrix(current_distances, "current_distances")
    previous = _as_float_matrix(previous_distances, "previous_distances")
    if previous is None:
        return float("inf")
    if current.shape != previous.shape:
        raise ValueError("current_distances and previous_distances must have the same shape.")
    delta = np.abs(current - previous) + eps
    return float(delta.size / np.sum(1.0 / delta))


def membership_entropy(current_membership: np.ndarray, eps: float = 1.0e-12) -> float:
    """Mean Shannon entropy of membership rows."""
    membership = _as_float_matrix(current_membership, "current_membership")
    membership = np.clip(membership, eps, 1.0)
    return float(-np.mean(np.sum(membership * np.log(membership), axis=1)))


def entropy_change(
    current_membership: np.ndarray,
    previous_membership: Optional[np.ndarray],
    eps: float = 1.0e-12,
) -> float:
    """Absolute change in mean membership entropy."""
    current = _as_float_matrix(current_membership, "current_membership")
    previous = _as_float_matrix(previous_membership, "previous_membership")
    if previous is None:
        return float("inf")
    if current.shape != previous.shape:
        raise ValueError("current_membership and previous_membership must have the same shape.")
    return abs(membership_entropy(current, eps=eps) - membership_entropy(previous, eps=eps))


def square_probability_change(
    current_membership: np.ndarray,
    previous_membership: Optional[np.ndarray],
) -> float:
    """Mean squared change in squared membership probabilities."""
    current = _as_float_matrix(current_membership, "current_membership")
    previous = _as_float_matrix(previous_membership, "previous_membership")
    if previous is None:
        return float("inf")
    if current.shape != previous.shape:
        raise ValueError("current_membership and previous_membership must have the same shape.")
    return float(np.mean((current**2 - previous**2) ** 2))


def hereditary_partition_distance(
    current_labels: np.ndarray,
    previous_labels: Optional[np.ndarray],
    normalize: bool = True,
) -> float:
    """Hereditary partition distance based on intersection refinement."""
    current = _as_label_vector(current_labels, "current_labels")
    previous = _as_label_vector(previous_labels, "previous_labels")
    if previous is None:
        return float("inf")
    if current.shape[0] != previous.shape[0]:
        raise ValueError("current_labels and previous_labels must have the same length.")

    _, current_inverse = np.unique(current, return_inverse=True)
    _, previous_inverse = np.unique(previous, return_inverse=True)
    n_current = int(current_inverse.max()) + 1
    n_previous = int(previous_inverse.max()) + 1

    contingency = np.zeros((n_current, n_previous), dtype=float)
    np.add.at(contingency, (current_inverse, previous_inverse), 1.0)

    current_sizes = contingency.sum(axis=1)
    previous_sizes = contingency.sum(axis=0)
    rho_current = np.sum(current_sizes**2)
    rho_previous = np.sum(previous_sizes**2)
    rho_refinement = np.sum(contingency**2)

    distance = 0.5 * (rho_current + rho_previous) - rho_refinement
    if normalize:
        n = float(current.shape[0])
        if n > 0:
            distance = distance / (n * n)
    return float(max(distance, 0.0))


def check_truncation(
    method: str,
    current_distances: Optional[np.ndarray] = None,
    previous_distances: Optional[np.ndarray] = None,
    current_membership: Optional[np.ndarray] = None,
    previous_membership: Optional[np.ndarray] = None,
    current_labels: Optional[np.ndarray] = None,
    previous_labels: Optional[np.ndarray] = None,
    tolerance: float = 1.0e-5,
    eps: float = 1.0e-12,
) -> TruncationDecision:
    """Evaluate a named truncation rule."""
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")

    key = method.lower().strip().replace("-", "_")
    if key in {"dtd", "distance_table_difference"}:
        value = distance_table_difference(current_distances, previous_distances)
        name = "dtd"
    elif key in {"harmonic", "hm", "harmonic_distance"}:
        value = harmonic_distance_change(current_distances, previous_distances, eps=eps)
        name = "harmonic"
    elif key in {"entropy", "shannon_entropy"}:
        value = entropy_change(current_membership, previous_membership, eps=eps)
        name = "entropy"
    elif key in {"sp", "square_probability"}:
        value = square_probability_change(current_membership, previous_membership)
        name = "sp"
    elif key in {"hpd", "hereditary", "hereditary_partition_distance"}:
        value = hereditary_partition_distance(current_labels, previous_labels, normalize=True)
        name = "hpd"
    else:
        raise ValueError(
            "method must be one of: dtd, harmonic, entropy, sp, hpd."
        )

    return TruncationDecision(
        value=float(value),
        converged=bool(np.isfinite(value) and value <= tolerance),
        method=name,
    )


__all__ = [
    "TruncationDecision",
    "distance_table_difference",
    "harmonic_distance_change",
    "membership_entropy",
    "entropy_change",
    "square_probability_change",
    "hereditary_partition_distance",
    "check_truncation",
]

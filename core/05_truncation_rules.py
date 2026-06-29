"""
Truncation rules for iterative crisp and fuzzy partition learning.

The partition update can be stopped by numerical distance movement, membership
movement, membership uncertainty, or structural change between two induced
partitions. These quantities are stored in the iteration history so that the
learning path remains inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


EPSILON = 1e-12
SUPPORTED_TRUNCATION_RULES = ("dtd", "harmonic", "sp", "entropy", "hpd")


@dataclass(frozen=True)
class TruncationState:
    """Values used to compare two successive partition iterations."""

    distance_table: Optional[np.ndarray] = None
    membership: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def _as_labels(labels, name: str) -> np.ndarray:
    value = np.asarray(labels, dtype=int).reshape(-1)
    if value.size == 0:
        raise ValueError(f"{name} must contain at least one label.")
    return value


def distance_table_difference(current_distance_table, previous_distance_table) -> float:
    """Average absolute movement of the distance table."""
    current = _as_2d_float(current_distance_table, "current_distance_table")
    previous = _as_2d_float(previous_distance_table, "previous_distance_table")
    if current.shape != previous.shape:
        raise ValueError("Distance tables must have the same shape.")
    return float(np.mean(np.abs(current - previous)))


def harmonic_distance_difference(
    current_distance_table,
    previous_distance_table,
    epsilon: float = EPSILON,
) -> float:
    """Harmonic summary of the absolute distance-table movement."""
    current = _as_2d_float(current_distance_table, "current_distance_table")
    previous = _as_2d_float(previous_distance_table, "previous_distance_table")
    if current.shape != previous.shape:
        raise ValueError("Distance tables must have the same shape.")

    change = np.abs(current - previous)
    return float(change.size / np.sum(1.0 / (change + epsilon)))


def square_probability_difference(current_membership, previous_membership) -> float:
    """Average squared movement of the squared membership probabilities."""
    current = _as_2d_float(current_membership, "current_membership")
    previous = _as_2d_float(previous_membership, "previous_membership")
    if current.shape != previous.shape:
        raise ValueError("Membership tables must have the same shape.")
    return float(np.mean((current ** 2 - previous ** 2) ** 2))


def membership_entropy(membership, epsilon: float = EPSILON) -> float:
    """Average Shannon entropy of the membership rows."""
    membership = _as_2d_float(membership, "membership")
    safe = np.clip(membership, epsilon, 1.0)
    return float(-np.mean(np.sum(safe * np.log(safe), axis=1)))


def entropy_difference(current_membership, previous_membership) -> float:
    """Absolute change in average membership entropy."""
    return float(abs(membership_entropy(current_membership) - membership_entropy(previous_membership)))


def contingency_table(labels_a, labels_b) -> np.ndarray:
    """Build the contingency table of two hard partitions."""
    labels_a = _as_labels(labels_a, "labels_a")
    labels_b = _as_labels(labels_b, "labels_b")
    if labels_a.shape[0] != labels_b.shape[0]:
        raise ValueError("The two label vectors must have the same length.")

    _, inv_a = np.unique(labels_a, return_inverse=True)
    _, inv_b = np.unique(labels_b, return_inverse=True)
    table = np.zeros((inv_a.max() + 1, inv_b.max() + 1), dtype=np.int64)
    np.add.at(table, (inv_a, inv_b), 1)
    return table


def _choose_two(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * (values - 1.0) / 2.0


def hereditary_partition_distance(labels_a, labels_b, normalize: bool = True) -> float:
    """Pair-weighted hereditary distance between two induced partitions.

    The implementation uses the pair contribution sigma(A)=C(|A|, 2). This is a
    stable finite-sample form of the hereditary partition distance: it compares
    the pair structure preserved by two successive partitions and becomes zero
    when the induced partitions are identical.
    """
    table = contingency_table(labels_a, labels_b)
    n_samples = int(table.sum())
    if n_samples <= 1:
        return 0.0

    rho_a = float(np.sum(_choose_two(table.sum(axis=1))))
    rho_b = float(np.sum(_choose_two(table.sum(axis=0))))
    rho_refinement = float(np.sum(_choose_two(table)))
    distance = 0.5 * (rho_a + rho_b) - rho_refinement

    if normalize:
        total_pairs = n_samples * (n_samples - 1.0) / 2.0
        distance = distance / max(total_pairs, EPSILON)
    return float(distance)


def truncation_value(rule: str, current: TruncationState, previous: TruncationState) -> float:
    """Compute one truncation quantity from two successive iteration states."""
    rule = rule.lower()
    if rule not in SUPPORTED_TRUNCATION_RULES:
        raise ValueError(f"Unsupported truncation rule: {rule}")

    if rule == "dtd":
        if current.distance_table is None or previous.distance_table is None:
            raise ValueError("DTD requires two distance tables.")
        return distance_table_difference(current.distance_table, previous.distance_table)

    if rule == "harmonic":
        if current.distance_table is None or previous.distance_table is None:
            raise ValueError("Harmonic truncation requires two distance tables.")
        return harmonic_distance_difference(current.distance_table, previous.distance_table)

    if rule == "sp":
        if current.membership is None or previous.membership is None:
            raise ValueError("SP truncation requires two membership tables.")
        return square_probability_difference(current.membership, previous.membership)

    if rule == "entropy":
        if current.membership is None or previous.membership is None:
            raise ValueError("Entropy truncation requires two membership tables.")
        return entropy_difference(current.membership, previous.membership)

    if current.labels is None or previous.labels is None:
        raise ValueError("HPD truncation requires two induced label vectors.")
    return hereditary_partition_distance(current.labels, previous.labels)


def should_stop(value: float, tolerance: float) -> bool:
    """Return whether a truncation value satisfies the tolerance condition."""
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")
    return bool(value <= tolerance)


def summarize_truncation_values(current: TruncationState, previous: TruncationState) -> Dict[str, float]:
    """Return all truncation quantities that can be computed from the states."""
    summary: Dict[str, float] = {}
    for rule in SUPPORTED_TRUNCATION_RULES:
        try:
            summary[rule] = truncation_value(rule, current, previous)
        except ValueError:
            continue
    return summary


if __name__ == "__main__":
    labels_0 = np.array([0, 0, 1, 1, 2, 2])
    labels_1 = np.array([0, 0, 1, 2, 2, 2])
    print("HPD:", hereditary_partition_distance(labels_0, labels_1))

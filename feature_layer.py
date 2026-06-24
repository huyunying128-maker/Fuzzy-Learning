"""
Membership-based feature construction for partition-weighted learning.

The module builds local weighted inputs, normalized aggregation weights, and
partition-weighted feature matrices for external machine-learning models.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray


def _as_2d_float_array(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def validate_x_membership(x: ArrayLike, membership: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Return checked input and membership matrices."""
    x_arr = _as_2d_float_array(x, "x")
    u_arr = _as_2d_float_array(membership, "membership")
    if x_arr.shape[0] != u_arr.shape[0]:
        raise ValueError("x and membership must have the same number of rows.")
    row_sums = u_arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("each membership row must have a positive sum.")
    u_arr = u_arr / row_sums
    return x_arr, u_arr


def aggregation_weights(
    membership: ArrayLike,
    f: float = 2.0,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Compute normalized fuzzy aggregation weights."""
    u_arr = _as_2d_float_array(membership, "membership")
    if f <= 0:
        raise ValueError("f must be positive.")
    weights = np.power(np.maximum(u_arr, eps), f)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), eps)
    return weights


def membership_weighted_inputs(
    x: ArrayLike,
    membership: ArrayLike,
) -> np.ndarray:
    """Build local inputs u_ij x_i for all observations and groups."""
    x_arr, u_arr = validate_x_membership(x, membership)
    return u_arr[:, :, None] * x_arr[:, None, :]


def local_input_for_group(
    x: ArrayLike,
    membership: ArrayLike,
    group_index: int,
) -> np.ndarray:
    """Return the membership-weighted input matrix for one local group."""
    weighted = membership_weighted_inputs(x, membership)
    if group_index < 0 or group_index >= weighted.shape[1]:
        raise IndexError("group_index is outside the membership range.")
    return weighted[:, group_index, :]


def flatten_local_inputs(weighted_inputs: ArrayLike) -> np.ndarray:
    """Flatten a three-dimensional local-input tensor into a feature matrix."""
    arr = np.asarray(weighted_inputs, dtype=float)
    if arr.ndim != 3:
        raise ValueError("weighted_inputs must have shape (n_samples, k, n_features).")
    n_samples = arr.shape[0]
    return arr.reshape(n_samples, -1)


def partition_weighted_feature_layer(
    x: ArrayLike,
    membership: ArrayLike,
    include_original: bool = True,
    include_membership: bool = True,
    include_weighted_inputs: bool = True,
    f: Optional[float] = None,
) -> np.ndarray:
    """Build the feature layer [x, u, u_1 x, ..., u_k x]."""
    x_arr, u_arr = validate_x_membership(x, membership)
    pieces: List[np.ndarray] = []

    if include_original:
        pieces.append(x_arr)
    if include_membership:
        pieces.append(u_arr)
    if include_weighted_inputs:
        local_inputs = membership_weighted_inputs(x_arr, u_arr)
        if f is not None:
            local_inputs = aggregation_weights(u_arr, f=f)[:, :, None] * x_arr[:, None, :]
        pieces.append(flatten_local_inputs(local_inputs))

    if not pieces:
        raise ValueError("at least one feature component must be included.")
    return np.concatenate(pieces, axis=1)


def feature_layer_names(
    base_feature_names: Sequence[str],
    k: int,
    include_original: bool = True,
    include_membership: bool = True,
    include_weighted_inputs: bool = True,
) -> List[str]:
    """Create column names for a partition-weighted feature layer."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    base_names = [str(name) for name in base_feature_names]
    names: List[str] = []

    if include_original:
        names.extend(base_names)
    if include_membership:
        names.extend([f"membership_{j}" for j in range(k)])
    if include_weighted_inputs:
        for j in range(k):
            names.extend([f"group_{j}_weighted_{name}" for name in base_names])
    return names


def dominant_membership_summary(membership: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Return dominant group labels and dominant membership values."""
    u_arr = _as_2d_float_array(membership, "membership")
    labels = np.argmax(u_arr, axis=1)
    strengths = u_arr[np.arange(u_arr.shape[0]), labels]
    return labels, strengths


__all__ = [
    "validate_x_membership",
    "aggregation_weights",
    "membership_weighted_inputs",
    "local_input_for_group",
    "flatten_local_inputs",
    "partition_weighted_feature_layer",
    "feature_layer_names",
    "dominant_membership_summary",
]

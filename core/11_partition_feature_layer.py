"""
Membership-based feature-layer construction for external learners.

The feature layer concatenates the original input, the membership vector, and
membership-weighted copies of the input. This representation lets neural,
kernel, tree-based, and boosting models receive the same partition information
used by the local learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import numpy as np


@dataclass
class FeatureLayerMetadata:
    """Column names and dimensions for a partition-weighted feature layer."""

    n_original_features: int
    n_groups: int
    original_feature_names: List[str]
    membership_feature_names: List[str]
    gated_feature_names: List[str]

    @property
    def feature_names(self) -> List[str]:
        return self.original_feature_names + self.membership_feature_names + self.gated_feature_names


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def build_partition_feature_layer(
    X,
    membership,
    include_original: bool = True,
    include_membership: bool = True,
    include_gated: bool = True,
) -> np.ndarray:
    """Create z_i = [x_i, u_i, u_i1 x_i, ..., u_ik x_i]."""
    X = _as_2d_float(X, "X")
    membership = _as_2d_float(membership, "membership")
    if X.shape[0] != membership.shape[0]:
        raise ValueError("X and membership must have the same number of rows.")
    if not (include_original or include_membership or include_gated):
        raise ValueError("At least one feature block must be included.")

    blocks = []
    if include_original:
        blocks.append(X)
    if include_membership:
        blocks.append(membership)
    if include_gated:
        gated = membership[:, :, None] * X[:, None, :]
        blocks.append(gated.reshape(X.shape[0], membership.shape[1] * X.shape[1]))
    return np.concatenate(blocks, axis=1)


def feature_layer_metadata(
    n_original_features: int,
    n_groups: int,
    original_feature_names: Optional[Sequence[str]] = None,
) -> FeatureLayerMetadata:
    """Create column metadata for the partition-weighted feature layer."""
    if n_original_features < 1:
        raise ValueError("n_original_features must be positive.")
    if n_groups < 1:
        raise ValueError("n_groups must be positive.")

    if original_feature_names is None:
        original = [f"x{index + 1}" for index in range(n_original_features)]
    else:
        original = [str(name) for name in original_feature_names]
        if len(original) != n_original_features:
            raise ValueError("original_feature_names must match n_original_features.")

    membership_names = [f"u{j + 1}" for j in range(n_groups)]
    gated_names = [f"u{j + 1}_{name}" for j in range(n_groups) for name in original]
    return FeatureLayerMetadata(
        n_original_features=int(n_original_features),
        n_groups=int(n_groups),
        original_feature_names=original,
        membership_feature_names=membership_names,
        gated_feature_names=gated_names,
    )


def save_feature_layer(
    X_layer,
    output_path: Path,
    metadata: Optional[FeatureLayerMetadata] = None,
) -> Path:
    """Save a feature-layer array and optional metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X_layer=np.asarray(X_layer, dtype=float))

    if metadata is not None:
        metadata_path = output_path.with_suffix(".metadata.joblib")
        joblib.dump(metadata, metadata_path)
    return output_path


def load_feature_layer(input_path: Path) -> np.ndarray:
    """Load a saved partition-weighted feature-layer array."""
    data = np.load(Path(input_path))
    return data["X_layer"]


def concatenate_original_and_membership(X, membership) -> np.ndarray:
    """Build the compact layer [x_i, u_i]."""
    return build_partition_feature_layer(
        X,
        membership,
        include_original=True,
        include_membership=True,
        include_gated=False,
    )


def concatenate_gated_only(X, membership) -> np.ndarray:
    """Build the gated-copy block [u_i1 x_i, ..., u_ik x_i]."""
    return build_partition_feature_layer(
        X,
        membership,
        include_original=False,
        include_membership=False,
        include_gated=True,
    )


if __name__ == "__main__":
    X_demo = np.array([[1.0, 2.0], [3.0, 4.0]])
    u_demo = np.array([[0.8, 0.2], [0.1, 0.9]])
    print(build_partition_feature_layer(X_demo, u_demo))

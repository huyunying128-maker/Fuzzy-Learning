"""
Polynomial feature construction for local regression and local logit models.

The local learner evaluates degrees 1 through 4 as fixed model settings. Each
basis contains an intercept, the first-order variables, and the interaction
monomials up to the selected degree.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Iterable, List, Sequence, Tuple

import numpy as np


SUPPORTED_DEGREES = (1, 2, 3, 4)


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def validate_degree(degree: int) -> int:
    """Return a validated polynomial degree."""
    degree = int(degree)
    if degree not in SUPPORTED_DEGREES:
        raise ValueError(f"degree must be one of {SUPPORTED_DEGREES}.")
    return degree


def monomial_powers(n_features: int, degree: int) -> List[Tuple[int, ...]]:
    """List the feature-index tuples used by the polynomial basis."""
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    degree = validate_degree(degree)

    powers: List[Tuple[int, ...]] = []
    for order in range(1, degree + 1):
        powers.extend(combinations_with_replacement(range(n_features), order))
    return powers


def polynomial_feature_names(feature_names: Sequence[str], degree: int) -> List[str]:
    """Create readable names for the polynomial basis columns."""
    if not feature_names:
        raise ValueError("feature_names must contain at least one name.")
    degree = validate_degree(degree)

    names = ["intercept"]
    for power in monomial_powers(len(feature_names), degree):
        counts = {}
        for index in power:
            counts[index] = counts.get(index, 0) + 1
        terms = []
        for index in sorted(counts):
            base = str(feature_names[index])
            exponent = counts[index]
            terms.append(base if exponent == 1 else f"{base}^{exponent}")
        names.append("*".join(terms))
    return names


def polynomial_design_matrix(X, degree: int, include_intercept: bool = True) -> np.ndarray:
    """Build a dense polynomial design matrix up to the selected degree."""
    X = _as_2d_float(X, "X")
    degree = validate_degree(degree)

    columns = []
    if include_intercept:
        columns.append(np.ones(X.shape[0], dtype=float))

    for power in monomial_powers(X.shape[1], degree):
        column = np.ones(X.shape[0], dtype=float)
        for index in power:
            column *= X[:, index]
        columns.append(column)

    return np.column_stack(columns)


def transform_local_weighted_inputs(local_inputs, degree: int) -> List[np.ndarray]:
    """Build one polynomial design matrix for each local group."""
    local_inputs = np.asarray(local_inputs, dtype=float)
    if local_inputs.ndim != 3:
        raise ValueError("local_inputs must have shape (n_samples, n_groups, n_features).")

    return [
        polynomial_design_matrix(local_inputs[:, j, :], degree=degree, include_intercept=True)
        for j in range(local_inputs.shape[1])
    ]


def count_polynomial_features(n_features: int, degree: int, include_intercept: bool = True) -> int:
    """Return the number of columns in the polynomial design matrix."""
    total = len(monomial_powers(n_features, degree))
    if include_intercept:
        total += 1
    return total


def screen_columns_by_variance(design_matrix, min_variance: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Remove numerically constant non-intercept columns from a design matrix."""
    design_matrix = _as_2d_float(design_matrix, "design_matrix")
    if design_matrix.shape[1] == 0:
        raise ValueError("design_matrix must contain at least one column.")
    if min_variance < 0:
        raise ValueError("min_variance must be nonnegative.")

    keep = np.ones(design_matrix.shape[1], dtype=bool)
    if design_matrix.shape[1] > 1:
        variances = np.var(design_matrix[:, 1:], axis=0)
        keep[1:] = variances >= min_variance
    return design_matrix[:, keep], keep


if __name__ == "__main__":
    X_demo = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(polynomial_design_matrix(X_demo, degree=2))

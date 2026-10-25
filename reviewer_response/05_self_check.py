"""Run small, data-free checks for the core reviewer-response utilities."""

from __future__ import annotations

import numpy as np

from _shared import (
    _distance,
    _fuzzy,
    evaluate_predictions,
    fit_local_predict,
    fit_partition,
    membership_for,
)


def main() -> None:
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(60, 3))
    y_reg = 1.7 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.1, size=60)

    metrics = evaluate_predictions("regression", y_reg, y_reg + 0.1)
    expected_r2 = 1.0 - metrics["mse"] / metrics["target_variance"]
    assert abs(metrics["r2"] - expected_r2) < 1e-12

    centroids = X[[0, 10, 20, 30]]
    full = _distance.minkowski_distance_matrix(X, centroids, p=2.0, chunk_size=len(X))
    chunked = _distance.minkowski_distance_matrix(X, centroids, p=2.0, chunk_size=7)
    assert np.allclose(full, chunked, rtol=1e-12, atol=1e-12)

    partition, _ = fit_partition(
        X, k=3, f=1.5, p=2.0, truncation_rule="hpd",
        tolerance=1e-6, max_iter=10, random_state=7
    )
    U = membership_for(X, partition, f=1.5, p=2.0)
    assert U.shape == (X.shape[0], 3)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-8)

    y_pred = fit_local_predict(
        X[:45], y_reg[:45], U[:45], X[45:], U[45:],
        task="regression", degree=1, regularization=0.1, f=1.5,
        input_gating=True, membership_weighted_fit=True,
        aggregation="fuzzy", random_state=7,
    )
    assert y_pred.shape == (15,)
    assert np.all(np.isfinite(y_pred))

    y_class = (X[:, 0] > 0).astype(int)
    class_partition, _ = fit_partition(
        X, k=3, f=1.5, p=2.0, truncation_rule="hpd",
        tolerance=1e-6, max_iter=10, random_state=8
    )
    U_class = membership_for(X, class_partition, f=1.5, p=2.0)
    probabilities = fit_local_predict(
        X[:45], y_class[:45], U_class[:45], X[45:], U_class[45:],
        task="classification", degree=1, regularization=0.1, f=1.5,
        input_gating=True, membership_weighted_fit=True,
        aggregation="fuzzy", random_state=8,
    )
    assert probabilities.shape == (15, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)

    print("Self-check passed: metrics, chunked distances, fuzzy memberships, and local regression/classification.")


if __name__ == "__main__":
    main()

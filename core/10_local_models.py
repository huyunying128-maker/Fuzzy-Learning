"""
Local supervised models for partition-weighted learning.

The regression branch fits one weighted ridge model for each local group and
aggregates the local predictions by normalized membership weights. The
classification branch fits local logit models and aggregates local logits before
applying the multiclass softmax transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import importlib.util
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


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


_metrics = _load_core_module("02_metrics.py", "fpw_metrics")
_membership = _load_core_module("04_membership_functions.py", "fpw_membership_functions")
_poly = _load_core_module("09_polynomial_features.py", "fpw_polynomial_features")


@dataclass
class LocalRegressionResult:
    """Fitted local ridge models and aggregation settings."""

    degree: int
    f: float
    alpha: float
    models: List[Ridge]
    n_groups: int
    n_features: int
    group_weight_sums: np.ndarray
    training_metrics: Dict[str, float]


@dataclass
class ConstantLogitModel:
    """Fallback local classifier for a group with one observed class."""

    log_prior: np.ndarray

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X)
        return np.tile(self.log_prior, (X.shape[0], 1))


@dataclass
class LocalClassificationResult:
    """Fitted local logit models and class metadata."""

    degree: int
    f: float
    models: List[object]
    classes: np.ndarray
    n_groups: int
    n_features: int
    group_weight_sums: np.ndarray
    training_metrics: Dict[str, float]


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def _validate_membership(X: np.ndarray, membership) -> np.ndarray:
    membership = _as_2d_float(membership, "membership")
    if membership.shape[0] != X.shape[0]:
        raise ValueError("X and membership must have the same number of rows.")
    return membership


def _effective_weights(membership: np.ndarray, f: float) -> np.ndarray:
    if f < 1.0:
        raise ValueError("The fuzzy degree f must be at least 1.00.")
    return np.maximum(membership, 0.0) ** float(f)


def fit_partition_weighted_regression(
    X,
    y,
    membership,
    degree: int,
    f: float = 2.0,
    alpha: float = 1e-4,
) -> LocalRegressionResult:
    """Fit the local polynomial ridge regression branch."""
    X = _as_2d_float(X, "X")
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y must have one value for each observation.")
    membership = _validate_membership(X, membership)
    degree = _poly.validate_degree(degree)

    local_inputs = _membership.local_weighted_inputs(X, membership)
    weights = _effective_weights(membership, f=f)
    models: List[Ridge] = []

    for j in range(membership.shape[1]):
        design = _poly.polynomial_design_matrix(local_inputs[:, j, :], degree=degree)
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(design, y, sample_weight=np.maximum(weights[:, j], EPSILON))
        models.append(model)

    fitted = LocalRegressionResult(
        degree=degree,
        f=float(f),
        alpha=float(alpha),
        models=models,
        n_groups=int(membership.shape[1]),
        n_features=int(X.shape[1]),
        group_weight_sums=np.sum(weights, axis=0),
        training_metrics={},
    )
    fitted.training_metrics = _metrics.regression_metrics(
        y, predict_partition_weighted_regression(fitted, X, membership)
    )
    return fitted


def local_regression_predictions(result: LocalRegressionResult, X, membership) -> np.ndarray:
    """Return the matrix of local regression predictions."""
    X = _as_2d_float(X, "X")
    membership = _validate_membership(X, membership)
    if membership.shape[1] != result.n_groups:
        raise ValueError("membership has a different number of groups from the fitted result.")

    local_inputs = _membership.local_weighted_inputs(X, membership)
    predictions = np.zeros((X.shape[0], result.n_groups), dtype=float)
    for j, model in enumerate(result.models):
        design = _poly.polynomial_design_matrix(local_inputs[:, j, :], degree=result.degree)
        predictions[:, j] = model.predict(design)
    return predictions


def predict_partition_weighted_regression(result: LocalRegressionResult, X, membership) -> np.ndarray:
    """Aggregate local regression predictions into the final prediction."""
    local_preds = local_regression_predictions(result, X, membership)
    weights = _membership.aggregation_weights(membership, f=result.f)
    return np.sum(weights * local_preds, axis=1)


def _class_prior_logits(y: np.ndarray, classes: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    counts = np.zeros(classes.shape[0], dtype=float)
    for index, cls in enumerate(classes):
        counts[index] = np.sum(sample_weight[y == cls])
    probabilities = (counts + 1.0) / (np.sum(counts) + classes.shape[0])
    return np.log(np.clip(probabilities, EPSILON, 1.0))


def fit_partition_weighted_classification(
    X,
    y,
    membership,
    degree: int,
    f: float = 2.0,
    max_iter: int = 1000,
    c_value: float = 1.0,
) -> LocalClassificationResult:
    """Fit the local logit classification branch."""
    X = _as_2d_float(X, "X")
    y = np.asarray(y, dtype=int).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y must have one label for each observation.")
    membership = _validate_membership(X, membership)
    degree = _poly.validate_degree(degree)

    classes = np.unique(y)
    local_inputs = _membership.local_weighted_inputs(X, membership)
    weights = _effective_weights(membership, f=f)
    models: List[object] = []

    for j in range(membership.shape[1]):
        design = _poly.polynomial_design_matrix(local_inputs[:, j, :], degree=degree)
        sample_weight = np.maximum(weights[:, j], EPSILON)
        active_classes = np.unique(y[sample_weight > EPSILON])

        if active_classes.shape[0] < 2 or np.sum(sample_weight) <= EPSILON:
            models.append(ConstantLogitModel(_class_prior_logits(y, classes, sample_weight)))
            continue

        model = LogisticRegression(
            C=float(c_value),
            max_iter=int(max_iter),
            solver="lbfgs",
            multi_class="auto",
        )
        model.fit(design, y, sample_weight=sample_weight)
        models.append(model)

    fitted = LocalClassificationResult(
        degree=degree,
        f=float(f),
        models=models,
        classes=classes,
        n_groups=int(membership.shape[1]),
        n_features=int(X.shape[1]),
        group_weight_sums=np.sum(weights, axis=0),
        training_metrics={},
    )
    fitted.training_metrics = _metrics.classification_metrics(
        y, probabilities=predict_partition_weighted_classification(fitted, X, membership)
    )
    return fitted


def _aligned_logits(model: object, design: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if isinstance(model, ConstantLogitModel):
        return model.decision_function(design)

    raw = model.decision_function(design)
    if raw.ndim == 1:
        raw = np.column_stack([-raw, raw])

    aligned = np.zeros((design.shape[0], classes.shape[0]), dtype=float)
    model_classes = getattr(model, "classes_", classes)
    for source_index, cls in enumerate(model_classes):
        target = int(np.where(classes == cls)[0][0])
        aligned[:, target] = raw[:, source_index]
    return aligned


def local_classification_logits(result: LocalClassificationResult, X, membership) -> np.ndarray:
    """Return local logit tensors with shape n_samples by n_groups by n_classes."""
    X = _as_2d_float(X, "X")
    membership = _validate_membership(X, membership)
    if membership.shape[1] != result.n_groups:
        raise ValueError("membership has a different number of groups from the fitted result.")

    local_inputs = _membership.local_weighted_inputs(X, membership)
    logits = np.zeros((X.shape[0], result.n_groups, result.classes.shape[0]), dtype=float)
    for j, model in enumerate(result.models):
        design = _poly.polynomial_design_matrix(local_inputs[:, j, :], degree=result.degree)
        logits[:, j, :] = _aligned_logits(model, design, result.classes)
    return logits


def predict_partition_weighted_classification(result: LocalClassificationResult, X, membership) -> np.ndarray:
    """Aggregate local logits and return multiclass probabilities."""
    logits = local_classification_logits(result, X, membership)
    weights = _membership.aggregation_weights(membership, f=result.f)
    final_logits = np.sum(weights[:, :, None] * logits, axis=1)
    return _metrics.softmax(final_logits)


if __name__ == "__main__":
    X_demo = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.1, 1.0]])
    u_demo = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]])
    y_demo = np.array([0.0, 0.1, 1.0, 1.1])
    model_demo = fit_partition_weighted_regression(X_demo, y_demo, u_demo, degree=1)
    print(np.round(predict_partition_weighted_regression(model_demo, X_demo, u_demo), 3))

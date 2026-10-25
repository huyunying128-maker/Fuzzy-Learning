"""Shared utilities for reviewer-response robustness and ablation experiments."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.utils import check_random_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"


def _load_core_module(file_name: str, alias: str):
    path = CORE_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load core module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_distance = _load_core_module("03_distance_functions.py", "reviewer_distance_functions")
_membership = _load_core_module("04_membership_functions.py", "reviewer_membership_functions")
_fuzzy = _load_core_module("07_fuzzy_partition.py", "reviewer_fuzzy_partition")
_metrics = _load_core_module("02_metrics.py", "reviewer_metrics")


def parse_csv_list(text: str, cast=str) -> List[Any]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one comma-separated value is required.")
    return [cast(value) for value in values]


def load_raw_dataset(
    input_path: Path,
    task: str,
    target: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], bool, Optional[LabelEncoder]]:
    """Load a numeric CSV or an NPZ with x/y arrays.

    NPZ inputs are treated as image/pixel-like arrays and kept on their original
    scale when already in [0, 1]; values in [0, 255] are divided by 255.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    is_pixel_data = input_path.suffix.lower() == ".npz"
    if is_pixel_data:
        payload = np.load(input_path, allow_pickle=True)
        if "x" not in payload or "y" not in payload:
            raise ValueError("NPZ input must contain arrays named 'x' and 'y'.")
        X = np.asarray(payload["x"])
        y = np.asarray(payload["y"]).reshape(-1)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X = X.astype(float)
        finite = X[np.isfinite(X)]
        if finite.size and np.nanmax(finite) > 1.0:
            X = X / 255.0
        feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    else:
        if target is None:
            raise ValueError("--target is required when --input is a CSV.")
        frame = pd.read_csv(input_path)
        if target not in frame.columns:
            lower_to_original = {str(c).strip().lower(): c for c in frame.columns}
            if target.strip().lower() not in lower_to_original:
                raise ValueError(
                    f"Target column {target!r} was not found. Available columns: {list(frame.columns)}"
                )
            target = lower_to_original[target.strip().lower()]
        y = frame[target].to_numpy()
        feature_frame = frame.drop(columns=[target]).copy()
        for column in feature_frame.columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
        X = feature_frame.to_numpy(dtype=float)
        feature_names = [str(c) for c in feature_frame.columns]

    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2-D and have the same number of rows as y.")
    if X.shape[0] < 10:
        raise ValueError("At least 10 observations are required for repeated split analysis.")
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'.")

    label_encoder = None
    if task == "regression":
        y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))

    if X.shape[0] < 10:
        raise ValueError("Fewer than 10 usable observations remain after cleaning.")
    return X, y, feature_names, is_pixel_data, label_encoder


def fit_preprocessor(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    is_pixel_data: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit imputation and, for tabular data, standardization on training rows only."""
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(X_train)
    eval_imputed = imputer.transform(X_eval)
    scaler = None
    if not is_pixel_data:
        scaler = StandardScaler()
        train_processed = scaler.fit_transform(train_imputed)
        eval_processed = scaler.transform(eval_imputed)
    else:
        train_processed = train_imputed
        eval_processed = eval_imputed
    train_processed = np.nan_to_num(train_processed, nan=0.0, posinf=0.0, neginf=0.0)
    eval_processed = np.nan_to_num(eval_processed, nan=0.0, posinf=0.0, neginf=0.0)
    return train_processed, eval_processed, {"imputer": imputer, "scaler": scaler}


def transform_with_preprocessor(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    is_pixel_data: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    train_processed, eval_processed, _ = fit_preprocessor(X_train, X_eval, is_pixel_data)
    return train_processed, eval_processed


def fit_partition(
    X_fit: np.ndarray,
    k: int,
    f: float,
    p: float,
    truncation_rule: str,
    tolerance: float,
    max_iter: int,
    random_state: int,
    partition_sample_size: int = 0,
):
    """Fit a fuzzy partition, optionally using a documented subsample for centroids."""
    X_fit = np.asarray(X_fit, dtype=float)
    if partition_sample_size and partition_sample_size > 0 and X_fit.shape[0] > partition_sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(X_fit.shape[0], size=int(partition_sample_size), replace=False)
        X_partition = X_fit[indices]
    else:
        X_partition = X_fit
    result = _fuzzy.fit_fuzzy_partition(
        X_partition,
        k=int(k),
        f=float(f),
        p=float(p),
        max_iter=int(max_iter),
        tolerance=float(tolerance),
        truncation_rule=str(truncation_rule).lower(),
        random_state=int(random_state),
    )
    return result, int(X_partition.shape[0])


def membership_for(X: np.ndarray, partition, f: float, p: float) -> np.ndarray:
    membership, _, _ = _fuzzy.predict_fuzzy_partition(
        np.asarray(X, dtype=float),
        partition.centroids,
        f=float(f),
        p=float(p),
    )
    return membership


def _polynomial_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    transformer = PolynomialFeatures(degree=int(degree), include_bias=True)
    return transformer.fit_transform(np.asarray(X, dtype=float))


def _make_estimator(task: str, regularization: float, random_state: int = 42):
    if task == "regression":
        return Ridge(alpha=float(regularization), fit_intercept=False)
    # C is inverse regularization strength for scikit-learn logistic regression.
    C = 1.0 / max(float(regularization), 1e-8)
    return LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        random_state=int(random_state),
    )


def _as_logits(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Return class-aligned logits even when a local expert lacks a class."""
    scores = np.asarray(model.decision_function(X), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if scores.ndim == 1:
        if len(classes) != 2:
            out = np.zeros((X.shape[0], n_classes), dtype=float)
            out[:, classes[0]] = 0.0
            return out
        out = np.zeros((X.shape[0], n_classes), dtype=float)
        out[:, classes[0]] = -scores / 2.0
        out[:, classes[1]] = scores / 2.0
        return out
    out = np.full((X.shape[0], n_classes), -20.0, dtype=float)
    for local_col, class_id in enumerate(classes):
        if 0 <= class_id < n_classes and local_col < scores.shape[1]:
            out[:, class_id] = scores[:, local_col]
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(logits, -700, 700))
    denominator = np.sum(exp_values, axis=1, keepdims=True)
    return exp_values / np.maximum(denominator, 1e-12)


def fit_global_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    task: str,
    degree: int,
    regularization: float,
    random_state: int = 42,
):
    """Fit a global polynomial ridge or logistic baseline."""
    estimator = _make_estimator(task, regularization, random_state)
    design_train = _polynomial_matrix(X_train, degree)
    design_eval = _polynomial_matrix(X_eval, degree)
    estimator.fit(design_train, y_train)
    if task == "regression":
        return estimator.predict(design_eval)
    logits = _as_logits(estimator, design_eval, int(np.max(y_train)) + 1)
    return _softmax(logits)


def fit_local_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    U_train: np.ndarray,
    X_eval: np.ndarray,
    U_eval: np.ndarray,
    task: str,
    degree: int,
    regularization: float,
    f: float,
    input_gating: bool = True,
    membership_weighted_fit: bool = True,
    aggregation: str = "fuzzy",
    random_state: int = 42,
):
    """Fit local experts and aggregate regression predictions or classification logits.

    Ablation controls:
      * input_gating: use u_ij * x_i as the expert input.
      * membership_weighted_fit: weight training observations by u_ij**f.
      * aggregation: 'fuzzy' combines all experts; 'hard' selects the dominant expert.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    U_train = np.asarray(U_train, dtype=float)
    U_eval = np.asarray(U_eval, dtype=float)
    n_groups = U_train.shape[1]
    if U_eval.shape[1] != n_groups:
        raise ValueError("Training and evaluation membership matrices must have the same number of columns.")

    n_classes = int(np.max(y_train)) + 1 if task == "classification" else 0
    group_outputs = []
    for j in range(n_groups):
        if input_gating:
            Xj_train = X_train * U_train[:, j:j + 1]
            Xj_eval = X_eval * U_eval[:, j:j + 1]
        else:
            Xj_train = X_train
            Xj_eval = X_eval

        design_train = _polynomial_matrix(Xj_train, degree)
        design_eval = _polynomial_matrix(Xj_eval, degree)
        estimator = _make_estimator(task, regularization, random_state + j)
        weights = np.maximum(U_train[:, j], 0.0) ** float(f) if membership_weighted_fit else None

        if task == "classification":
            # A local expert with one effective class is replaced by a global expert.
            effective_classes = np.unique(y_train if weights is None else y_train[weights > 1e-8])
            if effective_classes.size < 2:
                fallback = _make_estimator(task, regularization, random_state + j)
                fallback.fit(_polynomial_matrix(X_train, degree), y_train)
                logits = _as_logits(fallback, _polynomial_matrix(X_eval, degree), n_classes)
            else:
                try:
                    estimator.fit(design_train, y_train, sample_weight=weights)
                    logits = _as_logits(estimator, design_eval, n_classes)
                except (ValueError, FloatingPointError):
                    fallback = _make_estimator(task, regularization, random_state + j)
                    fallback.fit(_polynomial_matrix(X_train, degree), y_train)
                    logits = _as_logits(fallback, _polynomial_matrix(X_eval, degree), n_classes)
            group_outputs.append(logits)
        else:
            try:
                estimator.fit(design_train, y_train, sample_weight=weights)
                group_outputs.append(estimator.predict(design_eval))
            except (ValueError, FloatingPointError):
                fallback = _make_estimator(task, regularization, random_state + j)
                fallback.fit(_polynomial_matrix(X_train, degree), y_train)
                group_outputs.append(fallback.predict(_polynomial_matrix(X_eval, degree)))

    if task == "regression":
        stacked = np.column_stack(group_outputs)
    else:
        stacked = np.stack(group_outputs, axis=1)  # n_eval x n_groups x n_classes

    if aggregation == "hard":
        dominant = np.argmax(U_eval, axis=1)
        if task == "regression":
            return stacked[np.arange(X_eval.shape[0]), dominant]
        return _softmax(stacked[np.arange(X_eval.shape[0]), dominant, :])

    weights_eval = np.maximum(U_eval, 0.0) ** float(f)
    weights_eval = weights_eval / np.maximum(np.sum(weights_eval, axis=1, keepdims=True), 1e-12)
    if task == "regression":
        return np.sum(stacked * weights_eval, axis=1)
    aggregated_logits = np.sum(stacked * weights_eval[:, :, None], axis=1)
    return _softmax(aggregated_logits)


def make_feature_layer(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Construct z_i = [x_i, u_i, u_i1*x_i, ..., u_ik*x_i]."""
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    gated = (U[:, :, None] * X[:, None, :]).reshape(X.shape[0], -1)
    return np.concatenate([X, U, gated], axis=1)


def evaluate_predictions(task: str, y_true: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    if task == "regression":
        return _metrics.regression_metrics(y_true, prediction)
    probabilities = np.asarray(prediction, dtype=float)
    predicted = np.argmax(probabilities, axis=1)
    return _metrics.classification_metrics(y_true, probabilities=probabilities, predicted_labels=predicted)


def metric_name(task: str) -> str:
    return "mse" if task == "regression" else "cross_entropy"


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import t

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n_seeds": 0, "mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
    else:
        half = float("nan")
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half if np.isfinite(half) else np.nan,
        "ci95_high": mean + half if np.isfinite(half) else np.nan,
    }


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)

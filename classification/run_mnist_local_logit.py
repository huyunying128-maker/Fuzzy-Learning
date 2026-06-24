"""
MNIST local-logit experiments for partition-weighted classification.

The script evaluates raw multinomial logits, crisp local logits, and fuzzy local
logits with the same partition and truncation objects used in the regression
experiments.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.prepare_mnist_data import build_mnist_bundle, load_bundle, save_bundle
from feature_layer import aggregation_weights, membership_weighted_inputs
from fpwl_core import learn_partition, membership_for_new_points
from metrics_utils import classification_metrics, save_results

try:
    from classification_config import (
        DEFAULT_FEATURE_CAP,
        DEGREE_GRID,
        DISTANCE_ORDER_GRID,
        FUZZIFIER_GRID,
        LOGIT_MAX_ITER,
        MAX_ITER,
        MNIST_SPEC,
        N_CLASSES,
        QUICK_K_GRID,
        RANDOM_STATE,
        TOLERANCE,
        TRUNCATION_GRID,
        degree_name,
        parse_float_list,
        parse_int_list,
        parse_str_list,
    )
except ImportError:  # pragma: no cover
    from .classification_config import (
        DEFAULT_FEATURE_CAP,
        DEGREE_GRID,
        DISTANCE_ORDER_GRID,
        FUZZIFIER_GRID,
        LOGIT_MAX_ITER,
        MAX_ITER,
        MNIST_SPEC,
        N_CLASSES,
        QUICK_K_GRID,
        RANDOM_STATE,
        TOLERANCE,
        TRUNCATION_GRID,
        degree_name,
        parse_float_list,
        parse_int_list,
        parse_str_list,
    )


@dataclass
class LocalLogitResult:
    """Prediction output of a partition-weighted local-logit model."""

    probabilities: np.ndarray
    local_probabilities: list[np.ndarray]
    weights: np.ndarray


class PriorClassifier:
    """Constant probability classifier for small or single-class local groups."""

    def __init__(self, n_classes: int = N_CLASSES, alpha: float = 1.0e-3):
        self.n_classes = int(n_classes)
        self.alpha = float(alpha)
        self.probability_: np.ndarray | None = None

    def fit(self, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "PriorClassifier":
        labels = np.asarray(y, dtype=int).reshape(-1)
        if sample_weight is None:
            weights = np.ones(labels.shape[0], dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        counts = np.full(self.n_classes, self.alpha, dtype=float)
        for label, weight in zip(labels, weights):
            if 0 <= label < self.n_classes:
                counts[label] += max(float(weight), 0.0)
        self.probability_ = counts / counts.sum()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.probability_ is None:
            raise RuntimeError("The classifier has not been fitted.")
        return np.tile(self.probability_, (np.asarray(x).shape[0], 1))


def _select_feature_block(x_train: np.ndarray, x_test: np.ndarray, feature_cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Select a variance-ranked feature block for polynomial logit models."""
    if feature_cap <= 0 or feature_cap >= x_train.shape[1]:
        return x_train, x_test
    variance = np.var(x_train, axis=0)
    indices = np.argsort(variance)[::-1][:feature_cap]
    indices.sort()
    return x_train[:, indices], x_test[:, indices]


def make_logit_pipeline(degree: int, max_iter: int, random_state: int) -> Pipeline:
    """Create a polynomial multinomial-logit pipeline."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("poly", PolynomialFeatures(degree=int(degree), include_bias=False)),
            (
                "logit",
                LogisticRegression(
                    solver="saga",
                    penalty="l2",
                    C=1.0,
                    max_iter=int(max_iter),
                    multi_class="auto",
                    n_jobs=-1,
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def align_probability_columns(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Align classifier probability columns with the full class index."""
    aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
    for col, label in enumerate(classes.astype(int)):
        if 0 <= label < n_classes:
            aligned[:, label] = proba[:, col]
    row_sums = aligned.sum(axis=1, keepdims=True)
    zero_rows = row_sums.reshape(-1) <= 0
    if np.any(zero_rows):
        aligned[zero_rows] = 1.0 / n_classes
        row_sums = aligned.sum(axis=1, keepdims=True)
    return aligned / row_sums


def fit_raw_logit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    degree: int,
    feature_cap: int = DEFAULT_FEATURE_CAP,
    max_iter: int = LOGIT_MAX_ITER,
    random_state: int = RANDOM_STATE,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    """Fit a raw multinomial logit model and return test probabilities."""
    x_train_small, x_test_small = _select_feature_block(x_train, x_test, feature_cap)
    model = make_logit_pipeline(degree=degree, max_iter=max_iter, random_state=random_state)
    model.fit(x_train_small, y_train)
    proba = model.predict_proba(x_test_small)
    classes = model.named_steps["logit"].classes_
    return align_probability_columns(proba, classes, n_classes=n_classes)


def fit_local_logit_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_membership: np.ndarray,
    degree: int,
    partition: str,
    f: float,
    feature_cap: int = DEFAULT_FEATURE_CAP,
    max_iter: int = LOGIT_MAX_ITER,
    random_state: int = RANDOM_STATE,
    n_classes: int = N_CLASSES,
) -> list[Pipeline | PriorClassifier]:
    """Fit one local logit model for each membership group."""
    x_train_small, _ = _select_feature_block(x_train, x_train, feature_cap)
    local_inputs = membership_weighted_inputs(x_train_small, train_membership)
    k = train_membership.shape[1]
    models: list[Pipeline | PriorClassifier] = []

    for group_index in range(k):
        x_group = local_inputs[:, group_index, :]
        if partition == "crisp":
            sample_weight = train_membership[:, group_index]
        else:
            sample_weight = np.power(train_membership[:, group_index], f)

        active = sample_weight > 1.0e-8
        unique_labels = np.unique(y_train[active]) if np.any(active) else np.array([], dtype=int)
        if unique_labels.shape[0] < 2 or np.sum(active) < max(20, n_classes):
            fallback = PriorClassifier(n_classes=n_classes)
            fallback.fit(y_train, sample_weight=sample_weight)
            models.append(fallback)
            continue

        model = make_logit_pipeline(
            degree=degree,
            max_iter=max_iter,
            random_state=random_state + group_index,
        )
        model.fit(x_group, y_train, logit__sample_weight=sample_weight)
        models.append(model)

    return models


def predict_local_logit(
    models: list[Pipeline | PriorClassifier],
    x_train_reference: np.ndarray,
    x_test: np.ndarray,
    test_membership: np.ndarray,
    f: float,
    feature_cap: int = DEFAULT_FEATURE_CAP,
    n_classes: int = N_CLASSES,
) -> LocalLogitResult:
    """Aggregate local logit probabilities by normalized membership weights."""
    _, x_test_small = _select_feature_block(x_train_reference, x_test, feature_cap)
    local_inputs = membership_weighted_inputs(x_test_small, test_membership)
    weights = aggregation_weights(test_membership, f=f)
    combined = np.zeros((x_test.shape[0], n_classes), dtype=float)
    local_outputs: list[np.ndarray] = []

    for group_index, model in enumerate(models):
        x_group = local_inputs[:, group_index, :]
        proba = model.predict_proba(x_group)
        if isinstance(model, Pipeline):
            classes = model.named_steps["logit"].classes_
            proba = align_probability_columns(proba, classes, n_classes=n_classes)
        local_outputs.append(proba)
        combined += weights[:, [group_index]] * proba

    combined = np.clip(combined, 1.0e-12, 1.0)
    combined = combined / combined.sum(axis=1, keepdims=True)
    return LocalLogitResult(probabilities=combined, local_probabilities=local_outputs, weights=weights)


def run_raw_logit_rows(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    degrees: Iterable[int],
    feature_cap: int,
    max_iter: int,
    random_state: int,
) -> list[dict[str, object]]:
    """Evaluate raw logit baselines across local degrees."""
    rows: list[dict[str, object]] = []
    for degree in degrees:
        start = time.perf_counter()
        proba = fit_raw_logit(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            degree=degree,
            feature_cap=feature_cap,
            max_iter=max_iter,
            random_state=random_state,
        )
        elapsed = time.perf_counter() - start
        metrics = classification_metrics(y_test, proba=proba)
        rows.append(
            {
                "dataset": "mnist",
                "family": "raw_logit",
                "degree": int(degree),
                "degree_name": degree_name(int(degree)),
                "partition": "none",
                "truncation": "none",
                "k": np.nan,
                "f": np.nan,
                "p": np.nan,
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "n_iter": np.nan,
                "runtime_sec": elapsed,
            }
        )
    return rows


def run_local_logit_rows(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    degrees: Iterable[int],
    partitions: Iterable[str],
    truncations: Iterable[str],
    k_values: Iterable[int],
    f_values: Iterable[float],
    p_values: Iterable[float],
    feature_cap: int,
    partition_max_iter: int,
    logit_max_iter: int,
    tolerance: float,
    random_state: int,
) -> list[dict[str, object]]:
    """Evaluate crisp and fuzzy local-logit rows across partition settings."""
    rows: list[dict[str, object]] = []
    for partition in partitions:
        partition_key = partition.lower().strip()
        for degree in degrees:
            for truncation in truncations:
                for k in k_values:
                    current_f_values = [1.0] if partition_key == "crisp" else list(f_values)
                    current_p_values = list(p_values)
                    for f in current_f_values:
                        for p in current_p_values:
                            start = time.perf_counter()
                            partition_result = learn_partition(
                                x=x_train,
                                k=int(k),
                                partition=partition_key,
                                f=max(float(f), 1.000001),
                                p=float(p),
                                truncation=truncation,
                                tolerance=tolerance,
                                max_iter=partition_max_iter,
                                random_state=random_state,
                            )
                            test_membership, _, _ = membership_for_new_points(
                                x_new=x_test,
                                centroids=partition_result.centroids,
                                partition=partition_key,
                                f=max(float(f), 1.000001),
                                p=float(p),
                            )
                            models = fit_local_logit_models(
                                x_train=x_train,
                                y_train=y_train,
                                train_membership=partition_result.membership,
                                degree=int(degree),
                                partition=partition_key,
                                f=max(float(f), 1.000001),
                                feature_cap=feature_cap,
                                max_iter=logit_max_iter,
                                random_state=random_state,
                            )
                            prediction = predict_local_logit(
                                models=models,
                                x_train_reference=x_train,
                                x_test=x_test,
                                test_membership=test_membership,
                                f=max(float(f), 1.000001),
                                feature_cap=feature_cap,
                            )
                            elapsed = time.perf_counter() - start
                            metrics = classification_metrics(y_test, proba=prediction.probabilities)
                            rows.append(
                                {
                                    "dataset": "mnist",
                                    "family": f"{partition_key}_local",
                                    "degree": int(degree),
                                    "degree_name": degree_name(int(degree)),
                                    "partition": partition_key,
                                    "truncation": truncation,
                                    "k": int(k),
                                    "f": np.nan if partition_key == "crisp" else float(f),
                                    "p": float(p),
                                    "accuracy": metrics["accuracy"],
                                    "cross_entropy": metrics["cross_entropy"],
                                    "n_iter": int(partition_result.n_iter),
                                    "converged": bool(partition_result.converged),
                                    "runtime_sec": elapsed,
                                }
                            )
    return rows


def load_or_prepare_bundle(bundle_path: Path, source: str, n_samples: int | None, random_state: int):
    """Load an existing MNIST bundle or create it from the selected source."""
    if bundle_path.exists():
        return load_bundle(bundle_path)
    bundle = build_mnist_bundle(source=source, n_samples=n_samples, random_state=random_state)
    save_bundle(bundle, bundle_path)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MNIST raw-logit and local-logit experiments.")
    parser.add_argument("--bundle", default=str(MNIST_SPEC.default_bundle_path))
    parser.add_argument("--source", choices=["keras", "openml"], default="keras")
    parser.add_argument("--output", default=str(MNIST_SPEC.default_result_path / "mnist_local_logit_results.csv"))
    parser.add_argument("--degrees", default="1")
    parser.add_argument("--partitions", default="crisp,fuzzy")
    parser.add_argument("--truncations", default="hpd")
    parser.add_argument("--k-values", default="10")
    parser.add_argument("--f-values", default="1.3")
    parser.add_argument("--p-values", default="2.0")
    parser.add_argument("--feature-cap", type=int, default=DEFAULT_FEATURE_CAP)
    parser.add_argument("--partition-max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--logit-max-iter", type=int, default=LOGIT_MAX_ITER)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()

    bundle = load_or_prepare_bundle(
        bundle_path=Path(args.bundle),
        source=args.source,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )

    degrees = parse_int_list(args.degrees, DEGREE_GRID)
    partitions = parse_str_list(args.partitions, ["crisp", "fuzzy"])
    truncations = parse_str_list(args.truncations, TRUNCATION_GRID)
    k_values = parse_int_list(args.k_values, QUICK_K_GRID)
    f_values = parse_float_list(args.f_values, FUZZIFIER_GRID)
    p_values = parse_float_list(args.p_values, DISTANCE_ORDER_GRID)

    rows: list[dict[str, object]] = []
    if not args.skip_raw:
        rows.extend(
            run_raw_logit_rows(
                x_train=bundle.x_train,
                y_train=bundle.y_train,
                x_test=bundle.x_test,
                y_test=bundle.y_test,
                degrees=degrees,
                feature_cap=args.feature_cap,
                max_iter=args.logit_max_iter,
                random_state=args.random_state,
            )
        )
    if not args.skip_local:
        rows.extend(
            run_local_logit_rows(
                x_train=bundle.x_train,
                y_train=bundle.y_train,
                x_test=bundle.x_test,
                y_test=bundle.y_test,
                degrees=degrees,
                partitions=partitions,
                truncations=truncations,
                k_values=k_values,
                f_values=f_values,
                p_values=p_values,
                feature_cap=args.feature_cap,
                partition_max_iter=args.partition_max_iter,
                logit_max_iter=args.logit_max_iter,
                tolerance=args.tolerance,
                random_state=args.random_state,
            )
        )

    frame = save_results(rows, args.output)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

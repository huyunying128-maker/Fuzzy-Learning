"""
External MNIST classifiers with the partition-weighted feature layer.

The script compares original pixel features with the membership-based feature
layer built from a learned crisp or fuzzy partition.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.prepare_mnist_data import build_mnist_bundle, load_bundle, save_bundle
from feature_layer import partition_weighted_feature_layer
from fpwl_core import learn_partition, membership_for_new_points
from metrics_utils import classification_metrics, save_results

try:
    from classification_config import (
        CLASSIFIER_NAMES,
        DEFAULT_PCA_COMPONENTS,
        MAX_ITER,
        MNIST_SPEC,
        N_CLASSES,
        RANDOM_STATE,
        TOLERANCE,
        parse_str_list,
    )
except ImportError:  # pragma: no cover
    from .classification_config import (
        CLASSIFIER_NAMES,
        DEFAULT_PCA_COMPONENTS,
        MAX_ITER,
        MNIST_SPEC,
        N_CLASSES,
        RANDOM_STATE,
        TOLERANCE,
        parse_str_list,
    )


def align_probability_columns(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Align classifier probability columns with the complete class index."""
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


def make_classifier(name: str, random_state: int) -> object:
    """Create one external classifier by name."""
    key = name.lower().strip().replace("-", "_")
    if key == "ann":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1.0e-4,
            learning_rate_init=1.0e-3,
            max_iter=50,
            random_state=random_state,
        )
    if key == "adam_ann":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1.0e-4,
            learning_rate_init=5.0e-4,
            max_iter=60,
            random_state=random_state,
        )
    if key == "deep_learning":
        return MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            solver="adam",
            alpha=1.0e-5,
            learning_rate_init=5.0e-4,
            max_iter=80,
            random_state=random_state,
        )
    if key == "svm":
        return SVC(C=5.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state)
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover
            raise ImportError("xgboost is not installed.") from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown classifier name: {name}")


def make_pipeline(name: str, n_components: int, random_state: int) -> Pipeline:
    """Create a scaling, optional PCA, and classifier pipeline."""
    steps: list[tuple[str, object]] = [("scale", StandardScaler(with_mean=True, with_std=True))]
    if n_components > 0:
        steps.append(("pca", PCA(n_components=int(n_components), random_state=random_state)))
    steps.append(("model", make_classifier(name, random_state=random_state)))
    return Pipeline(steps=steps)


def predict_proba_aligned(model: Pipeline, x: np.ndarray, n_classes: int) -> np.ndarray:
    """Return aligned class probabilities from a fitted pipeline."""
    raw = model.predict_proba(x)
    classes = model.named_steps["model"].classes_
    return align_probability_columns(raw, classes, n_classes=n_classes)


def build_pw_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
    partition: str,
    f: float,
    p: float,
    truncation: str,
    tolerance: float,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Learn a partition and construct train-test partition-weighted features."""
    result = learn_partition(
        x=x_train,
        k=int(k),
        partition=partition,
        f=max(float(f), 1.000001),
        p=float(p),
        truncation=truncation,
        tolerance=tolerance,
        max_iter=max_iter,
        random_state=random_state,
    )
    test_membership, _, _ = membership_for_new_points(
        x_new=x_test,
        centroids=result.centroids,
        partition=partition,
        f=max(float(f), 1.000001),
        p=float(p),
    )
    x_train_pw = partition_weighted_feature_layer(
        x_train,
        result.membership,
        include_original=True,
        include_membership=True,
        include_weighted_inputs=True,
    )
    x_test_pw = partition_weighted_feature_layer(
        x_test,
        test_membership,
        include_original=True,
        include_membership=True,
        include_weighted_inputs=True,
    )
    return x_train_pw, x_test_pw, int(result.n_iter), bool(result.converged)


def evaluate_classifier(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
    random_state: int,
    n_classes: int = N_CLASSES,
) -> dict[str, float]:
    """Fit one classifier and return classification metrics with runtime."""
    start = time.perf_counter()
    model = make_pipeline(name=name, n_components=n_components, random_state=random_state)
    model.fit(x_train, y_train)
    proba = predict_proba_aligned(model, x_test, n_classes=n_classes)
    elapsed = time.perf_counter() - start
    metrics = classification_metrics(y_test, proba=proba)
    return {
        "accuracy": metrics["accuracy"],
        "cross_entropy": metrics["cross_entropy"],
        "runtime_sec": elapsed,
    }


def run_classifier_rows(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classifiers: Iterable[str],
    k: int,
    partition: str,
    f: float,
    p: float,
    truncation: str,
    n_components: int,
    tolerance: float,
    max_iter: int,
    random_state: int,
) -> list[dict[str, object]]:
    """Compare original and partition-weighted features for external classifiers."""
    rows: list[dict[str, object]] = []

    pw_start = time.perf_counter()
    x_train_pw, x_test_pw, partition_iter, converged = build_pw_features(
        x_train=x_train,
        x_test=x_test,
        k=k,
        partition=partition,
        f=f,
        p=p,
        truncation=truncation,
        tolerance=tolerance,
        max_iter=max_iter,
        random_state=random_state,
    )
    partition_runtime = time.perf_counter() - pw_start

    for name in classifiers:
        original = evaluate_classifier(
            name=name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            n_components=n_components,
            random_state=random_state,
        )
        weighted = evaluate_classifier(
            name=name,
            x_train=x_train_pw,
            y_train=y_train,
            x_test=x_test_pw,
            y_test=y_test,
            n_components=n_components,
            random_state=random_state,
        )
        rows.append(
            {
                "dataset": "mnist",
                "classifier": name,
                "partition": partition,
                "truncation": truncation,
                "k": int(k),
                "f": np.nan if partition == "crisp" else float(f),
                "p": float(p),
                "original_accuracy": original["accuracy"],
                "original_cross_entropy": original["cross_entropy"],
                "original_runtime_sec": original["runtime_sec"],
                "pw_accuracy": weighted["accuracy"],
                "pw_cross_entropy": weighted["cross_entropy"],
                "pw_runtime_sec": weighted["runtime_sec"],
                "accuracy_gain": weighted["accuracy"] - original["accuracy"],
                "cross_entropy_reduction": original["cross_entropy"] - weighted["cross_entropy"],
                "partition_iter": partition_iter,
                "partition_converged": converged,
                "partition_runtime_sec": partition_runtime,
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
    parser = argparse.ArgumentParser(description="Run MNIST external classifiers with PW features.")
    parser.add_argument("--bundle", default=str(MNIST_SPEC.default_bundle_path))
    parser.add_argument("--source", choices=["keras", "openml"], default="keras")
    parser.add_argument("--output", default=str(MNIST_SPEC.default_result_path / "mnist_pw_classifier_results.csv"))
    parser.add_argument("--classifiers", default="ann,random_forest")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--partition", choices=["crisp", "fuzzy"], default="fuzzy")
    parser.add_argument("--f", type=float, default=1.3)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--truncation", default="hpd")
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--partition-max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()

    bundle = load_or_prepare_bundle(
        bundle_path=Path(args.bundle),
        source=args.source,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )

    classifiers = parse_str_list(args.classifiers, CLASSIFIER_NAMES)
    rows = run_classifier_rows(
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_test=bundle.x_test,
        y_test=bundle.y_test,
        classifiers=classifiers,
        k=args.k,
        partition=args.partition,
        f=args.f,
        p=args.p,
        truncation=args.truncation,
        n_components=args.pca_components,
        tolerance=args.tolerance,
        max_iter=args.partition_max_iter,
        random_state=args.random_state,
    )
    frame = save_results(rows, args.output)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

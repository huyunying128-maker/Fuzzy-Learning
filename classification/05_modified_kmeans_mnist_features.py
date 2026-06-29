"""
MNIST feature layers for modified k-means and partition-weighted comparison.

This module builds the feature matrices used by external classifiers: the
original image input, a modified k-means feature input, and a fuzzy
partition-weighted feature input. The saved arrays keep the three input views in
one compressed file so that matched classifier comparisons use the same split.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
CLASSIFICATION_DIR = PROJECT_ROOT / "classification"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification" / "mnist"
EPSILON = 1e-12


def _load_core_module(file_name: str, alias: str):
    path = CORE_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_classification_module(file_name: str, alias: str):
    path = CLASSIFICATION_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_config = _load_core_module("01_config.py", "fpw_mnist_feature_config")
_metrics = _load_core_module("02_metrics.py", "fpw_mnist_feature_metrics")
_modified = _load_core_module("08_modified_kmeans.py", "fpw_mnist_modified_kmeans")
_fuzzy = _load_core_module("07_fuzzy_partition.py", "fpw_mnist_feature_fuzzy")
_feature_layer = _load_core_module("11_partition_feature_layer.py", "fpw_mnist_feature_layer")
_local = _load_classification_module("03_crisp_local_logit.py", "fpw_mnist_feature_helpers")


def build_feature_layer(X: np.ndarray, membership: np.ndarray, layer_mode: str = "compact") -> np.ndarray:
    """Build a membership feature layer with a selectable block structure."""
    layer_mode = str(layer_mode).lower()
    if layer_mode == "compact":
        return _feature_layer.build_partition_feature_layer(
            X,
            membership,
            include_original=True,
            include_membership=True,
            include_gated=False,
        )
    if layer_mode == "gated":
        return _feature_layer.build_partition_feature_layer(
            X,
            membership,
            include_original=True,
            include_membership=True,
            include_gated=True,
        )
    if layer_mode == "gated_only":
        return _feature_layer.build_partition_feature_layer(
            X,
            membership,
            include_original=False,
            include_membership=False,
            include_gated=True,
        )
    raise ValueError("layer_mode must be 'compact', 'gated', or 'gated_only'.")


def _probabilities_from_logit(model: LogisticRegression, X: np.ndarray, n_classes: int = 10) -> np.ndarray:
    probabilities = np.full((X.shape[0], n_classes), EPSILON, dtype=float)
    raw = model.predict_proba(X)
    for source_index, cls in enumerate(model.classes_):
        probabilities[:, int(cls)] = raw[:, source_index]
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    return probabilities


def score_feature_layer_with_logit(
    x_train_layer: np.ndarray,
    y_train: np.ndarray,
    x_valid_layer: np.ndarray,
    y_valid: np.ndarray,
    c_value: float = 1.0,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """Evaluate a feature layer using a simple multinomial logit readout."""
    model = LogisticRegression(C=float(c_value), max_iter=int(max_iter), solver="lbfgs")
    model.fit(x_train_layer, np.asarray(y_train, dtype=int))
    probabilities = _probabilities_from_logit(model, x_valid_layer, n_classes=10)
    return _metrics.classification_metrics(y_valid, probabilities=probabilities)


def choose_distance_fuzzy_degree(
    hard_result,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    f_values: Sequence[float],
    layer_mode: str,
    c_value: float,
    max_iter_logit: int,
) -> Tuple[object, List[Dict[str, float]]]:
    """Choose the post-partition fuzzy degree for the modified k-means layer."""
    records: List[Dict[str, float]] = []
    best_result = None
    best_loss = float("inf")

    for f in f_values:
        candidate = _modified.add_distance_based_fuzzy_layer(hard_result, f=float(f))
        train_layer = build_feature_layer(x_train, candidate.fuzzy_membership, layer_mode=layer_mode)
        valid_membership, _, _ = _modified.transform_with_modified_kmeans(x_valid, candidate, use_fuzzy=True)
        valid_layer = build_feature_layer(x_valid, valid_membership, layer_mode=layer_mode)
        metrics = score_feature_layer_with_logit(
            train_layer,
            y_train,
            valid_layer,
            y_valid,
            c_value=c_value,
            max_iter=max_iter_logit,
        )
        records.append(
            {
                "branch": "modified_kmeans_f",
                "k": int(candidate.k),
                "p": float(candidate.p),
                "f": float(f),
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_cross_entropy": float(metrics["cross_entropy"]),
            }
        )
        if metrics["cross_entropy"] < best_loss:
            best_loss = float(metrics["cross_entropy"])
            best_result = candidate

    if best_result is None:
        raise RuntimeError("No fuzzy-degree candidate was selected.")
    best_result.f_search_records = records
    return best_result, records


def run_mnist_feature_layers(
    split_path: Path,
    output_npz: Path = OUTPUT_DIR / "05_mnist_feature_layers.npz",
    summary_output: Path = OUTPUT_DIR / "05_mnist_feature_layers_summary.csv",
    k_values_text: Optional[str] = "10",
    p_values_text: Optional[str] = "2.00",
    f_values_text: Optional[str] = "1.30",
    pw_k: int = 10,
    pw_p: float = 2.00,
    pw_f: float = 1.30,
    layer_mode: str = "compact",
    max_candidates: Optional[int] = None,
    max_iter_partition: int = 300,
    tolerance: float = 1e-6,
    random_state: int = 42,
    c_value: float = 1.0,
    max_iter_logit: int = 1000,
) -> pd.DataFrame:
    """Create and save original, modified k-means, and PW feature layers."""
    start = time.perf_counter()
    split = _local.load_split_npz(split_path)
    x_train = np.asarray(split["x_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=int)
    x_valid = np.asarray(split["x_valid"], dtype=float)
    y_valid = np.asarray(split["y_valid"], dtype=int)
    x_test = np.asarray(split["x_test"], dtype=float)
    y_test = np.asarray(split["y_test"], dtype=int)

    default_k, default_p, default_f = _local.default_partition_grids(x_train.shape[0])
    k_values = _local.parse_numeric_grid(k_values_text, default_k, cast=int)
    p_values = _local.parse_numeric_grid(p_values_text, default_p, cast=float)
    f_values = _local.parse_numeric_grid(f_values_text, default_f, cast=float)

    hard_result = _modified.search_hard_kp_reference(
        x_train,
        k_values=k_values,
        p_values=p_values,
        max_iter=max_iter_partition,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
    )
    km_result, f_records = choose_distance_fuzzy_degree(
        hard_result,
        x_train,
        y_train,
        x_valid,
        y_valid,
        f_values=f_values,
        layer_mode=layer_mode,
        c_value=c_value,
        max_iter_logit=max_iter_logit,
    )

    km_hard_train = build_feature_layer(x_train, hard_result.hard_membership, layer_mode=layer_mode)
    km_hard_test_membership, _, _ = _modified.transform_with_modified_kmeans(x_test, hard_result, use_fuzzy=False)
    km_hard_test = build_feature_layer(x_test, km_hard_test_membership, layer_mode=layer_mode)

    km_train = build_feature_layer(x_train, km_result.fuzzy_membership, layer_mode=layer_mode)
    km_test_membership, _, _ = _modified.transform_with_modified_kmeans(x_test, km_result, use_fuzzy=True)
    km_test = build_feature_layer(x_test, km_test_membership, layer_mode=layer_mode)

    pw_partition = _fuzzy.fit_fuzzy_partition(
        x_train,
        k=int(pw_k),
        f=float(pw_f),
        p=float(pw_p),
        max_iter=max_iter_partition,
        tolerance=tolerance,
        truncation_rule="hpd",
        random_state=random_state,
    )
    pw_test_membership, _, _ = _fuzzy.predict_fuzzy_partition(
        x_test,
        pw_partition.centroids,
        f=float(pw_f),
        p=float(pw_p),
    )
    pw_train = build_feature_layer(x_train, pw_partition.membership, layer_mode=layer_mode)
    pw_test = build_feature_layer(x_test, pw_test_membership, layer_mode=layer_mode)

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        km_hard_train=km_hard_train,
        km_hard_test=km_hard_test,
        km_train=km_train,
        km_test=km_test,
        pw_train=pw_train,
        pw_test=pw_test,
        layer_mode=np.array(layer_mode),
        km_k=np.array(km_result.k),
        km_p=np.array(km_result.p),
        km_f=np.array(km_result.f),
        pw_k=np.array(pw_k),
        pw_p=np.array(pw_p),
        pw_f=np.array(pw_f),
    )

    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
    summary_records = [
        {
            "dataset": "mnist",
            "branch": "modified_kmeans_hard",
            "layer_mode": layer_mode,
            "k": int(hard_result.k),
            "p": float(hard_result.p),
            "f": np.nan,
            "n_features": int(km_hard_train.shape[1]),
            "n_iter": int(hard_result.n_iter),
            "runtime_sec": elapsed,
        },
        {
            "dataset": "mnist",
            "branch": "modified_kmeans_fuzzy",
            "layer_mode": layer_mode,
            "k": int(km_result.k),
            "p": float(km_result.p),
            "f": float(km_result.f),
            "n_features": int(km_train.shape[1]),
            "n_iter": int(km_result.n_iter),
            "runtime_sec": elapsed,
        },
        {
            "dataset": "mnist",
            "branch": "partition_weighted_fuzzy",
            "layer_mode": layer_mode,
            "k": int(pw_k),
            "p": float(pw_p),
            "f": float(pw_f),
            "n_features": int(pw_train.shape[1]),
            "n_iter": int(pw_partition.n_iter),
            "runtime_sec": elapsed,
        },
    ]
    summary = pd.DataFrame(summary_records)
    summary_output = Path(summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)

    if f_records:
        pd.DataFrame(f_records).to_csv(summary_output.with_name(summary_output.stem + "_f_search.csv"), index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MNIST external-classifier feature layers.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, default=OUTPUT_DIR / "05_mnist_feature_layers.npz")
    parser.add_argument("--summary-output", type=Path, default=OUTPUT_DIR / "05_mnist_feature_layers_summary.csv")
    parser.add_argument("--k-values", default="10")
    parser.add_argument("--p-values", default="2.00")
    parser.add_argument("--f-values", default="1.30")
    parser.add_argument("--pw-k", type=int, default=10)
    parser.add_argument("--pw-p", type=float, default=2.00)
    parser.add_argument("--pw-f", type=float, default=1.30)
    parser.add_argument("--layer-mode", default="compact", choices=["compact", "gated", "gated_only"])
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-iter-partition", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter-logit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_mnist_feature_layers(
        split_path=args.split,
        output_npz=args.output_npz,
        summary_output=args.summary_output,
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        f_values_text=args.f_values,
        pw_k=args.pw_k,
        pw_p=args.pw_p,
        pw_f=args.pw_f,
        layer_mode=args.layer_mode,
        max_candidates=args.max_candidates,
        max_iter_partition=args.max_iter_partition,
        tolerance=args.tolerance,
        random_state=args.random_state,
        c_value=args.c_value,
        max_iter_logit=args.max_iter_logit,
    )
    print(f"Saved MNIST feature-layer summary with {summary.shape[0]} rows")


if __name__ == "__main__":
    main()

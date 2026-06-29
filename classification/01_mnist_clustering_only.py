"""
MNIST clustering-only classification references.

This module evaluates crisp and fuzzy partitions before local logit models are
added. Cluster labels are converted to digit labels by a training-set majority
mapping, and the resulting rows report accuracy, cross entropy, iteration count,
and runtime for the partition-only classification references.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
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


_config = _load_core_module("01_config.py", "fpw_cls_config")
_metrics = _load_core_module("02_metrics.py", "fpw_cls_metrics")
_crisp = _load_core_module("06_crisp_partition.py", "fpw_cls_crisp_partition")
_fuzzy = _load_core_module("07_fuzzy_partition.py", "fpw_cls_fuzzy_partition")


def load_split_npz(split_path: Path) -> Dict[str, np.ndarray]:
    """Load a train-validation-test split saved by the dataset utilities."""
    data = np.load(Path(split_path), allow_pickle=True)
    required = ("x_train", "y_train", "x_test", "y_test")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing arrays in split file: {missing}")
    split = {name: data[name] for name in data.files}
    if "x_valid" not in split or "y_valid" not in split:
        split["x_valid"] = split["x_train"]
        split["y_valid"] = split["y_train"]
    return split


def parse_numeric_grid(text: Optional[str], default_values: Sequence[float], cast=float) -> List:
    """Parse a comma-separated grid or return the default grid."""
    if text is None or str(text).strip() == "":
        return [cast(value) for value in default_values]
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast(float(part) if cast is int else part))
    if not values:
        raise ValueError("The parsed grid is empty.")
    return values


def _default_grids(n_samples: int) -> Tuple[List[int], List[float], List[float]]:
    grid = _config.make_partition_grid(n_samples)
    return grid.k_values, grid.p_values, grid.f_values


def majority_label_mapping(cluster_labels, y_true, n_clusters: int, n_classes: int = 10) -> np.ndarray:
    """Map each cluster to the most frequent class observed in that cluster."""
    cluster_labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    mapping = np.zeros(n_clusters, dtype=int)
    global_counts = np.bincount(y_true, minlength=n_classes)
    global_label = int(np.argmax(global_counts))

    for j in range(n_clusters):
        mask = cluster_labels == j
        if np.any(mask):
            counts = np.bincount(y_true[mask], minlength=n_classes)
            mapping[j] = int(np.argmax(counts))
        else:
            mapping[j] = global_label
    return mapping


def probability_from_cluster_labels(cluster_labels, mapping: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Convert mapped cluster labels into a smoothed probability matrix."""
    cluster_labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    labels = mapping[cluster_labels]
    probabilities = np.full((labels.shape[0], n_classes), EPSILON, dtype=float)
    probabilities[np.arange(labels.shape[0]), labels] = 1.0 - EPSILON * (n_classes - 1)
    return probabilities


def _evaluate_crisp_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    k: int,
    p: float,
    truncation: str,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> Tuple[float, Dict[str, float], object]:
    result = _crisp.fit_crisp_partition(
        x_train,
        k=int(k),
        p=float(p),
        max_iter=max_iter,
        tolerance=tolerance,
        truncation_rule=truncation,
        random_state=random_state,
    )
    mapping = majority_label_mapping(result.labels, y_train, n_clusters=int(k), n_classes=10)
    valid_membership, valid_labels, _ = _crisp.predict_crisp_partition(x_valid, result.centroids, p=float(p))
    probabilities = probability_from_cluster_labels(valid_labels, mapping, n_classes=10)
    metrics = _metrics.classification_metrics(y_valid, probabilities=probabilities)
    return float(metrics["cross_entropy"]), metrics, result


def _evaluate_fuzzy_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    k: int,
    p: float,
    f: float,
    truncation: str,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> Tuple[float, Dict[str, float], object]:
    result = _fuzzy.fit_fuzzy_partition(
        x_train,
        k=int(k),
        f=float(f),
        p=float(p),
        max_iter=max_iter,
        tolerance=tolerance,
        truncation_rule=truncation,
        random_state=random_state,
    )
    mapping = majority_label_mapping(result.labels, y_train, n_clusters=int(k), n_classes=10)
    valid_membership, valid_labels, _ = _fuzzy.predict_fuzzy_partition(x_valid, result.centroids, f=float(f), p=float(p))
    probabilities = probability_from_cluster_labels(valid_labels, mapping, n_classes=10)
    metrics = _metrics.classification_metrics(y_valid, probabilities=probabilities)
    return float(metrics["cross_entropy"]), metrics, result


def _candidate_iterator_crisp(k_values, p_values, max_candidates: Optional[int]):
    count = 0
    for k in k_values:
        for p in p_values:
            yield int(k), float(p)
            count += 1
            if max_candidates is not None and count >= max_candidates:
                return


def _candidate_iterator_fuzzy(k_values, p_values, f_values, max_candidates: Optional[int]):
    count = 0
    for k in k_values:
        for p in p_values:
            for f in f_values:
                yield int(k), float(p), float(f)
                count += 1
                if max_candidates is not None and count >= max_candidates:
                    return


def run_mnist_clustering_only(
    split_path: Path,
    output_path: Path = OUTPUT_DIR / "01_mnist_clustering_only.csv",
    k_values_text: Optional[str] = "10",
    p_values_text: Optional[str] = "2.00",
    f_values_text: Optional[str] = "1.30",
    truncation: str = "hpd",
    max_iter: int = 300,
    tolerance: float = 1e-6,
    random_state: int = 42,
    max_candidates: Optional[int] = None,
    search_record_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fit crisp and fuzzy clustering-only references for MNIST."""
    split = load_split_npz(split_path)
    x_train = np.asarray(split["x_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=int)
    x_valid = np.asarray(split["x_valid"], dtype=float)
    y_valid = np.asarray(split["y_valid"], dtype=int)
    x_test = np.asarray(split["x_test"], dtype=float)
    y_test = np.asarray(split["y_test"], dtype=int)

    default_k, default_p, default_f = _default_grids(x_train.shape[0])
    k_values = parse_numeric_grid(k_values_text, default_k, cast=int)
    p_values = parse_numeric_grid(p_values_text, default_p, cast=float)
    f_values = parse_numeric_grid(f_values_text, default_f, cast=float)

    records: List[Dict[str, float]] = []
    search_records: List[Dict[str, float]] = []

    start = time.perf_counter()
    best_crisp = None
    best_crisp_loss = float("inf")
    for k, p in _candidate_iterator_crisp(k_values, p_values, max_candidates):
        loss, valid_metrics, result = _evaluate_crisp_candidate(
            x_train, y_train, x_valid, y_valid, k, p, truncation, max_iter, tolerance, random_state
        )
        search_records.append({
            "family": "crisp_clustering_only",
            "k": k,
            "p": p,
            "f": np.nan,
            "validation_cross_entropy": valid_metrics["cross_entropy"],
            "validation_accuracy": valid_metrics["accuracy"],
            "n_iter": result.n_iter,
        })
        if loss < best_crisp_loss:
            best_crisp_loss = loss
            best_crisp = (k, p, result)
    if best_crisp is None:
        raise ValueError("No crisp clustering candidate was evaluated.")

    k, p, crisp_result = best_crisp
    crisp_mapping = majority_label_mapping(crisp_result.labels, y_train, n_clusters=k, n_classes=10)
    _, test_labels, _ = _crisp.predict_crisp_partition(x_test, crisp_result.centroids, p=p)
    test_prob = probability_from_cluster_labels(test_labels, crisp_mapping, n_classes=10)
    crisp_test_metrics = _metrics.classification_metrics(y_test, probabilities=test_prob)
    records.append({
        "dataset": "mnist",
        "task": "classification",
        "method_family": "clustering_only",
        "method_name": "Crisp clustering only",
        "degree": np.nan,
        "k": k,
        "f": np.nan,
        "p": p,
        "truncation": truncation,
        "test_accuracy": crisp_test_metrics["accuracy"],
        "test_cross_entropy": crisp_test_metrics["cross_entropy"],
        "n_iter": crisp_result.n_iter,
        "runtime_sec": _metrics.summarize_elapsed_seconds(start, time.perf_counter()),
    })

    start = time.perf_counter()
    best_fuzzy = None
    best_fuzzy_loss = float("inf")
    for k, p, f in _candidate_iterator_fuzzy(k_values, p_values, f_values, max_candidates):
        loss, valid_metrics, result = _evaluate_fuzzy_candidate(
            x_train, y_train, x_valid, y_valid, k, p, f, truncation, max_iter, tolerance, random_state
        )
        search_records.append({
            "family": "fuzzy_clustering_only",
            "k": k,
            "p": p,
            "f": f,
            "validation_cross_entropy": valid_metrics["cross_entropy"],
            "validation_accuracy": valid_metrics["accuracy"],
            "n_iter": result.n_iter,
        })
        if loss < best_fuzzy_loss:
            best_fuzzy_loss = loss
            best_fuzzy = (k, p, f, result)
    if best_fuzzy is None:
        raise ValueError("No fuzzy clustering candidate was evaluated.")

    k, p, f, fuzzy_result = best_fuzzy
    fuzzy_mapping = majority_label_mapping(fuzzy_result.labels, y_train, n_clusters=k, n_classes=10)
    _, test_labels, _ = _fuzzy.predict_fuzzy_partition(x_test, fuzzy_result.centroids, f=f, p=p)
    test_prob = probability_from_cluster_labels(test_labels, fuzzy_mapping, n_classes=10)
    fuzzy_test_metrics = _metrics.classification_metrics(y_test, probabilities=test_prob)
    records.append({
        "dataset": "mnist",
        "task": "classification",
        "method_family": "clustering_only",
        "method_name": "Fuzzy clustering only",
        "degree": np.nan,
        "k": k,
        "f": f,
        "p": p,
        "truncation": truncation,
        "test_accuracy": fuzzy_test_metrics["accuracy"],
        "test_cross_entropy": fuzzy_test_metrics["cross_entropy"],
        "n_iter": fuzzy_result.n_iter,
        "runtime_sec": _metrics.summarize_elapsed_seconds(start, time.perf_counter()),
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output_path, index=False)

    if search_record_path is not None:
        search_path = Path(search_record_path)
        search_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(search_records).to_csv(search_path, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MNIST clustering-only references.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "01_mnist_clustering_only.csv")
    parser.add_argument("--k-values", default="10")
    parser.add_argument("--p-values", default="2.00")
    parser.add_argument("--f-values", default="1.30")
    parser.add_argument("--truncation", default="hpd")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--search-records", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_mnist_clustering_only(
        split_path=args.split,
        output_path=args.output,
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        f_values_text=args.f_values,
        truncation=args.truncation,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        random_state=args.random_state,
        max_candidates=args.max_candidates,
        search_record_path=args.search_records,
    )
    print(f"Saved MNIST clustering-only table with {table.shape[0]} rows")


if __name__ == "__main__":
    main()

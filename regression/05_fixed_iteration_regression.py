"""
Fixed-iteration fuzzy regression reference.

This script fits a fuzzy partition for a fixed number of updates and then uses
partition-only group means for regression. The output records the held-out MSE,
MAE, R-squared value, iteration count, and runtime in the same format as the
main regression comparison tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"


def _load_core_module(file_name: str, alias: str):
    path = CORE_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_config = _load_core_module("01_config.py", "fpw_config")
_metrics = _load_core_module("02_metrics.py", "fpw_metrics")
_membership = _load_core_module("04_membership_functions.py", "fpw_membership_functions")
_fuzzy = _load_core_module("07_fuzzy_partition.py", "fpw_fuzzy_partition")


EPSILON = 1e-12


@dataclass(frozen=True)
class FixedIterationCandidate:
    """One evaluated fixed-iteration fuzzy partition reference."""

    k: int
    f: float
    p: float
    fixed_iterations: int
    validation_mse: float
    validation_mae: float
    validation_r2: float
    n_iter: int
    runtime_sec: float


def load_split_npz(split_path: Path) -> Dict[str, np.ndarray]:
    """Load split arrays produced by the dataset preparation workflow."""
    data = np.load(Path(split_path), allow_pickle=True)
    required = ("x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing arrays in split file: {missing}")
    return {name: data[name] for name in data.files}


def combine_train_valid(split: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the full non-test training section."""
    x_full = np.vstack([split["x_train"], split["x_valid"]])
    y_full = np.concatenate([split["y_train"], split["y_valid"]])
    return x_full, y_full


def parse_float_values(text: Optional[str], default_values: Sequence[float]) -> List[float]:
    """Parse a comma-separated float grid or return the supplied default grid."""
    if text is None or text.strip() == "":
        return [float(value) for value in default_values]
    return [round(float(part.strip()), 2) for part in text.split(",") if part.strip()]


def parse_int_values(text: Optional[str], default_values: Sequence[int]) -> List[int]:
    """Parse a comma-separated integer grid or return the supplied default grid."""
    if text is None or text.strip() == "":
        return [int(value) for value in default_values]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_iteration_values(text: str) -> List[int]:
    """Parse fixed iteration counts."""
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("Fixed iteration counts must be positive integers.")
    return values


def group_target_means(y: np.ndarray, membership: np.ndarray, f: float) -> np.ndarray:
    """Estimate one target mean for each fuzzy local group."""
    y = np.asarray(y, dtype=float).reshape(-1)
    membership = np.asarray(membership, dtype=float)
    if membership.ndim != 2:
        raise ValueError("membership must be two-dimensional.")
    if membership.shape[0] != y.shape[0]:
        raise ValueError("membership and y must have the same number of rows.")

    weights = np.maximum(membership, 0.0) ** float(f)
    denominators = np.sum(weights, axis=0)
    global_mean = float(np.mean(y))
    means = np.full(membership.shape[1], global_mean, dtype=float)
    for j in range(membership.shape[1]):
        if denominators[j] > EPSILON:
            means[j] = float(weights[:, j] @ y / denominators[j])
    return means


def predict_from_group_means(membership: np.ndarray, group_means: np.ndarray, f: float) -> np.ndarray:
    """Predict by aggregating group means with normalized fuzzy weights."""
    weights = _membership.aggregation_weights(membership, f=float(f))
    return weights @ np.asarray(group_means, dtype=float).reshape(-1)


def _candidate_grid(
    n_samples: int,
    k_values_text: Optional[str],
    p_values_text: Optional[str],
    f_values_text: Optional[str],
) -> Tuple[List[int], List[float], List[float]]:
    grid = _config.make_partition_grid(n_samples)
    k_values = parse_int_values(k_values_text, grid.k_values)
    p_values = parse_float_values(p_values_text, grid.p_values)
    f_values = parse_float_values(f_values_text, grid.f_values)
    k_values = [k for k in k_values if 2 <= k <= n_samples]
    p_values = [p for p in p_values if p >= 1.0]
    f_values = [f for f in f_values if f >= 1.0]
    if not k_values or not p_values or not f_values:
        raise ValueError("The partition grid must contain feasible k, p, and f values.")
    return k_values, p_values, f_values


def fit_fixed_fuzzy_partition(
    X: np.ndarray,
    k: int,
    f: float,
    p: float,
    fixed_iterations: int,
    random_state: int,
):
    """Fit the fuzzy partition with an iteration limit used as the cutoff."""
    return _fuzzy.fit_fuzzy_partition(
        X,
        k=int(k),
        f=float(f),
        p=float(p),
        max_iter=int(fixed_iterations),
        tolerance=0.0,
        truncation_rule="dtd",
        random_state=int(random_state),
    )


def evaluate_fixed_iteration_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    k: int,
    f: float,
    p: float,
    fixed_iterations: int,
    random_state: int,
) -> FixedIterationCandidate:
    """Evaluate one fixed-iteration fuzzy partition on the validation section."""
    start = time.perf_counter()
    partition = fit_fixed_fuzzy_partition(
        x_train,
        k=k,
        f=f,
        p=p,
        fixed_iterations=fixed_iterations,
        random_state=random_state,
    )
    means = group_target_means(y_train, partition.membership, f=f)
    valid_membership, _, _ = _fuzzy.predict_fuzzy_partition(x_valid, partition.centroids, f=f, p=p)
    y_pred = predict_from_group_means(valid_membership, means, f=f)
    metrics = _metrics.regression_metrics(y_valid, y_pred)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
    return FixedIterationCandidate(
        k=int(k),
        f=float(f),
        p=float(p),
        fixed_iterations=int(fixed_iterations),
        validation_mse=float(metrics["mse"]),
        validation_mae=float(metrics["mae"]),
        validation_r2=float(metrics["r2"]),
        n_iter=int(partition.n_iter),
        runtime_sec=elapsed,
    )


def refit_and_test_fixed_iteration(
    split: Dict[str, np.ndarray],
    selected: FixedIterationCandidate,
    random_state: int,
) -> Dict[str, float]:
    """Refit the selected fixed-iteration partition and evaluate the test split."""
    x_full, y_full = combine_train_valid(split)
    start = time.perf_counter()
    partition = fit_fixed_fuzzy_partition(
        x_full,
        k=selected.k,
        f=selected.f,
        p=selected.p,
        fixed_iterations=selected.fixed_iterations,
        random_state=random_state,
    )
    means = group_target_means(y_full, partition.membership, f=selected.f)
    test_membership, _, _ = _fuzzy.predict_fuzzy_partition(
        split["x_test"],
        partition.centroids,
        f=selected.f,
        p=selected.p,
    )
    y_pred = predict_from_group_means(test_membership, means, f=selected.f)
    metrics = _metrics.regression_metrics(split["y_test"], y_pred)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())

    row: Dict[str, float] = {
        "k": selected.k,
        "f": selected.f,
        "p": selected.p,
        "fixed_iterations": selected.fixed_iterations,
        "validation_mse": selected.validation_mse,
        "validation_mae": selected.validation_mae,
        "validation_r2": selected.validation_r2,
        "validation_runtime_sec": selected.runtime_sec,
        "test_runtime_sec": elapsed,
        "n_iter": int(partition.n_iter),
        "converged": bool(partition.converged),
        "partition_objective": float(partition.objective),
    }
    row.update({f"test_{key}": value for key, value in metrics.items()})
    return row


def run_fixed_iteration_regression(
    split_path: Path,
    dataset_name: str,
    output_path: Path,
    fixed_iterations: Sequence[int],
    k_values_text: Optional[str] = None,
    p_values_text: Optional[str] = None,
    f_values_text: Optional[str] = None,
    random_state: int = _config.RANDOM_STATE,
    max_candidates: Optional[int] = None,
    search_record_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Search fixed-iteration fuzzy references and save the selected test rows."""
    split = load_split_npz(split_path)
    k_values, p_values, f_values = _candidate_grid(
        split["x_train"].shape[0],
        k_values_text,
        p_values_text,
        f_values_text,
    )
    candidate_tuples = list(itertools.product(k_values, f_values, p_values, fixed_iterations))
    if max_candidates is not None:
        candidate_tuples = candidate_tuples[: int(max_candidates)]

    best: Optional[FixedIterationCandidate] = None
    records: List[Dict[str, float]] = []
    for run_index, (k, f, p, n_iter) in enumerate(candidate_tuples):
        candidate = evaluate_fixed_iteration_candidate(
            split["x_train"],
            split["y_train"],
            split["x_valid"],
            split["y_valid"],
            k=int(k),
            f=float(f),
            p=float(p),
            fixed_iterations=int(n_iter),
            random_state=int(random_state + run_index),
        )
        record = candidate.__dict__.copy()
        record.update({"dataset": dataset_name, "method_family": "fixed_iteration"})
        records.append(record)
        if best is None or candidate.validation_mse < best.validation_mse:
            best = candidate

    if best is None:
        raise RuntimeError("No fixed-iteration candidate was evaluated.")

    row = refit_and_test_fixed_iteration(split, selected=best, random_state=random_state)
    row.update(
        {
            "dataset": dataset_name,
            "task": "regression",
            "method_family": "fixed_iteration",
            "method_name": "Fixed-iteration fuzzy clustering",
            "degree": np.nan,
            "truncation": "fixed_iteration",
            "n_train": int(split["x_train"].shape[0]),
            "n_valid": int(split["x_valid"].shape[0]),
            "n_test": int(split["x_test"].shape[0]),
            "n_features": int(split["x_train"].shape[1]),
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row])
    table.to_csv(output_path, index=False)

    if search_record_path is not None:
        search_record_path = Path(search_record_path)
        search_record_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(search_record_path, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the fixed-iteration fuzzy regression reference.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/regression/fixed_iteration_regression.csv"))
    parser.add_argument("--search-records", type=Path, default=None)
    parser.add_argument("--fixed-iterations", default="8,14")
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--f-values", default=None)
    parser.add_argument("--random-state", type=int, default=_config.RANDOM_STATE)
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_fixed_iteration_regression(
        split_path=args.split,
        dataset_name=args.dataset_name,
        output_path=args.output,
        fixed_iterations=parse_iteration_values(args.fixed_iterations),
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        f_values_text=args.f_values,
        random_state=args.random_state,
        max_candidates=args.max_candidates,
        search_record_path=args.search_records,
    )
    print(f"Saved {table.shape[0]} fixed-iteration regression row to {args.output}")


if __name__ == "__main__":
    main()

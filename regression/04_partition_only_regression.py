"""
Partition-only regression references.

The partition-only rows use the learned crisp or fuzzy partition without fitting
local polynomial surfaces. Each group stores a target mean, and prediction is
formed from hard group membership or fuzzy aggregation weights.
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
_crisp = _load_core_module("06_crisp_partition.py", "fpw_crisp_partition")
_fuzzy = _load_core_module("07_fuzzy_partition.py", "fpw_fuzzy_partition")


EPSILON = 1e-12


@dataclass(frozen=True)
class PartitionOnlyCandidate:
    """One evaluated partition-only regression configuration."""

    partition_type: str
    truncation: str
    k: int
    f: float
    p: float
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


def group_target_means(y: np.ndarray, membership: np.ndarray, f: float = 1.0) -> np.ndarray:
    """Estimate one target mean for each local group."""
    y = np.asarray(y, dtype=float).reshape(-1)
    membership = np.asarray(membership, dtype=float)
    if membership.ndim != 2:
        raise ValueError("membership must be a two-dimensional array.")
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


def predict_from_group_means(membership: np.ndarray, group_means: np.ndarray, f: float = 1.0) -> np.ndarray:
    """Predict by aggregating group-level target means."""
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


def evaluate_crisp_partition_only(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    truncation: str,
    k: int,
    p: float,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> PartitionOnlyCandidate:
    """Evaluate one crisp partition-only candidate on the validation section."""
    start = time.perf_counter()
    partition = _crisp.fit_crisp_partition(
        x_train,
        k=int(k),
        p=float(p),
        max_iter=int(max_iter),
        tolerance=float(tolerance),
        truncation_rule=truncation,
        random_state=int(random_state),
    )
    means = group_target_means(y_train, partition.membership, f=1.0)
    valid_membership, _, _ = _crisp.predict_crisp_partition(x_valid, partition.centroids, p=float(p))
    y_pred = predict_from_group_means(valid_membership, means, f=1.0)
    metrics = _metrics.regression_metrics(y_valid, y_pred)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
    return PartitionOnlyCandidate(
        partition_type="crisp",
        truncation=truncation,
        k=int(k),
        f=1.0,
        p=float(p),
        validation_mse=float(metrics["mse"]),
        validation_mae=float(metrics["mae"]),
        validation_r2=float(metrics["r2"]),
        n_iter=int(partition.n_iter),
        runtime_sec=elapsed,
    )


def evaluate_fuzzy_partition_only(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    truncation: str,
    k: int,
    f: float,
    p: float,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> PartitionOnlyCandidate:
    """Evaluate one fuzzy partition-only candidate on the validation section."""
    start = time.perf_counter()
    partition = _fuzzy.fit_fuzzy_partition(
        x_train,
        k=int(k),
        f=float(f),
        p=float(p),
        max_iter=int(max_iter),
        tolerance=float(tolerance),
        truncation_rule=truncation,
        random_state=int(random_state),
    )
    means = group_target_means(y_train, partition.membership, f=float(f))
    valid_membership, _, _ = _fuzzy.predict_fuzzy_partition(
        x_valid,
        partition.centroids,
        f=float(f),
        p=float(p),
    )
    y_pred = predict_from_group_means(valid_membership, means, f=float(f))
    metrics = _metrics.regression_metrics(y_valid, y_pred)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
    return PartitionOnlyCandidate(
        partition_type="fuzzy",
        truncation=truncation,
        k=int(k),
        f=float(f),
        p=float(p),
        validation_mse=float(metrics["mse"]),
        validation_mae=float(metrics["mae"]),
        validation_r2=float(metrics["r2"]),
        n_iter=int(partition.n_iter),
        runtime_sec=elapsed,
    )


def refit_and_test_partition_only(
    split: Dict[str, np.ndarray],
    selected: PartitionOnlyCandidate,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> Dict[str, float]:
    """Refit a selected partition-only candidate and evaluate the test section."""
    x_full, y_full = combine_train_valid(split)
    start = time.perf_counter()

    if selected.partition_type == "crisp":
        partition = _crisp.fit_crisp_partition(
            x_full,
            k=selected.k,
            p=selected.p,
            max_iter=max_iter,
            tolerance=tolerance,
            truncation_rule=selected.truncation,
            random_state=random_state,
        )
        means = group_target_means(y_full, partition.membership, f=1.0)
        test_membership, _, _ = _crisp.predict_crisp_partition(split["x_test"], partition.centroids, p=selected.p)
        y_pred = predict_from_group_means(test_membership, means, f=1.0)
    elif selected.partition_type == "fuzzy":
        partition = _fuzzy.fit_fuzzy_partition(
            x_full,
            k=selected.k,
            f=selected.f,
            p=selected.p,
            max_iter=max_iter,
            tolerance=tolerance,
            truncation_rule=selected.truncation,
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
    else:
        raise ValueError(f"Unknown partition type: {selected.partition_type}")

    metrics = _metrics.regression_metrics(split["y_test"], y_pred)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
    row: Dict[str, float] = {
        "partition_type": selected.partition_type,
        "truncation": selected.truncation,
        "k": selected.k,
        "f": selected.f,
        "p": selected.p,
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


def run_partition_only_regression(
    split_path: Path,
    dataset_name: str,
    output_path: Path,
    partition_types: Sequence[str] = ("crisp", "fuzzy"),
    truncations: Sequence[str] = _config.TRUNCATION_RULES,
    k_values_text: Optional[str] = None,
    p_values_text: Optional[str] = None,
    f_values_text: Optional[str] = None,
    max_iter: int = _config.MAX_ITERATIONS,
    tolerance: float = _config.TOLERANCE,
    random_state: int = _config.RANDOM_STATE,
    max_candidates: Optional[int] = None,
    search_record_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Search partition-only candidates and save the selected test rows."""
    split = load_split_npz(split_path)
    k_values, p_values, f_values = _candidate_grid(
        split["x_train"].shape[0],
        k_values_text,
        p_values_text,
        f_values_text,
    )

    crisp_pairs = list(itertools.product(k_values, p_values))
    fuzzy_tuples = list(itertools.product(k_values, f_values, p_values))
    if max_candidates is not None:
        crisp_pairs = crisp_pairs[: int(max_candidates)]
        fuzzy_tuples = fuzzy_tuples[: int(max_candidates)]

    selected_rows: List[Dict[str, float]] = []
    search_records: List[Dict[str, float]] = []

    for partition_type in partition_types:
        partition_type = partition_type.lower().strip()
        for truncation in truncations:
            best: Optional[PartitionOnlyCandidate] = None
            if partition_type == "crisp":
                iterator = enumerate(crisp_pairs)
                for run_index, (k, p) in iterator:
                    candidate = evaluate_crisp_partition_only(
                        split["x_train"],
                        split["y_train"],
                        split["x_valid"],
                        split["y_valid"],
                        truncation=str(truncation),
                        k=int(k),
                        p=float(p),
                        max_iter=int(max_iter),
                        tolerance=float(tolerance),
                        random_state=int(random_state + run_index),
                    )
                    record = candidate.__dict__.copy()
                    record.update({"dataset": dataset_name, "method_family": "partition_only"})
                    search_records.append(record)
                    if best is None or candidate.validation_mse < best.validation_mse:
                        best = candidate
            elif partition_type == "fuzzy":
                iterator = enumerate(fuzzy_tuples)
                for run_index, (k, f, p) in iterator:
                    candidate = evaluate_fuzzy_partition_only(
                        split["x_train"],
                        split["y_train"],
                        split["x_valid"],
                        split["y_valid"],
                        truncation=str(truncation),
                        k=int(k),
                        f=float(f),
                        p=float(p),
                        max_iter=int(max_iter),
                        tolerance=float(tolerance),
                        random_state=int(random_state + run_index),
                    )
                    record = candidate.__dict__.copy()
                    record.update({"dataset": dataset_name, "method_family": "partition_only"})
                    search_records.append(record)
                    if best is None or candidate.validation_mse < best.validation_mse:
                        best = candidate
            else:
                raise ValueError(f"Unsupported partition type: {partition_type}")

            if best is None:
                raise RuntimeError("No partition-only candidate was evaluated.")
            row = refit_and_test_partition_only(
                split,
                selected=best,
                max_iter=int(max_iter),
                tolerance=float(tolerance),
                random_state=int(random_state),
            )
            row.update(
                {
                    "dataset": dataset_name,
                    "task": "regression",
                    "method_family": "partition_only",
                    "method_name": f"{partition_type.title()} clustering only",
                    "degree": np.nan,
                    "n_train": int(split["x_train"].shape[0]),
                    "n_valid": int(split["x_valid"].shape[0]),
                    "n_test": int(split["x_test"].shape[0]),
                    "n_features": int(split["x_train"].shape[1]),
                }
            )
            selected_rows.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_table = pd.DataFrame(selected_rows)
    result_table.to_csv(output_path, index=False)

    if search_record_path is not None:
        search_record_path = Path(search_record_path)
        search_record_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(search_records).to_csv(search_record_path, index=False)
    return result_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit partition-only regression references.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/regression/partition_only_regression.csv"))
    parser.add_argument("--search-records", type=Path, default=None)
    parser.add_argument("--partition-types", default="crisp,fuzzy")
    parser.add_argument("--truncations", default=",".join(_config.TRUNCATION_RULES))
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--f-values", default=None)
    parser.add_argument("--max-iter", type=int, default=_config.MAX_ITERATIONS)
    parser.add_argument("--tolerance", type=float, default=_config.TOLERANCE)
    parser.add_argument("--random-state", type=int, default=_config.RANDOM_STATE)
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_partition_only_regression(
        split_path=args.split,
        dataset_name=args.dataset_name,
        output_path=args.output,
        partition_types=[part.strip() for part in args.partition_types.split(",") if part.strip()],
        truncations=[part.strip() for part in args.truncations.split(",") if part.strip()],
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        f_values_text=args.f_values,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        random_state=args.random_state,
        max_candidates=args.max_candidates,
        search_record_path=args.search_records,
    )
    print(f"Saved {table.shape[0]} partition-only rows to {args.output}")


if __name__ == "__main__":
    main()

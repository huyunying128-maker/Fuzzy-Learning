"""
Modified k-means regression references.

The first reference selects k and p by a hard nearest-centroid search. The second
reference keeps the selected hard partition fixed and learns a fuzzy degree f
from validation performance. Both references predict by group-level target means
so that the comparison isolates the partition layer.
"""

from __future__ import annotations

import argparse
import importlib.util
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
_modified = _load_core_module("08_modified_kmeans.py", "fpw_modified_kmeans")


EPSILON = 1e-12


@dataclass(frozen=True)
class ModifiedKMeansRegressionRow:
    """One selected modified k-means regression reference."""

    variant: str
    k: int
    p: float
    f: float
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


def group_target_means(y: np.ndarray, membership: np.ndarray, f: float) -> np.ndarray:
    """Estimate one target mean for each local group."""
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
        raise ValueError("The modified k-means grid must contain feasible k, p, and f values.")
    return k_values, p_values, f_values


def _evaluate_membership_prediction(
    y_train: np.ndarray,
    train_membership: np.ndarray,
    y_valid: np.ndarray,
    valid_membership: np.ndarray,
    f: float,
) -> Dict[str, float]:
    means = group_target_means(y_train, train_membership, f=f)
    y_pred = predict_from_group_means(valid_membership, means, f=f)
    return _metrics.regression_metrics(y_valid, y_pred)


def fit_hard_reference_on_train(
    x_train: np.ndarray,
    k_values: Sequence[int],
    p_values: Sequence[float],
    max_iter: int,
    tolerance: float,
    random_state: int,
    max_candidates: Optional[int],
):
    """Search the hard k,p reference on the training section."""
    return _modified.search_hard_kp_reference(
        x_train,
        k_values=k_values,
        p_values=p_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
    )


def evaluate_hard_kp_row(
    hard_result,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> ModifiedKMeansRegressionRow:
    """Evaluate the selected hard k,p reference on the validation section."""
    start = time.perf_counter()
    valid_membership, _, _ = _modified.transform_with_modified_kmeans(x_valid, hard_result, use_fuzzy=False)
    metrics = _evaluate_membership_prediction(
        y_train,
        hard_result.hard_membership,
        y_valid,
        valid_membership,
        f=1.0,
    )
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
    return ModifiedKMeansRegressionRow(
        variant="k_p",
        k=int(hard_result.k),
        p=float(hard_result.p),
        f=float("nan"),
        validation_mse=float(metrics["mse"]),
        validation_mae=float(metrics["mae"]),
        validation_r2=float(metrics["r2"]),
        n_iter=int(hard_result.n_iter),
        runtime_sec=elapsed,
    )


def select_post_partition_f(
    hard_result,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    f_values: Sequence[float],
) -> Tuple[ModifiedKMeansRegressionRow, List[Dict[str, float]]]:
    """Select f after the hard k-means partition is fixed."""
    records: List[Dict[str, float]] = []
    best_row: Optional[ModifiedKMeansRegressionRow] = None

    for f in f_values:
        start = time.perf_counter()
        train_result = _modified.add_distance_based_fuzzy_layer(hard_result, f=float(f))
        valid_membership, _, _ = _modified.transform_with_modified_kmeans(x_valid, train_result, use_fuzzy=True)
        metrics = _evaluate_membership_prediction(
            y_train,
            train_result.fuzzy_membership,
            y_valid,
            valid_membership,
            f=float(f),
        )
        elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
        row = ModifiedKMeansRegressionRow(
            variant="k_p_f",
            k=int(hard_result.k),
            p=float(hard_result.p),
            f=float(f),
            validation_mse=float(metrics["mse"]),
            validation_mae=float(metrics["mae"]),
            validation_r2=float(metrics["r2"]),
            n_iter=int(hard_result.n_iter),
            runtime_sec=elapsed,
        )
        records.append(row.__dict__.copy())
        if best_row is None or row.validation_mse < best_row.validation_mse:
            best_row = row

    if best_row is None:
        raise RuntimeError("No post-partition fuzzy-degree candidate was evaluated.")
    return best_row, records


def refit_hard_and_test(
    split: Dict[str, np.ndarray],
    selected: ModifiedKMeansRegressionRow,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> Dict[str, float]:
    """Refit a selected modified k-means row and evaluate the test split."""
    x_full, y_full = combine_train_valid(split)
    start = time.perf_counter()
    hard_result = _modified.fit_hard_kmeans_reference(
        x_full,
        k=selected.k,
        p=selected.p,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
    )
    if selected.variant == "k_p":
        train_membership = hard_result.hard_membership
        test_membership, _, _ = _modified.transform_with_modified_kmeans(split["x_test"], hard_result, use_fuzzy=False)
        aggregate_f = 1.0
    elif selected.variant == "k_p_f":
        fuzzy_result = _modified.add_distance_based_fuzzy_layer(hard_result, f=selected.f)
        train_membership = fuzzy_result.fuzzy_membership
        test_membership, _, _ = _modified.transform_with_modified_kmeans(split["x_test"], fuzzy_result, use_fuzzy=True)
        aggregate_f = selected.f
    else:
        raise ValueError(f"Unsupported modified k-means variant: {selected.variant}")

    means = group_target_means(y_full, train_membership, f=aggregate_f)
    y_pred = predict_from_group_means(test_membership, means, f=aggregate_f)
    metrics = _metrics.regression_metrics(split["y_test"], y_pred)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())

    row: Dict[str, float] = {
        "variant": selected.variant,
        "k": selected.k,
        "p": selected.p,
        "f": selected.f,
        "validation_mse": selected.validation_mse,
        "validation_mae": selected.validation_mae,
        "validation_r2": selected.validation_r2,
        "validation_runtime_sec": selected.runtime_sec,
        "test_runtime_sec": elapsed,
        "n_iter": int(hard_result.n_iter),
        "partition_objective": float(hard_result.objective),
    }
    row.update({f"test_{key}": value for key, value in metrics.items()})
    return row


def run_modified_kmeans_regression(
    split_path: Path,
    dataset_name: str,
    output_path: Path,
    k_values_text: Optional[str] = None,
    p_values_text: Optional[str] = None,
    f_values_text: Optional[str] = None,
    max_iter: int = _config.MAX_ITERATIONS,
    tolerance: float = _config.TOLERANCE,
    random_state: int = _config.RANDOM_STATE,
    max_candidates: Optional[int] = None,
    search_record_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fit the modified k-means regression references and save test rows."""
    split = load_split_npz(split_path)
    k_values, p_values, f_values = _candidate_grid(
        split["x_train"].shape[0],
        k_values_text,
        p_values_text,
        f_values_text,
    )

    hard_result = fit_hard_reference_on_train(
        split["x_train"],
        k_values=k_values,
        p_values=p_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
    )
    hard_row = evaluate_hard_kp_row(
        hard_result,
        split["x_train"],
        split["y_train"],
        split["x_valid"],
        split["y_valid"],
    )
    fuzzy_row, f_records = select_post_partition_f(
        hard_result,
        split["x_train"],
        split["y_train"],
        split["x_valid"],
        split["y_valid"],
        f_values=f_values,
    )

    selected_rows = [hard_row, fuzzy_row]
    output_rows: List[Dict[str, float]] = []
    for selected in selected_rows:
        row = refit_hard_and_test(
            split,
            selected=selected,
            max_iter=max_iter,
            tolerance=tolerance,
            random_state=random_state,
        )
        method_name = "Modified k-means, k, p" if selected.variant == "k_p" else "Modified k-means, k, p, f"
        row.update(
            {
                "dataset": dataset_name,
                "task": "regression",
                "method_family": "modified_kmeans",
                "method_name": method_name,
                "degree": np.nan,
                "truncation": "hard_kmeans_reference",
                "n_train": int(split["x_train"].shape[0]),
                "n_valid": int(split["x_valid"].shape[0]),
                "n_test": int(split["x_test"].shape[0]),
                "n_features": int(split["x_train"].shape[1]),
            }
        )
        output_rows.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(output_rows)
    table.to_csv(output_path, index=False)

    if search_record_path is not None:
        search_record_path = Path(search_record_path)
        search_record_path.parent.mkdir(parents=True, exist_ok=True)
        records = [dict(record, stage="hard_kp") for record in hard_result.search_records]
        records.extend(dict(record, stage="post_partition_f") for record in f_records)
        pd.DataFrame(records).to_csv(search_record_path, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit modified k-means regression references.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/regression/modified_kmeans_regression.csv"))
    parser.add_argument("--search-records", type=Path, default=None)
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
    table = run_modified_kmeans_regression(
        split_path=args.split,
        dataset_name=args.dataset_name,
        output_path=args.output,
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        f_values_text=args.f_values,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        random_state=args.random_state,
        max_candidates=args.max_candidates,
        search_record_path=args.search_records,
    )
    print(f"Saved {table.shape[0]} modified k-means regression rows to {args.output}")


if __name__ == "__main__":
    main()

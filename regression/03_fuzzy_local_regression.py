"""
Fuzzy partition-weighted local regression.

The fuzzy branch learns graded memberships, builds membership-weighted local
inputs, fits local polynomial ridge models, and aggregates local outputs with
normalized fuzzy weights.
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
_fuzzy = _load_core_module("07_fuzzy_partition.py", "fpw_fuzzy_partition")
_local = _load_core_module("10_local_models.py", "fpw_local_models")


@dataclass(frozen=True)
class FuzzyRegressionCandidate:
    """One evaluated fuzzy local-regression configuration."""

    degree: int
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


def parse_degrees(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one degree is required.")
    return values


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
    f_values = [f for f in f_values if f >= 1.0]
    p_values = [p for p in p_values if p >= 1.0]
    if not k_values:
        raise ValueError("No feasible k values were provided.")
    if not p_values:
        raise ValueError("No feasible p values were provided.")
    if not f_values:
        raise ValueError("No feasible f values were provided.")
    return k_values, p_values, f_values


def evaluate_fuzzy_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    degree: int,
    truncation: str,
    k: int,
    f: float,
    p: float,
    alpha: float,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> FuzzyRegressionCandidate:
    """Fit one fuzzy local-regression candidate and evaluate validation error."""
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
    model = _local.fit_partition_weighted_regression(
        x_train,
        y_train,
        partition.membership,
        degree=int(degree),
        f=float(f),
        alpha=float(alpha),
    )
    valid_membership, _, _ = _fuzzy.predict_fuzzy_partition(
        x_valid,
        partition.centroids,
        f=float(f),
        p=float(p),
    )
    y_pred = _local.predict_partition_weighted_regression(model, x_valid, valid_membership)
    valid_metrics = _metrics.regression_metrics(y_valid, y_pred)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())

    return FuzzyRegressionCandidate(
        degree=int(degree),
        truncation=truncation,
        k=int(k),
        f=float(f),
        p=float(p),
        validation_mse=float(valid_metrics["mse"]),
        validation_mae=float(valid_metrics["mae"]),
        validation_r2=float(valid_metrics["r2"]),
        n_iter=int(partition.n_iter),
        runtime_sec=elapsed,
    )


def refit_and_test_fuzzy_candidate(
    split: Dict[str, np.ndarray],
    selected: FuzzyRegressionCandidate,
    alpha: float,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> Dict[str, float]:
    """Refit the selected fuzzy candidate and evaluate the held-out test data."""
    x_full, y_full = combine_train_valid(split)
    start = time.perf_counter()
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
    model = _local.fit_partition_weighted_regression(
        x_full,
        y_full,
        partition.membership,
        degree=selected.degree,
        f=selected.f,
        alpha=alpha,
    )
    test_membership, _, _ = _fuzzy.predict_fuzzy_partition(
        split["x_test"],
        partition.centroids,
        f=selected.f,
        p=selected.p,
    )
    y_pred = _local.predict_partition_weighted_regression(model, split["x_test"], test_membership)
    test_metrics = _metrics.regression_metrics(split["y_test"], y_pred)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())

    row: Dict[str, float] = {
        "degree": selected.degree,
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
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    return row


def run_fuzzy_local_regression(
    split_path: Path,
    dataset_name: str,
    output_path: Path,
    degrees: Sequence[int],
    truncations: Sequence[str],
    k_values_text: Optional[str] = None,
    p_values_text: Optional[str] = None,
    f_values_text: Optional[str] = None,
    alpha: float = _config.RIDGE_ALPHA,
    max_iter: int = _config.MAX_ITERATIONS,
    tolerance: float = _config.TOLERANCE,
    random_state: int = _config.RANDOM_STATE,
    max_candidates: Optional[int] = None,
    search_record_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Search fuzzy local-regression candidates and save selected test rows."""
    split = load_split_npz(split_path)
    k_values, p_values, f_values = _candidate_grid(
        split["x_train"].shape[0],
        k_values_text,
        p_values_text,
        f_values_text,
    )

    candidate_tuples = list(itertools.product(k_values, f_values, p_values))
    if max_candidates is not None:
        candidate_tuples = candidate_tuples[: int(max_candidates)]

    selected_rows: List[Dict[str, float]] = []
    search_records: List[Dict[str, float]] = []
    for degree in degrees:
        for truncation in truncations:
            best: Optional[FuzzyRegressionCandidate] = None
            for run_index, (k, f, p) in enumerate(candidate_tuples):
                candidate = evaluate_fuzzy_candidate(
                    split["x_train"],
                    split["y_train"],
                    split["x_valid"],
                    split["y_valid"],
                    degree=int(degree),
                    truncation=str(truncation),
                    k=int(k),
                    f=float(f),
                    p=float(p),
                    alpha=float(alpha),
                    max_iter=int(max_iter),
                    tolerance=float(tolerance),
                    random_state=int(random_state + run_index),
                )
                record = candidate.__dict__.copy()
                record.update({"dataset": dataset_name, "method_family": "fuzzy_local"})
                search_records.append(record)
                if best is None or candidate.validation_mse < best.validation_mse:
                    best = candidate

            if best is None:
                raise RuntimeError("No fuzzy local-regression candidate was evaluated.")
            row = refit_and_test_fuzzy_candidate(
                split,
                selected=best,
                alpha=float(alpha),
                max_iter=int(max_iter),
                tolerance=float(tolerance),
                random_state=int(random_state),
            )
            row.update(
                {
                    "dataset": dataset_name,
                    "task": "regression",
                    "method_family": "fuzzy_local",
                    "method_name": "Fuzzy local regression",
                    "alpha": float(alpha),
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
    parser = argparse.ArgumentParser(description="Fit fuzzy partition-weighted local regression.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/regression/fuzzy_local_regression.csv"))
    parser.add_argument("--search-records", type=Path, default=None)
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--truncations", default=",".join(_config.TRUNCATION_RULES))
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--f-values", default=None)
    parser.add_argument("--alpha", type=float, default=_config.RIDGE_ALPHA)
    parser.add_argument("--max-iter", type=int, default=_config.MAX_ITERATIONS)
    parser.add_argument("--tolerance", type=float, default=_config.TOLERANCE)
    parser.add_argument("--random-state", type=int, default=_config.RANDOM_STATE)
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_fuzzy_local_regression(
        split_path=args.split,
        dataset_name=args.dataset_name,
        output_path=args.output,
        degrees=parse_degrees(args.degrees),
        truncations=[part.strip() for part in args.truncations.split(",") if part.strip()],
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        f_values_text=args.f_values,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        random_state=args.random_state,
        max_candidates=args.max_candidates,
        search_record_path=args.search_records,
    )
    print(f"Saved {table.shape[0]} fuzzy local-regression rows to {args.output}")


if __name__ == "__main__":
    main()

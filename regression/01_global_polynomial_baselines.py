"""
Global polynomial regression baselines for the tabular experiments.

This module fits degree 1 through degree 4 ridge-polynomial models on the raw
standardized input. The resulting rows provide the non-partitioned regression
reference used before crisp and fuzzy local learning are evaluated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


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
_poly = _load_core_module("09_polynomial_features.py", "fpw_polynomial_features")


def load_split_npz(split_path: Path) -> Dict[str, np.ndarray]:
    """Load a train-validation-test split saved by the dataset utilities."""
    data = np.load(Path(split_path), allow_pickle=True)
    required = ("x_train", "y_train", "x_test", "y_test")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing arrays in split file: {missing}")

    split = {name: data[name] for name in data.files}
    if "x_valid" not in split or "y_valid" not in split:
        split["x_valid"] = np.empty((0, split["x_train"].shape[1]), dtype=split["x_train"].dtype)
        split["y_valid"] = np.empty((0,), dtype=split["y_train"].dtype)
    return split


def combine_train_valid(split: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Combine the training and validation sections for final model fitting."""
    if split["x_valid"].shape[0] == 0:
        return {
            "x_train_full": split["x_train"],
            "y_train_full": split["y_train"],
        }
    return {
        "x_train_full": np.vstack([split["x_train"], split["x_valid"]]),
        "y_train_full": np.concatenate([split["y_train"], split["y_valid"]]),
    }


def _fit_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    degree: int,
    alpha: float,
) -> Dict[str, float]:
    design_train = _poly.polynomial_design_matrix(x_train, degree=degree)
    design_eval = _poly.polynomial_design_matrix(x_eval, degree=degree)

    model = Ridge(alpha=float(alpha), fit_intercept=False)
    model.fit(design_train, np.asarray(y_train, dtype=float).reshape(-1))
    prediction = model.predict(design_eval)
    return _metrics.regression_metrics(y_eval, prediction)


def run_global_polynomial_baselines(
    split_path: Path,
    dataset_name: str,
    output_path: Path,
    degrees: Sequence[int] = _config.LOCAL_DEGREES,
    alpha: float = _config.RIDGE_ALPHA,
) -> pd.DataFrame:
    """Fit and save the global polynomial baseline rows."""
    split = load_split_npz(split_path)
    combined = combine_train_valid(split)
    records: List[Dict[str, float]] = []

    for degree in degrees:
        start = time.perf_counter()
        test_metrics = _fit_and_evaluate(
            combined["x_train_full"],
            combined["y_train_full"],
            split["x_test"],
            split["y_test"],
            degree=int(degree),
            alpha=float(alpha),
        )
        elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())

        row: Dict[str, float] = {
            "dataset": dataset_name,
            "task": "regression",
            "method_family": "global_polynomial",
            "method_name": "Global polynomial baseline",
            "degree": int(degree),
            "k": np.nan,
            "f": np.nan,
            "p": np.nan,
            "truncation": "none",
            "alpha": float(alpha),
            "n_train_full": int(combined["x_train_full"].shape[0]),
            "n_test": int(split["x_test"].shape[0]),
            "n_features": int(split["x_train"].shape[1]),
            "runtime_sec": elapsed,
        }
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        records.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output_path, index=False)
    return table


def parse_degrees(text: str) -> List[int]:
    """Parse a comma-separated degree list."""
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    for value in values:
        _poly.validate_degree(value)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit global polynomial regression baselines.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/regression/global_polynomial_baselines.csv"))
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--alpha", type=float, default=_config.RIDGE_ALPHA)
    parser.add_argument("--metadata", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_global_polynomial_baselines(
        split_path=args.split,
        dataset_name=args.dataset_name,
        output_path=args.output,
        degrees=parse_degrees(args.degrees),
        alpha=args.alpha,
    )
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata.open("w", encoding="utf-8") as file:
            json.dump({"rows": int(table.shape[0]), "output": str(args.output)}, file, indent=2)
    print(f"Saved {table.shape[0]} baseline rows to {args.output}")


if __name__ == "__main__":
    main()

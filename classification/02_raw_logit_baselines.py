"""
Raw logit baselines for MNIST classification.

The raw-logit branch fits multiclass logistic regression directly on the image
input before any partition weighting is introduced. Degrees 1 through 4 are
represented by a polynomial image feature map, with a screened power basis used
for high-dimensional images to keep the design matrix practical.
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
from sklearn.linear_model import LogisticRegression


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


_config = _load_core_module("01_config.py", "fpw_rawlogit_config")
_metrics = _load_core_module("02_metrics.py", "fpw_rawlogit_metrics")
_poly = _load_core_module("09_polynomial_features.py", "fpw_rawlogit_poly")


@dataclass
class ClassificationDesignSpec:
    """Metadata for the raw-logit feature map."""

    degree: int
    mode: str
    selected_features: np.ndarray
    dense_basis: bool
    interaction_features: int
    n_input_features: int
    n_output_features: int


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


def combine_train_valid(split: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Combine train and validation arrays for final baseline fitting."""
    if split["x_valid"].shape[0] == 0:
        return np.asarray(split["x_train"], dtype=float), np.asarray(split["y_train"], dtype=int)
    return (
        np.vstack([np.asarray(split["x_train"], dtype=float), np.asarray(split["x_valid"], dtype=float)]),
        np.concatenate([np.asarray(split["y_train"], dtype=int), np.asarray(split["y_valid"], dtype=int)]),
    )


def parse_degrees(text: str) -> List[int]:
    """Parse a comma-separated list of logit degrees."""
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    for value in values:
        _poly.validate_degree(value)
    if not values:
        raise ValueError("At least one degree is required.")
    return values


def _selected_by_variance(X: np.ndarray, max_base_features: Optional[int]) -> np.ndarray:
    if max_base_features is None or max_base_features >= X.shape[1]:
        return np.arange(X.shape[1], dtype=int)
    if max_base_features < 1:
        raise ValueError("max_base_features must be positive when it is provided.")
    variances = np.var(X, axis=0)
    selected = np.argsort(variances)[-int(max_base_features):]
    return np.sort(selected.astype(int))


def fit_classification_design_spec(
    X: np.ndarray,
    degree: int,
    mode: str = "auto",
    dense_feature_limit: int = 32,
    max_base_features: Optional[int] = 784,
    interaction_features: int = 0,
) -> ClassificationDesignSpec:
    """Fit the metadata needed to transform raw inputs into logit features."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    degree = _poly.validate_degree(degree)
    mode = str(mode).lower()
    if mode not in {"auto", "dense", "screened_power"}:
        raise ValueError("mode must be 'auto', 'dense', or 'screened_power'.")

    use_dense = mode == "dense" or (mode == "auto" and X.shape[1] <= dense_feature_limit)
    if use_dense:
        n_output = _poly.count_polynomial_features(X.shape[1], degree, include_intercept=False)
        return ClassificationDesignSpec(
            degree=degree,
            mode="dense" if mode == "dense" else "auto_dense",
            selected_features=np.arange(X.shape[1], dtype=int),
            dense_basis=True,
            interaction_features=0,
            n_input_features=int(X.shape[1]),
            n_output_features=int(n_output),
        )

    selected = _selected_by_variance(X, max_base_features=max_base_features)
    interaction_features = int(max(0, interaction_features))
    if interaction_features > selected.shape[0]:
        interaction_features = int(selected.shape[0])

    n_power = int(selected.shape[0] * degree)
    n_interactions = 0
    if degree >= 2 and interaction_features >= 2:
        n_interactions = interaction_features * (interaction_features - 1) // 2
    return ClassificationDesignSpec(
        degree=degree,
        mode="screened_power",
        selected_features=selected,
        dense_basis=False,
        interaction_features=interaction_features,
        n_input_features=int(X.shape[1]),
        n_output_features=int(n_power + n_interactions),
    )


def transform_classification_design(X: np.ndarray, spec: ClassificationDesignSpec) -> np.ndarray:
    """Transform inputs according to a fitted raw-logit design specification."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if X.shape[1] != spec.n_input_features:
        raise ValueError("X has a different number of features from the fitted design spec.")

    if spec.dense_basis:
        return _poly.polynomial_design_matrix(X, degree=spec.degree, include_intercept=False)

    base = X[:, spec.selected_features]
    blocks = [base ** power for power in range(1, spec.degree + 1)]
    if spec.degree >= 2 and spec.interaction_features >= 2:
        interaction_base = base[:, : spec.interaction_features]
        pairs = []
        for left in range(spec.interaction_features):
            for right in range(left + 1, spec.interaction_features):
                pairs.append((interaction_base[:, left] * interaction_base[:, right])[:, None])
        if pairs:
            blocks.append(np.hstack(pairs))
    return np.hstack(blocks)


def _aligned_probabilities(model: LogisticRegression, X_design: np.ndarray, n_classes: int = 10) -> np.ndarray:
    raw = model.predict_proba(X_design)
    probabilities = np.full((X_design.shape[0], n_classes), EPSILON, dtype=float)
    for source_index, cls in enumerate(model.classes_):
        probabilities[:, int(cls)] = raw[:, source_index]
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    return probabilities


def fit_raw_logit_for_degree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    degree: int,
    c_value: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    feature_mode: str = "auto",
    dense_feature_limit: int = 32,
    max_base_features: Optional[int] = 784,
    interaction_features: int = 0,
) -> Dict[str, float]:
    """Fit one raw-logit baseline row for a selected degree."""
    start = time.perf_counter()
    spec = fit_classification_design_spec(
        x_train,
        degree=degree,
        mode=feature_mode,
        dense_feature_limit=dense_feature_limit,
        max_base_features=max_base_features,
        interaction_features=interaction_features,
    )
    design_train = transform_classification_design(x_train, spec)
    design_test = transform_classification_design(x_test, spec)

    model = LogisticRegression(
        C=float(c_value),
        max_iter=int(max_iter),
        solver=solver,
        n_jobs=None if solver == "lbfgs" else -1,
    )
    model.fit(design_train, np.asarray(y_train, dtype=int).reshape(-1))
    probabilities = _aligned_probabilities(model, design_test, n_classes=10)
    metrics = _metrics.classification_metrics(y_test, probabilities=probabilities)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())

    row: Dict[str, float] = {
        "dataset": "mnist",
        "task": "classification",
        "method_family": "raw_logit",
        "method_name": "Raw logit",
        "degree": int(degree),
        "k": np.nan,
        "f": np.nan,
        "p": np.nan,
        "truncation": "none",
        "feature_mode": spec.mode,
        "n_input_features": spec.n_input_features,
        "n_design_features": spec.n_output_features,
        "c_value": float(c_value),
        "max_iter": int(max_iter),
        "test_accuracy": metrics["accuracy"],
        "test_cross_entropy": metrics["cross_entropy"],
        "runtime_sec": elapsed,
    }
    return row


def run_raw_logit_baselines(
    split_path: Path,
    output_path: Path = OUTPUT_DIR / "02_raw_logit_baselines.csv",
    degrees: Sequence[int] = _config.LOCAL_DEGREES,
    c_value: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    feature_mode: str = "auto",
    dense_feature_limit: int = 32,
    max_base_features: Optional[int] = 784,
    interaction_features: int = 0,
) -> pd.DataFrame:
    """Fit and save raw-logit baseline rows for MNIST."""
    split = load_split_npz(split_path)
    x_train_full, y_train_full = combine_train_valid(split)
    x_test = np.asarray(split["x_test"], dtype=float)
    y_test = np.asarray(split["y_test"], dtype=int)

    records = []
    for degree in degrees:
        records.append(
            fit_raw_logit_for_degree(
                x_train_full,
                y_train_full,
                x_test,
                y_test,
                degree=int(degree),
                c_value=c_value,
                max_iter=max_iter,
                solver=solver,
                feature_mode=feature_mode,
                dense_feature_limit=dense_feature_limit,
                max_base_features=max_base_features,
                interaction_features=interaction_features,
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output_path, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MNIST raw-logit baselines.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "02_raw_logit_baselines.csv")
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--feature-mode", default="auto", choices=["auto", "dense", "screened_power"])
    parser.add_argument("--dense-feature-limit", type=int, default=32)
    parser.add_argument("--max-base-features", type=int, default=784)
    parser.add_argument("--interaction-features", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_raw_logit_baselines(
        split_path=args.split,
        output_path=args.output,
        degrees=parse_degrees(args.degrees),
        c_value=args.c_value,
        max_iter=args.max_iter,
        solver=args.solver,
        feature_mode=args.feature_mode,
        dense_feature_limit=args.dense_feature_limit,
        max_base_features=args.max_base_features,
        interaction_features=args.interaction_features,
    )
    print(f"Saved raw-logit baseline table with {table.shape[0]} rows")


if __name__ == "__main__":
    main()

"""
Crisp local-logit classification for MNIST.

The local classifier first learns a hard nearest-centroid partition, constructs
membership-weighted local inputs, fits one multinomial logit model for each
local group, and aggregates the local logits into the final softmax output.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


_config = _load_core_module("01_config.py", "fpw_crisp_logit_config")
_metrics = _load_core_module("02_metrics.py", "fpw_crisp_logit_metrics")
_crisp = _load_core_module("06_crisp_partition.py", "fpw_crisp_logit_partition")
_membership = _load_core_module("04_membership_functions.py", "fpw_crisp_logit_membership")
_poly = _load_core_module("09_polynomial_features.py", "fpw_crisp_logit_poly")


@dataclass
class ScreenedLocalDesignSpec:
    """Feature-map metadata used by the high-dimensional local logit models."""

    degree: int
    selected_features: np.ndarray
    interaction_features: int
    n_input_features: int
    n_output_features: int


@dataclass
class LocalLogitModel:
    """A fitted local logit model and its feature-map metadata."""

    model: object
    spec: ScreenedLocalDesignSpec


@dataclass
class ConstantLogitModel:
    """A stable fallback for a local group with a single effective class."""

    logits: np.ndarray

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.logits, (X.shape[0], 1))


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


def combine_train_valid(split: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Combine train and validation arrays for final model fitting."""
    x_train = np.asarray(split["x_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=int)
    x_valid = np.asarray(split.get("x_valid", np.empty((0, x_train.shape[1]))), dtype=float)
    y_valid = np.asarray(split.get("y_valid", np.empty((0,), dtype=int)), dtype=int)
    if x_valid.shape[0] == 0:
        return x_train, y_train
    return np.vstack([x_train, x_valid]), np.concatenate([y_train, y_valid])


def parse_degrees(text: str) -> List[int]:
    """Parse a comma-separated list of local logit degrees."""
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one degree is required.")
    return [_poly.validate_degree(value) for value in values]


def parse_numeric_grid(text: Optional[str], default_values: Sequence[float], cast=float) -> List:
    """Parse a comma-separated numeric grid."""
    if text is None or str(text).strip() == "":
        return [cast(value) for value in default_values]
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(cast(float(part)) if cast is int else cast(part))
    if not values:
        raise ValueError("The parsed grid is empty.")
    return values


def parse_truncations(text: str) -> List[str]:
    """Parse a comma-separated list of truncation rules."""
    values = [part.strip().lower() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one truncation rule is required.")
    return values


def default_partition_grids(n_samples: int) -> Tuple[List[int], List[float], List[float]]:
    """Return the standard k, p, and f search grids."""
    grid = _config.make_partition_grid(n_samples)
    return grid.k_values, grid.p_values, grid.f_values


def _selected_by_variance(X: np.ndarray, max_base_features: Optional[int]) -> np.ndarray:
    if max_base_features is None or max_base_features >= X.shape[1]:
        return np.arange(X.shape[1], dtype=int)
    if max_base_features < 1:
        raise ValueError("max_base_features must be positive when provided.")
    variances = np.var(X, axis=0)
    selected = np.argsort(variances)[-int(max_base_features):]
    return np.sort(selected.astype(int))


def fit_screened_design_spec(
    X: np.ndarray,
    degree: int,
    max_base_features: Optional[int] = 256,
    interaction_features: int = 0,
) -> ScreenedLocalDesignSpec:
    """Fit a practical local polynomial design for image vectors."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    degree = _poly.validate_degree(degree)
    selected = _selected_by_variance(X, max_base_features=max_base_features)
    interaction_features = int(max(0, min(interaction_features, selected.shape[0])))
    n_power = int(selected.shape[0] * degree)
    n_interactions = 0
    if degree >= 2 and interaction_features >= 2:
        n_interactions = interaction_features * (interaction_features - 1) // 2
    return ScreenedLocalDesignSpec(
        degree=degree,
        selected_features=selected,
        interaction_features=interaction_features,
        n_input_features=int(X.shape[1]),
        n_output_features=int(n_power + n_interactions),
    )


def transform_screened_design(X: np.ndarray, spec: ScreenedLocalDesignSpec) -> np.ndarray:
    """Transform local weighted inputs with the screened power basis."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if X.shape[1] != spec.n_input_features:
        raise ValueError("X has a different number of features from the design spec.")
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


def class_prior_logits(y: np.ndarray, sample_weight: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Compute smoothed log-prior logits for a local group."""
    y = np.asarray(y, dtype=int).reshape(-1)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    counts = np.zeros(n_classes, dtype=float)
    for label in range(n_classes):
        counts[label] = np.sum(weights[y == label])
    probabilities = (counts + 1.0) / (np.sum(counts) + n_classes)
    return np.log(np.clip(probabilities, EPSILON, 1.0))


def aligned_logits(model: object, X_design: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Align model logits to columns 0 through 9."""
    raw = model.decision_function(X_design)
    if raw.ndim == 1:
        raw = np.column_stack([-raw, raw])
    aligned = np.zeros((X_design.shape[0], n_classes), dtype=float)
    model_classes = getattr(model, "classes_", np.arange(n_classes))
    for source_index, cls in enumerate(model_classes):
        aligned[:, int(cls)] = raw[:, source_index]
    return aligned


def fit_local_logit_models(
    X: np.ndarray,
    y: np.ndarray,
    membership: np.ndarray,
    degree: int,
    f: float = 1.0,
    c_value: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    max_base_features: Optional[int] = 256,
    interaction_features: int = 0,
    n_classes: int = 10,
) -> List[LocalLogitModel]:
    """Fit one weighted local logit model for each group."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    membership = np.asarray(membership, dtype=float)
    if X.ndim != 2 or membership.ndim != 2:
        raise ValueError("X and membership must be two-dimensional.")
    if X.shape[0] != y.shape[0] or X.shape[0] != membership.shape[0]:
        raise ValueError("X, y, and membership must have matching rows.")

    local_inputs = _membership.local_weighted_inputs(X, membership)
    weights = np.maximum(membership, 0.0) ** float(f)
    models: List[LocalLogitModel] = []

    for j in range(membership.shape[1]):
        local_x = local_inputs[:, j, :]
        spec = fit_screened_design_spec(
            local_x,
            degree=degree,
            max_base_features=max_base_features,
            interaction_features=interaction_features,
        )
        design = transform_screened_design(local_x, spec)
        sample_weight = np.maximum(weights[:, j], EPSILON)
        active_classes = np.unique(y[sample_weight > EPSILON])

        if active_classes.shape[0] < 2 or np.sum(sample_weight) <= EPSILON:
            model = ConstantLogitModel(class_prior_logits(y, sample_weight, n_classes=n_classes))
        else:
            model = LogisticRegression(
                C=float(c_value),
                max_iter=int(max_iter),
                solver=solver,
                n_jobs=None if solver == "lbfgs" else -1,
            )
            model.fit(design, y, sample_weight=sample_weight)
        models.append(LocalLogitModel(model=model, spec=spec))
    return models


def predict_local_logit_probabilities(
    models: Sequence[LocalLogitModel],
    X: np.ndarray,
    membership: np.ndarray,
    f: float = 1.0,
    n_classes: int = 10,
) -> np.ndarray:
    """Aggregate local logits and return final class probabilities."""
    X = np.asarray(X, dtype=float)
    membership = np.asarray(membership, dtype=float)
    local_inputs = _membership.local_weighted_inputs(X, membership)
    weights = _membership.aggregation_weights(membership, f=float(f))
    logits = np.zeros((X.shape[0], len(models), n_classes), dtype=float)
    for j, fitted in enumerate(models):
        design = transform_screened_design(local_inputs[:, j, :], fitted.spec)
        logits[:, j, :] = aligned_logits(fitted.model, design, n_classes=n_classes)
    final_logits = np.sum(weights[:, :, None] * logits, axis=1)
    return _metrics.softmax(final_logits)


def candidate_iterator(k_values: Sequence[int], p_values: Sequence[float], max_candidates: Optional[int]):
    """Iterate through candidate k and p pairs."""
    count = 0
    for k in k_values:
        for p in p_values:
            yield int(k), float(p)
            count += 1
            if max_candidates is not None and count >= int(max_candidates):
                return


def evaluate_crisp_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    degree: int,
    k: int,
    p: float,
    truncation: str,
    max_iter_partition: int,
    tolerance: float,
    random_state: int,
    c_value: float,
    max_iter_logit: int,
    solver: str,
    max_base_features: Optional[int],
    interaction_features: int,
) -> Tuple[float, Dict[str, float], object, List[LocalLogitModel]]:
    """Fit and validate one crisp local-logit candidate."""
    partition = _crisp.fit_crisp_partition(
        x_train,
        k=int(k),
        p=float(p),
        max_iter=max_iter_partition,
        tolerance=tolerance,
        truncation_rule=truncation,
        random_state=random_state,
    )
    valid_membership, _, _ = _crisp.predict_crisp_partition(x_valid, partition.centroids, p=float(p))
    models = fit_local_logit_models(
        x_train,
        y_train,
        partition.membership,
        degree=degree,
        f=1.0,
        c_value=c_value,
        max_iter=max_iter_logit,
        solver=solver,
        max_base_features=max_base_features,
        interaction_features=interaction_features,
    )
    probabilities = predict_local_logit_probabilities(models, x_valid, valid_membership, f=1.0, n_classes=10)
    metrics = _metrics.classification_metrics(y_valid, probabilities=probabilities)
    return float(metrics["cross_entropy"]), metrics, partition, models


def run_crisp_local_logit(
    split_path: Path,
    output_path: Path = OUTPUT_DIR / "03_crisp_local_logit.csv",
    degrees: Sequence[int] = _config.LOCAL_DEGREES,
    truncations: Sequence[str] = _config.TRUNCATION_RULES,
    k_values_text: Optional[str] = "760,1020,1400,1750",
    p_values_text: Optional[str] = "2.00",
    max_candidates: Optional[int] = None,
    max_iter_partition: int = 300,
    tolerance: float = 1e-6,
    random_state: int = 42,
    c_value: float = 1.0,
    max_iter_logit: int = 1000,
    solver: str = "lbfgs",
    max_base_features: Optional[int] = 256,
    interaction_features: int = 0,
    search_record_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the crisp local-logit grid and save the selected test rows."""
    split = load_split_npz(split_path)
    x_train = np.asarray(split["x_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=int)
    x_valid = np.asarray(split["x_valid"], dtype=float)
    y_valid = np.asarray(split["y_valid"], dtype=int)
    x_test = np.asarray(split["x_test"], dtype=float)
    y_test = np.asarray(split["y_test"], dtype=int)

    default_k, default_p, _ = default_partition_grids(x_train.shape[0])
    k_values = parse_numeric_grid(k_values_text, default_k, cast=int)
    p_values = parse_numeric_grid(p_values_text, default_p, cast=float)

    records: List[Dict[str, float]] = []
    search_records: List[Dict[str, float]] = []

    for degree in degrees:
        for truncation in truncations:
            start = time.perf_counter()
            best = None
            best_loss = float("inf")
            best_metrics: Dict[str, float] = {}
            for run_index, (k, p) in enumerate(candidate_iterator(k_values, p_values, max_candidates)):
                try:
                    loss, valid_metrics, partition, _ = evaluate_crisp_candidate(
                        x_train,
                        y_train,
                        x_valid,
                        y_valid,
                        degree=int(degree),
                        k=k,
                        p=p,
                        truncation=truncation,
                        max_iter_partition=max_iter_partition,
                        tolerance=tolerance,
                        random_state=random_state + run_index,
                        c_value=c_value,
                        max_iter_logit=max_iter_logit,
                        solver=solver,
                        max_base_features=max_base_features,
                        interaction_features=interaction_features,
                    )
                    search_records.append(
                        {
                            "degree": int(degree),
                            "truncation": truncation,
                            "k": int(k),
                            "p": float(p),
                            "validation_cross_entropy": float(loss),
                            "validation_accuracy": float(valid_metrics["accuracy"]),
                            "status": "ok",
                        }
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best = (int(k), float(p), partition)
                        best_metrics = valid_metrics
                except Exception as exc:  # keeps long grids auditable
                    search_records.append(
                        {
                            "degree": int(degree),
                            "truncation": truncation,
                            "k": int(k),
                            "p": float(p),
                            "validation_cross_entropy": np.nan,
                            "validation_accuracy": np.nan,
                            "status": f"failed: {exc}",
                        }
                    )

            if best is None:
                continue

            selected_k, selected_p, _ = best
            x_full, y_full = combine_train_valid(split)
            final_partition = _crisp.fit_crisp_partition(
                x_full,
                k=selected_k,
                p=selected_p,
                max_iter=max_iter_partition,
                tolerance=tolerance,
                truncation_rule=truncation,
                random_state=random_state,
            )
            test_membership, _, _ = _crisp.predict_crisp_partition(x_test, final_partition.centroids, p=selected_p)
            final_models = fit_local_logit_models(
                x_full,
                y_full,
                final_partition.membership,
                degree=int(degree),
                f=1.0,
                c_value=c_value,
                max_iter=max_iter_logit,
                solver=solver,
                max_base_features=max_base_features,
                interaction_features=interaction_features,
            )
            probabilities = predict_local_logit_probabilities(final_models, x_test, test_membership, f=1.0, n_classes=10)
            test_metrics = _metrics.classification_metrics(y_test, probabilities=probabilities)
            elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())

            records.append(
                {
                    "dataset": "mnist",
                    "task": "classification",
                    "method_family": "crisp_local_logit",
                    "method_name": "Crisp local logit",
                    "degree": int(degree),
                    "truncation": truncation,
                    "k": int(selected_k),
                    "f": np.nan,
                    "p": float(selected_p),
                    "validation_accuracy": float(best_metrics.get("accuracy", np.nan)),
                    "validation_cross_entropy": float(best_loss),
                    "test_accuracy": float(test_metrics["accuracy"]),
                    "test_cross_entropy": float(test_metrics["cross_entropy"]),
                    "n_iter": int(final_partition.n_iter),
                    "runtime_sec": elapsed,
                    "n_design_features": int(final_models[0].spec.n_output_features) if final_models else 0,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output_path, index=False)

    if search_record_path is not None:
        search_record_path = Path(search_record_path)
        search_record_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(search_records).to_csv(search_record_path, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MNIST crisp local-logit models.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "03_crisp_local_logit.csv")
    parser.add_argument("--search-records", type=Path, default=None)
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--truncations", default="dtd,harmonic,sp,entropy,hpd")
    parser.add_argument("--k-values", default="760,1020,1400,1750")
    parser.add_argument("--p-values", default="2.00")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-iter-partition", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter-logit", type=int, default=1000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--max-base-features", type=int, default=256)
    parser.add_argument("--interaction-features", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_crisp_local_logit(
        split_path=args.split,
        output_path=args.output,
        degrees=parse_degrees(args.degrees),
        truncations=parse_truncations(args.truncations),
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        max_candidates=args.max_candidates,
        max_iter_partition=args.max_iter_partition,
        tolerance=args.tolerance,
        random_state=args.random_state,
        c_value=args.c_value,
        max_iter_logit=args.max_iter_logit,
        solver=args.solver,
        max_base_features=args.max_base_features,
        interaction_features=args.interaction_features,
        search_record_path=args.search_records,
    )
    print(f"Saved crisp local-logit table with {table.shape[0]} rows")


if __name__ == "__main__":
    main()

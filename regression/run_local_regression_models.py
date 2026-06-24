"""
Partition-weighted local polynomial regression experiments.

The script learns crisp or fuzzy partitions, fits group-specific polynomial
ridge models on membership-weighted inputs, and evaluates aggregated test
predictions for concrete and superconductivity regression tasks.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.prepare_regression_data import (  # noqa: E402
    RegressionDataBundle,
    load_bundle,
    prepare_concrete,
    prepare_superconductivity,
)
from feature_layer import aggregation_weights, local_input_for_group  # noqa: E402
from fpwl_core import learn_partition, membership_for_new_points  # noqa: E402
from metrics_utils import make_result_row, regression_metrics, save_results  # noqa: E402
from regression.regression_config import (  # noqa: E402
    DEGREE_GRID,
    DISTANCE_ORDER_GRID,
    FUZZIFIER_GRID,
    MAX_ITER,
    RANDOM_STATE,
    RIDGE_ALPHA,
    TEST_SIZE,
    TOLERANCE,
    TRUNCATION_GRID,
    degree_name,
    get_dataset_spec,
    k_grid,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    result_file,
)

try:
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for local polynomial regression.") from exc


def load_regression_data(
    dataset: str,
    bundle_path: str | Path | None = None,
    raw_path: str | Path | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> RegressionDataBundle:
    """Load a prepared regression bundle or build one from a raw table."""
    spec = get_dataset_spec(dataset)

    if bundle_path is not None:
        return load_bundle(bundle_path)

    candidate_bundle = spec.default_bundle_path
    if candidate_bundle.exists() and raw_path is None:
        return load_bundle(candidate_bundle)

    table_path = Path(raw_path) if raw_path is not None else spec.default_raw_path
    if dataset == "concrete":
        return prepare_concrete(table_path, test_size=test_size, random_state=random_state)
    if dataset == "superconductivity":
        return prepare_superconductivity(table_path, test_size=test_size, random_state=random_state)
    raise KeyError(f"Unsupported dataset: {dataset}")


def build_local_regressor(degree: int, alpha: float = RIDGE_ALPHA) -> Pipeline:
    """Create a local polynomial ridge model."""
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("model", Ridge(alpha=alpha, fit_intercept=False)),
        ]
    )


def _fit_group_model(
    x_group: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    degree: int,
    alpha: float,
) -> Pipeline:
    """Fit one group-specific polynomial ridge model."""
    model = build_local_regressor(degree=degree, alpha=alpha)
    weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    if np.sum(weight) <= 0:
        weight = np.ones_like(weight)
    model.fit(x_group, y, model__sample_weight=weight)
    return model


def fit_local_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    membership_train: np.ndarray,
    partition: str,
    degree: int,
    f: float,
    alpha: float = RIDGE_ALPHA,
) -> list[Pipeline]:
    """Fit a polynomial ridge model for each local group."""
    models: list[Pipeline] = []
    n_groups = membership_train.shape[1]
    if partition == "crisp":
        sample_weights = membership_train
    else:
        sample_weights = np.power(np.maximum(membership_train, 1.0e-12), f)

    for group_index in range(n_groups):
        x_group = local_input_for_group(x_train, membership_train, group_index)
        model = _fit_group_model(
            x_group=x_group,
            y=y_train,
            sample_weight=sample_weights[:, group_index],
            degree=degree,
            alpha=alpha,
        )
        models.append(model)
    return models


def predict_local_models(
    models: list[Pipeline],
    x: np.ndarray,
    membership: np.ndarray,
    f: float,
) -> np.ndarray:
    """Aggregate predictions from all local models."""
    local_predictions = []
    for group_index, model in enumerate(models):
        x_group = local_input_for_group(x, membership, group_index)
        local_predictions.append(np.asarray(model.predict(x_group), dtype=float).reshape(-1))

    local_matrix = np.column_stack(local_predictions)
    weights = aggregation_weights(membership, f=f)
    return np.sum(weights * local_matrix, axis=1)


def fit_and_evaluate_configuration(
    bundle: RegressionDataBundle,
    dataset: str,
    partition: str,
    degree: int,
    k: int,
    f: float,
    p: float,
    truncation: str,
    alpha: float = RIDGE_ALPHA,
    max_iter: int = MAX_ITER,
    tolerance: float = TOLERANCE,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Fit and evaluate one local-regression configuration."""
    start = time.perf_counter()

    partition_result = learn_partition(
        bundle.x_train,
        k=k,
        partition=partition,
        f=f,
        p=p,
        truncation=truncation,
        tolerance=tolerance,
        max_iter=max_iter,
        random_state=random_state,
    )
    membership_train = partition_result.membership
    membership_test, _, _ = membership_for_new_points(
        bundle.x_test,
        partition_result.centroids,
        partition=partition,
        f=f,
        p=p,
    )

    models = fit_local_models(
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        membership_train=membership_train,
        partition=partition,
        degree=degree,
        f=f,
        alpha=alpha,
    )
    prediction = predict_local_models(models, bundle.x_test, membership_test, f=f)
    elapsed = time.perf_counter() - start

    metrics = regression_metrics(bundle.y_test, prediction)
    metadata = {
        "dataset": dataset,
        "target": bundle.target_name,
        "method_family": "local_polynomial_regression",
        "method_name": f"{partition}_local_{degree_name(degree)}_{truncation}",
        "degree": int(degree),
        "k": int(k),
        "f": float(f) if partition == "fuzzy" else np.nan,
        "p": float(p),
        "partition": partition,
        "truncation": truncation,
        "random_state": int(random_state),
        "n_train": int(bundle.x_train.shape[0]),
        "n_test": int(bundle.x_test.shape[0]),
        "n_features": int(bundle.x_train.shape[1]),
        "iterations": int(partition_result.n_iter),
        "converged": bool(partition_result.converged),
        "runtime_sec": float(elapsed),
    }
    return make_result_row(metadata, metrics)


def configuration_grid(
    dataset: str,
    partitions: Iterable[str],
    degrees: Iterable[int],
    truncations: Iterable[str],
    k_values: Iterable[int] | None = None,
    f_values: Iterable[float] | None = None,
    p_values: Iterable[float] | None = None,
    grid_mode: str = "quick",
) -> list[tuple[str, int, str, int, float, float]]:
    """Build local-regression configurations."""
    if k_values is None:
        k_values = k_grid(dataset, mode=grid_mode)
    if f_values is None:
        f_values = FUZZIFIER_GRID
    if p_values is None:
        p_values = DISTANCE_ORDER_GRID

    rows: list[tuple[str, int, str, int, float, float]] = []
    for partition, degree, truncation, k, p in product(partitions, degrees, truncations, k_values, p_values):
        if partition == "crisp":
            rows.append((partition, int(degree), str(truncation), int(k), 2.0, float(p)))
        else:
            for f in f_values:
                rows.append((partition, int(degree), str(truncation), int(k), float(f), float(p)))
    return rows


def run_local_regression_models(
    dataset: str,
    partitions: Iterable[str] = ("crisp", "fuzzy"),
    degrees: Iterable[int] = DEGREE_GRID,
    truncations: Iterable[str] = TRUNCATION_GRID,
    k_values: Iterable[int] | None = None,
    f_values: Iterable[float] | None = None,
    p_values: Iterable[float] | None = None,
    grid_mode: str = "quick",
    bundle_path: str | Path | None = None,
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
    alpha: float = RIDGE_ALPHA,
    max_iter: int = MAX_ITER,
    tolerance: float = TOLERANCE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, object]]:
    """Run a grid of local polynomial regression models."""
    bundle = load_regression_data(
        dataset=dataset,
        bundle_path=bundle_path,
        raw_path=raw_path,
        test_size=test_size,
        random_state=random_state,
    )

    rows: list[dict[str, object]] = []
    configs = configuration_grid(
        dataset=dataset,
        partitions=partitions,
        degrees=degrees,
        truncations=truncations,
        k_values=k_values,
        f_values=f_values,
        p_values=p_values,
        grid_mode=grid_mode,
    )

    for partition, degree, truncation, k, f, p in configs:
        try:
            row = fit_and_evaluate_configuration(
                bundle=bundle,
                dataset=dataset,
                partition=partition,
                degree=degree,
                k=k,
                f=f,
                p=p,
                truncation=truncation,
                alpha=alpha,
                max_iter=max_iter,
                tolerance=tolerance,
                random_state=random_state,
            )
        except Exception as exc:
            row = {
                "dataset": dataset,
                "method_family": "local_polynomial_regression",
                "method_name": f"{partition}_local_{degree_name(degree)}_{truncation}",
                "degree": int(degree),
                "k": int(k),
                "f": float(f) if partition == "fuzzy" else np.nan,
                "p": float(p),
                "partition": partition,
                "truncation": truncation,
                "random_state": int(random_state),
                "status": "failed",
                "error_message": str(exc),
            }
        else:
            row["status"] = "ok"
            row["error_message"] = ""
        rows.append(row)

    if output_path is None:
        output_path = result_file(dataset, "local_regression_models.csv")
    save_results(rows, output_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run partition-weighted local regression models.")
    parser.add_argument("--dataset", choices=["concrete", "superconductivity"], required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--partitions", default="crisp,fuzzy")
    parser.add_argument("--degrees", default=None)
    parser.add_argument("--truncations", default=None)
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--f-values", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--grid-mode", choices=["quick", "paper"], default="quick")
    parser.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    rows = run_local_regression_models(
        dataset=args.dataset,
        partitions=parse_str_list(args.partitions, ("crisp", "fuzzy")),
        degrees=parse_int_list(args.degrees, DEGREE_GRID),
        truncations=parse_str_list(args.truncations, TRUNCATION_GRID),
        k_values=parse_int_list(args.k_values, []) if args.k_values else None,
        f_values=parse_float_list(args.f_values, FUZZIFIER_GRID) if args.f_values else None,
        p_values=parse_float_list(args.p_values, DISTANCE_ORDER_GRID) if args.p_values else None,
        grid_mode=args.grid_mode,
        bundle_path=args.bundle,
        raw_path=args.input,
        output_path=args.output,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"saved_rows={len(rows)}")


if __name__ == "__main__":
    main()

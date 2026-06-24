"""
Global polynomial regression baselines.

The script fits degree-one through degree-four polynomial models and records
held-out regression metrics for a selected dataset.
"""

from __future__ import annotations

import argparse
import sys
import time
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
from metrics_utils import make_result_row, regression_metrics, save_results  # noqa: E402
from regression.regression_config import (  # noqa: E402
    DEGREE_GRID,
    RANDOM_STATE,
    RIDGE_ALPHA,
    TEST_SIZE,
    degree_name,
    get_dataset_spec,
    parse_int_list,
    result_file,
)

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for the polynomial baselines.") from exc


def load_regression_data(
    dataset: str,
    bundle_path: str | Path | None = None,
    raw_path: str | Path | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> RegressionDataBundle:
    """Load a prepared bundle or build one from a raw regression table."""
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


def build_polynomial_model(degree: int, alpha: float = RIDGE_ALPHA) -> Pipeline:
    """Create a polynomial regression pipeline."""
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    if alpha > 0:
        estimator = Ridge(alpha=alpha, fit_intercept=False)
    else:
        estimator = LinearRegression(fit_intercept=False)

    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("model", estimator),
        ]
    )


def fit_one_degree(
    bundle: RegressionDataBundle,
    dataset: str,
    degree: int,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, object]:
    """Fit and evaluate one polynomial baseline."""
    start = time.perf_counter()
    model = build_polynomial_model(degree=degree, alpha=alpha)
    model.fit(bundle.x_train, bundle.y_train)
    prediction = np.asarray(model.predict(bundle.x_test), dtype=float)
    elapsed = time.perf_counter() - start

    metrics = regression_metrics(bundle.y_test, prediction)
    metadata = {
        "dataset": dataset,
        "target": bundle.target_name,
        "method_family": "global_polynomial",
        "method_name": f"global_{degree_name(degree)}",
        "degree": degree,
        "k": np.nan,
        "f": np.nan,
        "p": np.nan,
        "partition": "none",
        "truncation": "none",
        "random_state": np.nan,
        "n_train": int(bundle.x_train.shape[0]),
        "n_test": int(bundle.x_test.shape[0]),
        "n_features": int(bundle.x_train.shape[1]),
        "runtime_sec": float(elapsed),
    }
    return make_result_row(metadata, metrics)


def run_global_polynomial_baselines(
    dataset: str,
    degrees: Iterable[int] = DEGREE_GRID,
    bundle_path: str | Path | None = None,
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
    alpha: float = RIDGE_ALPHA,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, object]]:
    """Run all selected global polynomial baselines."""
    bundle = load_regression_data(
        dataset=dataset,
        bundle_path=bundle_path,
        raw_path=raw_path,
        test_size=test_size,
        random_state=random_state,
    )

    rows = [fit_one_degree(bundle, dataset=dataset, degree=int(degree), alpha=alpha) for degree in degrees]

    if output_path is None:
        output_path = result_file(dataset, "global_polynomial_baselines.csv")
    save_results(rows, output_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run global polynomial regression baselines.")
    parser.add_argument("--dataset", choices=["concrete", "superconductivity"], required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--degrees", default=None)
    parser.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    degrees = parse_int_list(args.degrees, DEGREE_GRID)
    rows = run_global_polynomial_baselines(
        dataset=args.dataset,
        degrees=degrees,
        bundle_path=args.bundle,
        raw_path=args.input,
        output_path=args.output,
        alpha=args.alpha,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

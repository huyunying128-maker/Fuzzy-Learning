"""
External regression models with the partition-weighted feature layer.

The script compares standard regressors trained on the original input with the
same regressors trained on membership-based feature representations.
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
from feature_layer import partition_weighted_feature_layer  # noqa: E402
from fpwl_core import learn_partition, membership_for_new_points  # noqa: E402
from metrics_utils import make_result_row, percentage_change, regression_metrics, save_results  # noqa: E402
from regression.regression_config import (  # noqa: E402
    MAX_ITER,
    RANDOM_STATE,
    TEST_SIZE,
    TOLERANCE,
    get_dataset_spec,
    parse_str_list,
    result_file,
)

try:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for external regression baselines.") from exc


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


def build_regressors(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Create the external regression models used in the comparison."""
    models: dict[str, object] = {
        "ANN": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64,),
                        activation="relu",
                        solver="adam",
                        alpha=1.0e-4,
                        learning_rate_init=1.0e-3,
                        max_iter=400,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Adam ANN": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1.0e-4,
                        learning_rate_init=5.0e-4,
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "SVR": Pipeline(steps=[("scale", StandardScaler()), ("model", SVR(C=10.0, epsilon=0.1))]),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(random_state=random_state),
    }

    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception:
        return models

    models["XGBoost"] = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    return models


def default_partition_k(dataset: str) -> int:
    """Return a compact default partition size for the feature-layer experiment."""
    if dataset == "concrete":
        return 36
    if dataset == "superconductivity":
        return 200
    raise KeyError(f"Unsupported dataset: {dataset}")


def fit_predict_model(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Fit one model and return test predictions."""
    fitted = clone(model)
    fitted.fit(x_train, y_train)
    return np.asarray(fitted.predict(x_test), dtype=float).reshape(-1)


def evaluate_original_and_pw(
    bundle: RegressionDataBundle,
    dataset: str,
    model_name: str,
    model: object,
    x_train_pw: np.ndarray,
    x_test_pw: np.ndarray,
    partition_metadata: dict[str, object],
) -> list[dict[str, object]]:
    """Evaluate one external model on original and partition-weighted inputs."""
    rows: list[dict[str, object]] = []

    start = time.perf_counter()
    pred_original = fit_predict_model(model, bundle.x_train, bundle.y_train, bundle.x_test)
    original_time = time.perf_counter() - start
    original_metrics = regression_metrics(bundle.y_test, pred_original)

    start = time.perf_counter()
    pred_pw = fit_predict_model(model, x_train_pw, bundle.y_train, x_test_pw)
    pw_time = time.perf_counter() - start
    pw_metrics = regression_metrics(bundle.y_test, pred_pw)

    base_metadata = {
        "dataset": dataset,
        "target": bundle.target_name,
        "method_family": "external_regressor",
        "model": model_name,
        "n_train": int(bundle.x_train.shape[0]),
        "n_test": int(bundle.x_test.shape[0]),
        "n_features_original": int(bundle.x_train.shape[1]),
        "n_features_pw": int(x_train_pw.shape[1]),
        **partition_metadata,
    }

    rows.append(
        make_result_row(
            {
                **base_metadata,
                "input_layer": "original",
                "method_name": f"{model_name}_original",
                "runtime_sec": float(original_time),
                "mse_reduction_percent": np.nan,
            },
            original_metrics,
        )
    )
    rows.append(
        make_result_row(
            {
                **base_metadata,
                "input_layer": "partition_weighted",
                "method_name": f"{model_name}_pw",
                "runtime_sec": float(pw_time),
                "mse_reduction_percent": -percentage_change(original_metrics["mse"], pw_metrics["mse"]),
            },
            pw_metrics,
        )
    )
    return rows


def build_partition_weighted_data(
    bundle: RegressionDataBundle,
    k: int,
    f: float,
    p: float,
    partition: str,
    truncation: str,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Learn a partition and construct partition-weighted train-test features."""
    result = learn_partition(
        bundle.x_train,
        k=k,
        partition=partition,
        f=f,
        p=p,
        truncation=truncation,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
    )
    membership_train = result.membership
    membership_test, _, _ = membership_for_new_points(
        bundle.x_test,
        result.centroids,
        partition=partition,
        f=f,
        p=p,
    )

    x_train_pw = partition_weighted_feature_layer(bundle.x_train, membership_train, f=f)
    x_test_pw = partition_weighted_feature_layer(bundle.x_test, membership_test, f=f)

    metadata = {
        "partition": partition,
        "truncation": truncation,
        "k": int(k),
        "f": float(f) if partition == "fuzzy" else np.nan,
        "p": float(p),
        "iterations": int(result.n_iter),
        "converged": bool(result.converged),
        "random_state": int(random_state),
    }
    return x_train_pw, x_test_pw, metadata


def run_pw_regression_baselines(
    dataset: str,
    models: Iterable[str] | None = None,
    bundle_path: str | Path | None = None,
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
    partition: str = "fuzzy",
    truncation: str = "hpd",
    k: int | None = None,
    f: float = 1.20,
    p: float = 1.25,
    max_iter: int = MAX_ITER,
    tolerance: float = TOLERANCE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, object]]:
    """Run external regression baselines with and without the feature layer."""
    bundle = load_regression_data(
        dataset=dataset,
        bundle_path=bundle_path,
        raw_path=raw_path,
        test_size=test_size,
        random_state=random_state,
    )
    if k is None:
        k = default_partition_k(dataset)

    x_train_pw, x_test_pw, partition_metadata = build_partition_weighted_data(
        bundle=bundle,
        k=k,
        f=f,
        p=p,
        partition=partition,
        truncation=truncation,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
    )

    available = build_regressors(random_state=random_state)
    if models is None:
        selected_names = list(available)
    else:
        lookup = {name.lower(): name for name in available}
        selected_names = [lookup[str(name).lower()] for name in models if str(name).lower() in lookup]

    rows: list[dict[str, object]] = []
    for model_name in selected_names:
        model = available[model_name]
        try:
            model_rows = evaluate_original_and_pw(
                bundle=bundle,
                dataset=dataset,
                model_name=model_name,
                model=model,
                x_train_pw=x_train_pw,
                x_test_pw=x_test_pw,
                partition_metadata=partition_metadata,
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "method_family": "external_regressor",
                    "model": model_name,
                    "status": "failed",
                    "error_message": str(exc),
                    **partition_metadata,
                }
            )
        else:
            for row in model_rows:
                row["status"] = "ok"
                row["error_message"] = ""
            rows.extend(model_rows)

    if output_path is None:
        output_path = result_file(dataset, "pw_regression_baselines.csv")
    save_results(rows, output_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run partition-weighted external regression baselines.")
    parser.add_argument("--dataset", choices=["concrete", "superconductivity"], required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--partition", choices=["crisp", "fuzzy"], default="fuzzy")
    parser.add_argument("--truncation", default="hpd")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--f", type=float, default=1.20)
    parser.add_argument("--p", type=float, default=1.25)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    rows = run_pw_regression_baselines(
        dataset=args.dataset,
        models=parse_str_list(args.models, ()) if args.models else None,
        bundle_path=args.bundle,
        raw_path=args.input,
        output_path=args.output,
        partition=args.partition,
        truncation=args.truncation,
        k=args.k,
        f=args.f,
        p=args.p,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"saved_rows={len(rows)}")


if __name__ == "__main__":
    main()

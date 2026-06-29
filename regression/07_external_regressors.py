"""
External regression models with original, k-means, and partition-weighted inputs.

The external comparison sends three matched feature sets to the same family of
regressors: the original input, a modified k-means feature input, and the full
partition-weighted feature layer. The output format follows the regression
metrics reported in the experiment tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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
_modified = _load_core_module("08_modified_kmeans.py", "fpw_modified_kmeans")
_fuzzy = _load_core_module("07_fuzzy_partition.py", "fpw_fuzzy_partition")
_feature = _load_core_module("11_partition_feature_layer.py", "fpw_partition_feature_layer")


@dataclass(frozen=True)
class FeatureBundle:
    """Original, k-means, and partition-weighted feature matrices."""

    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    km_train: np.ndarray
    km_valid: np.ndarray
    km_test: np.ndarray
    pw_train: np.ndarray
    pw_valid: np.ndarray
    pw_test: np.ndarray
    km_k: int
    km_p: float
    km_f: float
    pw_k: int
    pw_p: float
    pw_f: float
    pw_iter: int


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


def parse_model_names(text: str) -> List[str]:
    """Parse the list of external regressors to evaluate."""
    names = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one external regressor is required.")
    return names


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
        raise ValueError("The external-model feature grid must contain feasible k, p, and f values.")
    return k_values, p_values, f_values


def _fit_simple_ridge_score(x_train, y_train, x_valid, y_valid) -> float:
    """Score a feature layer with a small validation model."""
    from sklearn.linear_model import Ridge

    model = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1e-3))
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    return float(_metrics.regression_metrics(y_valid, pred)["mse"])


def _select_kmeans_f_for_features(
    hard_result,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    f_values: Sequence[float],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Choose the post-partition fuzzy degree for the k-means feature input."""
    best_f: Optional[float] = None
    best_score: Optional[float] = None
    best_train_layer: Optional[np.ndarray] = None
    best_valid_layer: Optional[np.ndarray] = None

    for f in f_values:
        result_f = _modified.add_distance_based_fuzzy_layer(hard_result, f=float(f))
        train_membership = result_f.fuzzy_membership
        valid_membership, _, _ = _modified.transform_with_modified_kmeans(x_valid, result_f, use_fuzzy=True)
        train_layer = _feature.build_partition_feature_layer(x_train, train_membership)
        valid_layer = _feature.build_partition_feature_layer(x_valid, valid_membership)
        score = _fit_simple_ridge_score(train_layer, y_train, valid_layer, y_valid)
        if best_score is None or score < best_score:
            best_score = score
            best_f = float(f)
            best_train_layer = train_layer
            best_valid_layer = valid_layer

    if best_f is None or best_train_layer is None or best_valid_layer is None:
        raise RuntimeError("No k-means fuzzy feature candidate was evaluated.")
    return best_f, best_train_layer, best_valid_layer


def _select_fuzzy_partition_for_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    k_values: Sequence[int],
    p_values: Sequence[float],
    f_values: Sequence[float],
    max_iter: int,
    tolerance: float,
    random_state: int,
    max_candidates: Optional[int],
):
    """Select a fuzzy partition for the external partition-weighted feature layer."""
    candidates = [(k, f, p) for k in k_values for f in f_values for p in p_values]
    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]

    best = None
    best_score: Optional[float] = None
    best_train_layer: Optional[np.ndarray] = None
    best_valid_layer: Optional[np.ndarray] = None
    for run_index, (k, f, p) in enumerate(candidates):
        partition = _fuzzy.fit_fuzzy_partition(
            x_train,
            k=int(k),
            f=float(f),
            p=float(p),
            max_iter=max_iter,
            tolerance=tolerance,
            truncation_rule="hpd",
            random_state=random_state + run_index,
        )
        valid_membership, _, _ = _fuzzy.predict_fuzzy_partition(x_valid, partition.centroids, f=float(f), p=float(p))
        train_layer = _feature.build_partition_feature_layer(x_train, partition.membership)
        valid_layer = _feature.build_partition_feature_layer(x_valid, valid_membership)
        score = _fit_simple_ridge_score(train_layer, y_train, valid_layer, y_valid)
        if best_score is None or score < best_score:
            best_score = score
            best = partition
            best_train_layer = train_layer
            best_valid_layer = valid_layer

    if best is None or best_train_layer is None or best_valid_layer is None:
        raise RuntimeError("No fuzzy partition-weighted feature candidate was evaluated.")
    return best, best_train_layer, best_valid_layer


def build_feature_bundle(
    split: Dict[str, np.ndarray],
    k_values_text: Optional[str] = None,
    p_values_text: Optional[str] = None,
    f_values_text: Optional[str] = None,
    max_iter: int = _config.MAX_ITERATIONS,
    tolerance: float = _config.TOLERANCE,
    random_state: int = _config.RANDOM_STATE,
    max_candidates: Optional[int] = None,
) -> FeatureBundle:
    """Construct the matched original, k-means, and partition-weighted inputs."""
    k_values, p_values, f_values = _candidate_grid(
        split["x_train"].shape[0],
        k_values_text,
        p_values_text,
        f_values_text,
    )

    hard_result = _modified.search_hard_kp_reference(
        split["x_train"],
        k_values=k_values,
        p_values=p_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
    )
    km_f, km_train, km_valid = _select_kmeans_f_for_features(
        hard_result,
        split["x_train"],
        split["y_train"],
        split["x_valid"],
        split["y_valid"],
        f_values=f_values,
    )
    hard_with_f = _modified.add_distance_based_fuzzy_layer(hard_result, f=km_f)
    km_test_membership, _, _ = _modified.transform_with_modified_kmeans(split["x_test"], hard_with_f, use_fuzzy=True)
    km_test = _feature.build_partition_feature_layer(split["x_test"], km_test_membership)

    fuzzy_partition, pw_train, pw_valid = _select_fuzzy_partition_for_features(
        split["x_train"],
        split["y_train"],
        split["x_valid"],
        split["y_valid"],
        k_values=k_values,
        p_values=p_values,
        f_values=f_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
    )
    pw_test_membership, _, _ = _fuzzy.predict_fuzzy_partition(
        split["x_test"],
        fuzzy_partition.centroids,
        f=fuzzy_partition.f,
        p=fuzzy_partition.p,
    )
    pw_test = _feature.build_partition_feature_layer(split["x_test"], pw_test_membership)

    return FeatureBundle(
        x_train=split["x_train"],
        x_valid=split["x_valid"],
        x_test=split["x_test"],
        km_train=km_train,
        km_valid=km_valid,
        km_test=km_test,
        pw_train=pw_train,
        pw_valid=pw_valid,
        pw_test=pw_test,
        km_k=int(hard_result.k),
        km_p=float(hard_result.p),
        km_f=float(km_f),
        pw_k=int(fuzzy_partition.centroids.shape[0]),
        pw_p=float(fuzzy_partition.p),
        pw_f=float(fuzzy_partition.f),
        pw_iter=int(fuzzy_partition.n_iter),
    )


def _xgboost_or_fallback(random_state: int):
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=random_state)


def make_regressor(model_name: str, random_state: int):
    """Create one external regressor by name."""
    name = model_name.lower().strip()
    if name == "ann":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128,), activation="relu", max_iter=400, random_state=random_state),
        )
    if name in {"adam_ann", "adam ann"}:
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), solver="adam", max_iter=500, random_state=random_state),
        )
    if name in {"dl", "deep_learning", "deep learning"}:
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", max_iter=500, random_state=random_state),
        )
    if name in {"cnn", "cnn1d", "cnn(1d)"}:
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(192, 96), activation="relu", max_iter=500, random_state=random_state),
        )
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=0.05))
    if name in {"random_forest", "random forest", "rf"}:
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=random_state, n_jobs=-1)
    if name in {"xgboost", "xgb"}:
        return _xgboost_or_fallback(random_state=random_state)
    raise ValueError(f"Unsupported external regressor: {model_name}")


def display_model_name(model_name: str) -> str:
    """Return the table label used for an external regressor."""
    mapping = {
        "ann": "ANN",
        "adam_ann": "Adam ANN",
        "adam ann": "Adam ANN",
        "dl": "DL",
        "deep_learning": "Deep learning",
        "deep learning": "Deep learning",
        "cnn": "CNN(1D)",
        "cnn1d": "CNN(1D)",
        "cnn(1d)": "CNN(1D)",
        "svr": "SVR",
        "random_forest": "Random forest",
        "random forest": "Random forest",
        "rf": "Random forest",
        "xgboost": "XGBoost",
        "xgb": "XGBoost",
    }
    return mapping.get(model_name.lower().strip(), model_name)


def evaluate_regressor(model_name: str, x_train, y_train, x_test, y_test, random_state: int) -> Dict[str, float]:
    """Fit one external regressor and report test metrics."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_regressor(model_name, random_state=random_state)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
    metrics = _metrics.regression_metrics(y_test, y_pred)
    return {"runtime_sec": elapsed, **metrics}


def run_external_regressors(
    split_path: Path,
    dataset_name: str,
    output_path: Path,
    model_names: Sequence[str],
    k_values_text: Optional[str] = None,
    p_values_text: Optional[str] = None,
    f_values_text: Optional[str] = None,
    max_iter: int = _config.MAX_ITERATIONS,
    tolerance: float = _config.TOLERANCE,
    random_state: int = _config.RANDOM_STATE,
    max_candidates: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate external regressors on original, k-means, and PW feature sets."""
    split = load_split_npz(split_path)
    bundle = build_feature_bundle(
        split,
        k_values_text=k_values_text,
        p_values_text=p_values_text,
        f_values_text=f_values_text,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
    )

    rows: List[Dict[str, float]] = []
    for model_index, model_name in enumerate(model_names):
        row: Dict[str, float] = {
            "dataset": dataset_name,
            "task": "regression",
            "method_family": "external_regressor",
            "method_name": display_model_name(model_name),
            "km_k": bundle.km_k,
            "km_p": bundle.km_p,
            "km_f": bundle.km_f,
            "pw_k": bundle.pw_k,
            "pw_p": bundle.pw_p,
            "pw_f": bundle.pw_f,
            "pw_iter": bundle.pw_iter,
            "n_train": int(split["x_train"].shape[0]),
            "n_valid": int(split["x_valid"].shape[0]),
            "n_test": int(split["x_test"].shape[0]),
            "n_features": int(split["x_train"].shape[1]),
        }
        feature_sets = {
            "orig": (bundle.x_train, bundle.x_test),
            "km": (bundle.km_train, bundle.km_test),
            "pw": (bundle.pw_train, bundle.pw_test),
        }
        for label, (x_train, x_test) in feature_sets.items():
            metrics = evaluate_regressor(
                model_name,
                x_train,
                split["y_train"],
                x_test,
                split["y_test"],
                random_state=random_state + model_index,
            )
            row[f"{label}_mse"] = float(metrics["mse"])
            row[f"{label}_rmse"] = float(metrics["rmse"])
            row[f"{label}_mae"] = float(metrics["mae"])
            row[f"{label}_r2"] = float(metrics["r2"])
            row[f"{label}_runtime_sec"] = float(metrics["runtime_sec"])
        rows.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit external regression baselines with partition feature layers.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/regression/external_regressors.csv"))
    parser.add_argument("--models", default="ann,adam_ann,dl,cnn1d,svr,random_forest,xgboost")
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
    table = run_external_regressors(
        split_path=args.split,
        dataset_name=args.dataset_name,
        output_path=args.output,
        model_names=parse_model_names(args.models),
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        f_values_text=args.f_values,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        random_state=args.random_state,
        max_candidates=args.max_candidates,
    )
    print(f"Saved {table.shape[0]} external regression rows to {args.output}")


if __name__ == "__main__":
    main()

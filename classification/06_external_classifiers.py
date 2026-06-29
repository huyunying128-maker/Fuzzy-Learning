"""
External classifiers for the MNIST original, k-means, and PW feature inputs.

The matched comparison fits the same classifier family to three input views:
raw image pixels, a modified k-means feature layer, and a fuzzy
partition-weighted feature layer. The output table reports accuracy, cross
entropy, and runtime for each view.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


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


_metrics = _load_core_module("02_metrics.py", "fpw_external_classifier_metrics")


def parse_models(text: str) -> List[str]:
    """Parse a comma-separated list of classifier names."""
    values = [part.strip().lower() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one classifier must be provided.")
    return values


def load_feature_layers(feature_path: Path) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Load original, modified k-means, and partition-weighted feature views."""
    data = np.load(Path(feature_path), allow_pickle=True)
    required = ("x_train", "x_test", "y_train", "y_test", "km_train", "km_test", "pw_train", "pw_test")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing arrays in feature file: {missing}")
    views = {
        "original": (np.asarray(data["x_train"], dtype=float), np.asarray(data["x_test"], dtype=float)),
        "modified_kmeans": (np.asarray(data["km_train"], dtype=float), np.asarray(data["km_test"], dtype=float)),
        "partition_weighted": (np.asarray(data["pw_train"], dtype=float), np.asarray(data["pw_test"], dtype=float)),
    }
    return views, np.asarray(data["y_train"], dtype=int), np.asarray(data["y_test"], dtype=int)


def maybe_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a reproducible training subset when a sample cap is provided."""
    if max_samples is None or max_samples >= X.shape[0]:
        return X, y
    if max_samples < 1:
        raise ValueError("max_samples must be positive when provided.")
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], size=int(max_samples), replace=False)
    return X[indices], y[indices]


def aligned_probabilities_from_model(model: object, X: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Return an n by 10 probability matrix from a fitted classifier."""
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        classes = getattr(model, "classes_", np.arange(raw.shape[1]))
        probabilities = np.full((X.shape[0], n_classes), EPSILON, dtype=float)
        for source_index, cls in enumerate(classes):
            probabilities[:, int(cls)] = raw[:, source_index]
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        return probabilities

    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
        if logits.ndim == 1:
            logits = np.column_stack([-logits, logits])
        aligned = np.zeros((X.shape[0], n_classes), dtype=float)
        classes = getattr(model, "classes_", np.arange(logits.shape[1]))
        for source_index, cls in enumerate(classes):
            aligned[:, int(cls)] = logits[:, source_index]
        return _metrics.softmax(aligned)

    labels = np.asarray(model.predict(X), dtype=int)
    probabilities = np.full((labels.shape[0], n_classes), EPSILON, dtype=float)
    probabilities[np.arange(labels.shape[0]), labels] = 1.0 - EPSILON * (n_classes - 1)
    return probabilities


def fit_sklearn_classifier(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    max_iter: int,
) -> object:
    """Create and fit one scikit-learn classifier family."""
    if model_name == "ann":
        model = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.01,
            max_iter=max_iter,
            random_state=random_state,
        )
    elif model_name == "adam_ann":
        model = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=random_state,
        )
    elif model_name == "deep_learning":
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=random_state,
        )
    elif model_name == "svm":
        model = SVC(C=5.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=random_state,
            )
        except Exception:
            model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=random_state,
            )
    elif model_name == "cnn_1d":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown classifier: {model_name}")
    model.fit(X_train, y_train)
    return model


def try_fit_tensorflow_classifier(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> Optional[np.ndarray]:
    """Fit TensorFlow CNN or dense models when TensorFlow is available."""
    if model_name not in {"cnn_1d", "deep_learning"}:
        return None
    try:
        import tensorflow as tf
    except Exception:
        return None

    tf.keras.utils.set_random_seed(int(random_state))
    n_classes = 10
    if model_name == "cnn_1d":
        train_input = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        test_input = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
                tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )
    else:
        train_input = X_train
        test_input = X_test
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_input, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)
    return np.asarray(model.predict(test_input, verbose=0), dtype=float)


def fit_and_score_classifier(
    model_name: str,
    input_view: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    max_train_samples: Optional[int],
    max_iter: int,
    tensorflow_epochs: int,
    tensorflow_batch_size: int,
) -> Dict[str, float]:
    """Fit one classifier on one input view and return a result row."""
    X_fit, y_fit = maybe_subsample(X_train, y_train, max_train_samples, random_state=random_state)
    start = time.perf_counter()
    probabilities = try_fit_tensorflow_classifier(
        model_name,
        X_fit,
        y_fit,
        X_test,
        epochs=tensorflow_epochs,
        batch_size=tensorflow_batch_size,
        random_state=random_state,
    )
    backend = "tensorflow" if probabilities is not None else "sklearn"
    if probabilities is None:
        model = fit_sklearn_classifier(
            model_name,
            X_fit,
            y_fit,
            random_state=random_state,
            max_iter=max_iter,
        )
        probabilities = aligned_probabilities_from_model(model, X_test, n_classes=10)

    metrics = _metrics.classification_metrics(y_test, probabilities=probabilities)
    elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())
    display_names = {
        "ann": "ANN",
        "adam_ann": "Adam ANN",
        "cnn_1d": "CNN(1D)",
        "deep_learning": "Deep learning",
        "svm": "SVM",
        "random_forest": "Random forest",
        "xgboost": "XGBoost",
    }
    return {
        "dataset": "mnist",
        "task": "classification",
        "classifier": display_names.get(model_name, model_name),
        "model_key": model_name,
        "input_view": input_view,
        "backend": backend,
        "n_train": int(X_fit.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_cross_entropy": float(metrics["cross_entropy"]),
        "runtime_sec": elapsed,
    }


def run_external_classifiers(
    feature_path: Path,
    output_path: Path = OUTPUT_DIR / "06_external_classifiers.csv",
    models: Sequence[str] = ("ann", "adam_ann", "cnn_1d", "deep_learning", "svm", "random_forest", "xgboost"),
    views: Sequence[str] = ("original", "modified_kmeans", "partition_weighted"),
    random_state: int = 42,
    max_train_samples: Optional[int] = None,
    max_iter: int = 100,
    tensorflow_epochs: int = 5,
    tensorflow_batch_size: int = 128,
) -> pd.DataFrame:
    """Fit external classifiers to the matched MNIST input views."""
    feature_views, y_train, y_test = load_feature_layers(feature_path)
    records: List[Dict[str, float]] = []
    for model_index, model_name in enumerate(models):
        for view_name in views:
            if view_name not in feature_views:
                raise ValueError(f"Unknown input view: {view_name}")
            X_train, X_test = feature_views[view_name]
            row = fit_and_score_classifier(
                model_name=model_name,
                input_view=view_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                random_state=random_state + model_index,
                max_train_samples=max_train_samples,
                max_iter=max_iter,
                tensorflow_epochs=tensorflow_epochs,
                tensorflow_batch_size=tensorflow_batch_size,
            )
            records.append(row)

    table = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MNIST external classifiers on original, KM, and PW inputs.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "06_external_classifiers.csv")
    parser.add_argument("--models", default="ann,adam_ann,cnn_1d,deep_learning,svm,random_forest,xgboost")
    parser.add_argument("--views", default="original,modified_kmeans,partition_weighted")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tensorflow-epochs", type=int, default=5)
    parser.add_argument("--tensorflow-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_external_classifiers(
        feature_path=args.features,
        output_path=args.output,
        models=parse_models(args.models),
        views=parse_models(args.views),
        random_state=args.random_state,
        max_train_samples=args.max_train_samples,
        max_iter=args.max_iter,
        tensorflow_epochs=args.tensorflow_epochs,
        tensorflow_batch_size=args.tensorflow_batch_size,
    )
    print(f"Saved external-classifier table with {table.shape[0]} rows")


if __name__ == "__main__":
    main()

"""
Create matched external-model comparison figures.

The external-model result layout compares original input, modified k-means
feature input, and partition-weighted feature input for regression and MNIST
classification. This module reads the saved external-model CSV files and writes
compact long-form tables and matched bar figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGRESSION_ROOT = PROJECT_ROOT / "outputs" / "regression"
DEFAULT_MNIST_ROOT = PROJECT_ROOT / "outputs" / "classification" / "mnist"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "external_models"

REGRESSION_VIEW_LABELS = {"orig": "Original", "km": "Modified k-means", "pw": "Partition-weighted"}
MNIST_VIEW_LABELS = {"original": "Original", "modified_kmeans": "Modified k-means", "partition_weighted": "Partition-weighted"}


def _first_column(table: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in table.columns:
            return name
    return None


def _as_float(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _method_column(table: pd.DataFrame) -> str:
    column = _first_column(table, ["model", "model_name", "method", "classifier"])
    if column is None:
        raise ValueError("The external-model table does not contain a model or method column.")
    return column


def find_regression_external_files(root: Path = DEFAULT_REGRESSION_ROOT) -> List[Path]:
    """Find external-regression result tables under the regression output root."""
    root = Path(root)
    patterns = ["*/external_regressors.csv", "*/*external*.csv", "external_regressors.csv"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(path for path in files if path.exists()))


def normalize_regression_external(table: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Convert a wide regression external table to a long input-view table."""
    table = table.copy()
    method_col = _method_column(table)
    rows: List[Dict[str, object]] = []
    for _, row in table.iterrows():
        model = row[method_col]
        for view, view_label in REGRESSION_VIEW_LABELS.items():
            mse_col = f"{view}_mse"
            mae_col = f"{view}_mae"
            r2_col = f"{view}_r2"
            time_col = f"{view}_runtime_sec"
            if mse_col not in table.columns:
                continue
            rows.append({
                "dataset": dataset_name,
                "task": "regression",
                "model": model,
                "input_view": view_label,
                "mse": row.get(mse_col, np.nan),
                "mae": row.get(mae_col, np.nan),
                "r2": row.get(r2_col, np.nan),
                "runtime_sec": row.get(time_col, np.nan),
            })
    out = pd.DataFrame(rows)
    for column in ["mse", "mae", "r2", "runtime_sec"]:
        if column in out.columns:
            out[column] = _as_float(out[column])
    return out


def load_regression_external(root: Path = DEFAULT_REGRESSION_ROOT) -> pd.DataFrame:
    """Load all available external-regression result files."""
    files = find_regression_external_files(root)
    frames: List[pd.DataFrame] = []
    for path in files:
        dataset_name = path.parent.name.lower()
        if "concrete" not in dataset_name and "superconduct" not in dataset_name:
            dataset_col = None
        else:
            dataset_col = dataset_name
        table = pd.read_csv(path)
        if dataset_col is None:
            candidate = _first_column(table, ["dataset", "dataset_name"])
            dataset_name = str(table[candidate].iloc[0]).lower() if candidate else path.stem
        frames.append(normalize_regression_external(table, dataset_name))
    if not frames:
        raise FileNotFoundError(f"No external-regression CSV files were found under {root}.")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_mnist_external(root: Path = DEFAULT_MNIST_ROOT) -> pd.DataFrame:
    """Load the MNIST external-classifier table."""
    candidates = [Path(root) / "06_external_classifiers.csv", Path(root) / "mnist_external_classifiers.csv"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No MNIST external-classifier CSV file was found under {root}.")
    table = pd.read_csv(path)
    method_col = _method_column(table)
    view_col = _first_column(table, ["input_view", "view", "feature_view"])
    acc_col = _first_column(table, ["accuracy", "test_accuracy", "acc"])
    ce_col = _first_column(table, ["cross_entropy", "test_cross_entropy", "ce"])
    time_col = _first_column(table, ["runtime_sec", "time_sec", "sec"])
    if view_col is None or acc_col is None:
        raise ValueError("The MNIST external-classifier table needs input_view and accuracy columns.")

    out = pd.DataFrame()
    out["dataset"] = "mnist"
    out["task"] = "classification"
    out["model"] = table[method_col].astype(str)
    raw_view = table[view_col].astype(str).str.lower()
    out["input_view"] = raw_view.map(MNIST_VIEW_LABELS).fillna(table[view_col].astype(str))
    out["accuracy"] = _as_float(table[acc_col])
    out["cross_entropy"] = _as_float(table[ce_col]) if ce_col else np.nan
    out["runtime_sec"] = _as_float(table[time_col]) if time_col else np.nan
    return out


def _ordered_views(table: pd.DataFrame) -> List[str]:
    order = ["Original", "Modified k-means", "Partition-weighted"]
    present = [view for view in order if view in set(table["input_view"])]
    extra = [view for view in table["input_view"].dropna().unique() if view not in present]
    return present + list(extra)


def plot_regression_external(table: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Create one MSE comparison figure for each regression dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for dataset, subset in table.groupby("dataset"):
        pivot = subset.pivot_table(index="model", columns="input_view", values="mse", aggfunc="min")
        view_order = [view for view in _ordered_views(subset) if view in pivot.columns]
        pivot = pivot[view_order].sort_values(view_order[-1] if view_order else pivot.columns[0])
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("External regressor")
        ax.set_ylabel("Test MSE")
        ax.set_title(f"{dataset.title()} external-regression comparison")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        path = output_dir / f"05_{dataset}_external_regression_mse.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths[f"{dataset}_regression_mse"] = path
    return paths


def plot_mnist_external(table: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Create MNIST accuracy and cross-entropy comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    view_order = _ordered_views(table)

    accuracy = table.pivot_table(index="model", columns="input_view", values="accuracy", aggfunc="max")
    accuracy = accuracy[[view for view in view_order if view in accuracy.columns]]
    accuracy = accuracy.sort_values(accuracy.columns[-1], ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    accuracy.plot(kind="bar", ax=ax)
    ax.set_xlabel("External classifier")
    ax.set_ylabel("Accuracy")
    ax.set_title("MNIST matched external-classifier comparison")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    paths["mnist_accuracy"] = output_dir / "05_mnist_external_accuracy.png"
    fig.savefig(paths["mnist_accuracy"], dpi=300)
    plt.close(fig)

    if table["cross_entropy"].notna().any():
        ce = table.pivot_table(index="model", columns="input_view", values="cross_entropy", aggfunc="min")
        ce = ce[[view for view in view_order if view in ce.columns]]
        ce = ce.sort_values(ce.columns[-1])
        fig, ax = plt.subplots(figsize=(10, 5))
        ce.plot(kind="bar", ax=ax)
        ax.set_xlabel("External classifier")
        ax.set_ylabel("Cross entropy")
        ax.set_title("MNIST external-classifier cross entropy")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        paths["mnist_cross_entropy"] = output_dir / "05_mnist_external_cross_entropy.png"
        fig.savefig(paths["mnist_cross_entropy"], dpi=300)
        plt.close(fig)
    return paths


def create_external_model_figures(
    regression_root: Path = DEFAULT_REGRESSION_ROOT,
    mnist_root: Path = DEFAULT_MNIST_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Path]:
    """Create CSV summaries and figures for external-model comparisons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    try:
        regression = load_regression_external(regression_root)
        regression_path = output_dir / "05_external_regression_long.csv"
        regression.to_csv(regression_path, index=False)
        paths["external_regression_table"] = regression_path
        paths.update(plot_regression_external(regression, output_dir))
    except FileNotFoundError:
        pass

    try:
        mnist = load_mnist_external(mnist_root)
        mnist_path = output_dir / "05_external_mnist_long.csv"
        mnist.to_csv(mnist_path, index=False)
        paths["external_mnist_table"] = mnist_path
        paths.update(plot_mnist_external(mnist, output_dir))
    except FileNotFoundError:
        pass

    if not paths:
        raise FileNotFoundError("No external-model result files were available for plotting.")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create external-model comparison figures.")
    parser.add_argument("--regression-root", type=Path, default=DEFAULT_REGRESSION_ROOT)
    parser.add_argument("--mnist-root", type=Path, default=DEFAULT_MNIST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_external_model_figures(args.regression_root, args.mnist_root, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

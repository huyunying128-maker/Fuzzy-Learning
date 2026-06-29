"""
Create a cross-dataset summary from the saved regression and classification results.

The summary follows the article-level comparison across Concrete, Superconductivity,
and MNIST. It keeps the strongest interpretable local row and the strongest external
feature-layer row for each dataset in one compact table and writes a matched figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "summary"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "cross_dataset"

REGRESSION_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "summary" / "regression_all_methods_long_summary.csv",
    PROJECT_ROOT / "outputs" / "regression" / "concrete" / "concrete_regression_all_results.csv",
    PROJECT_ROOT / "outputs" / "regression" / "superconductivity" / "superconductivity_regression_all_results.csv",
]
MNIST_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "classification" / "mnist" / "09_mnist_all_results_long.csv",
    PROJECT_ROOT / "outputs" / "classification" / "mnist" / "09_mnist_best_rows.csv",
]


def _read_existing_csv(path: Path) -> Optional[pd.DataFrame]:
    if Path(path).exists() and Path(path).is_file():
        table = pd.read_csv(path)
        if not table.empty:
            table["source_file"] = str(path)
            return table
    return None


def _read_many(paths: Iterable[Path]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in paths:
        table = _read_existing_csv(Path(path))
        if table is not None:
            frames.append(table)
    return frames


def _first_column(table: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in table.columns:
            return name
    return None


def _as_float(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def normalize_regression_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize regression results to a common cross-dataset layout."""
    table = table.copy()
    dataset_col = _first_column(table, ["dataset", "dataset_name"])
    family_col = _first_column(table, ["method_family", "family", "section"])
    method_col = _first_column(table, ["method_name", "method", "name", "model"])
    mse_col = _first_column(table, ["test_mse", "mse", "MSE"])
    mae_col = _first_column(table, ["test_mae", "mae", "MAE"])
    r2_col = _first_column(table, ["test_r2", "r2", "R2", "R^2"])
    degree_col = _first_column(table, ["degree", "Degree"])
    trunc_col = _first_column(table, ["truncation", "truncation_rule", "condition"])
    time_col = _first_column(table, ["runtime_sec", "time_sec", "sec", "Time"])

    out = pd.DataFrame(index=table.index)
    out["dataset"] = table[dataset_col].astype(str).str.lower() if dataset_col else "regression"
    out["task"] = "regression"
    out["family"] = table[family_col].astype(str) if family_col else ""
    out["method"] = table[method_col].astype(str) if method_col else ""
    out["degree"] = _as_float(table[degree_col]) if degree_col else np.nan
    out["truncation"] = table[trunc_col].astype(str) if trunc_col else "none"
    for col in ["k", "f", "p"]:
        out[col] = _as_float(table[col]) if col in table.columns else np.nan
    out["mse"] = _as_float(table[mse_col]) if mse_col else np.nan
    out["mae"] = _as_float(table[mae_col]) if mae_col else np.nan
    out["r2"] = _as_float(table[r2_col]) if r2_col else np.nan
    out["accuracy"] = np.nan
    out["cross_entropy"] = np.nan
    out["runtime_sec"] = _as_float(table[time_col]) if time_col else np.nan

    text = (out["family"] + " " + out["method"]).str.lower()
    out["row_type"] = np.select(
        [
            text.str.contains("external|xgboost|random forest|svr|ann|cnn|dl|deep"),
            text.str.contains("local"),
        ],
        ["external", "interpretable_local"],
        default="other",
    )
    return out.dropna(subset=["mse"], how="all")


def normalize_mnist_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize MNIST classification results to the cross-dataset layout."""
    table = table.copy()
    family_col = _first_column(table, ["family", "section", "method_family"])
    method_col = _first_column(table, ["method", "name", "classifier", "model"])
    view_col = _first_column(table, ["input_view", "view", "feature_view"])
    acc_col = _first_column(table, ["accuracy", "test_accuracy", "acc", "Accuracy"])
    ce_col = _first_column(table, ["cross_entropy", "test_cross_entropy", "ce", "CE"])
    degree_col = _first_column(table, ["degree", "Degree"])
    trunc_col = _first_column(table, ["truncation", "truncation_rule", "condition"])
    time_col = _first_column(table, ["runtime_sec", "time_sec", "sec", "Time"])

    out = pd.DataFrame(index=table.index)
    out["dataset"] = "mnist"
    out["task"] = "classification"
    out["family"] = table[family_col].astype(str) if family_col else ""
    out["method"] = table[method_col].astype(str) if method_col else ""
    out["input_view"] = table[view_col].astype(str) if view_col else ""
    out["degree"] = _as_float(table[degree_col]) if degree_col else np.nan
    out["truncation"] = table[trunc_col].astype(str) if trunc_col else "none"
    for col in ["k", "f", "p"]:
        out[col] = _as_float(table[col]) if col in table.columns else np.nan
    out["mse"] = np.nan
    out["mae"] = np.nan
    out["r2"] = np.nan
    out["accuracy"] = _as_float(table[acc_col]) if acc_col else np.nan
    out["cross_entropy"] = _as_float(table[ce_col]) if ce_col else np.nan
    out["runtime_sec"] = _as_float(table[time_col]) if time_col else np.nan

    text = (out["family"] + " " + out["method"] + " " + out["input_view"]).str.lower()
    out["row_type"] = np.select(
        [
            text.str.contains("external|classifier|xgboost|random forest|svm|ann|cnn|deep"),
            text.str.contains("local"),
        ],
        ["external", "interpretable_local"],
        default="other",
    )
    return out.dropna(subset=["accuracy"], how="all")


def load_cross_dataset_candidates(
    regression_paths: Optional[Sequence[Path]] = None,
    mnist_paths: Optional[Sequence[Path]] = None,
) -> pd.DataFrame:
    """Load all available result files for the cross-dataset comparison."""
    regression_paths = list(regression_paths) if regression_paths is not None else REGRESSION_CANDIDATES
    mnist_paths = list(mnist_paths) if mnist_paths is not None else MNIST_CANDIDATES

    frames: List[pd.DataFrame] = []
    for table in _read_many(regression_paths):
        frames.append(normalize_regression_table(table))
    for table in _read_many(mnist_paths):
        frames.append(normalize_mnist_table(table))

    if not frames:
        searched = [str(path) for path in list(regression_paths) + list(mnist_paths)]
        raise FileNotFoundError("No result files were available for the cross-dataset summary: " + "; ".join(searched))
    return pd.concat(frames, ignore_index=True, sort=False)


def _best_regression_row(group: pd.DataFrame) -> pd.Series:
    return group.dropna(subset=["mse"]).sort_values(["mse", "mae"], ascending=[True, True]).iloc[0]


def _best_classification_row(group: pd.DataFrame) -> pd.Series:
    return group.dropna(subset=["accuracy"]).sort_values(["accuracy", "cross_entropy"], ascending=[False, True]).iloc[0]


def select_best_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select best interpretable-local and external rows for each dataset."""
    rows: List[pd.Series] = []
    for (dataset, row_type), group in candidates.groupby(["dataset", "row_type"], dropna=False):
        if row_type not in {"interpretable_local", "external"}:
            continue
        task = str(group["task"].dropna().iloc[0]) if group["task"].notna().any() else ""
        valid = group.dropna(subset=["mse"] if task == "regression" else ["accuracy"])
        if valid.empty:
            continue
        rows.append(_best_regression_row(valid) if task == "regression" else _best_classification_row(valid))
    if not rows:
        return pd.DataFrame(columns=candidates.columns)
    best = pd.DataFrame(rows).reset_index(drop=True)
    best["summary_label"] = best["dataset"].str.title() + " - " + best["row_type"].str.replace("_", " ").str.title()
    return best


def make_article_style_summary(best_rows: pd.DataFrame) -> pd.DataFrame:
    """Create the compact article-level summary table."""
    records: List[Dict[str, object]] = []
    for dataset, group in best_rows.groupby("dataset"):
        task = str(group["task"].iloc[0])
        local = group[group["row_type"].eq("interpretable_local")]
        external = group[group["row_type"].eq("external")]
        record: Dict[str, object] = {"dataset": dataset, "main_task": task}
        if not local.empty:
            row = local.iloc[0]
            record.update({
                "strongest_interpretable_local_row": row.get("method", ""),
                "local_degree": row.get("degree", np.nan),
                "local_truncation": row.get("truncation", np.nan),
                "local_k": row.get("k", np.nan),
                "local_f": row.get("f", np.nan),
                "local_p": row.get("p", np.nan),
                "local_mse": row.get("mse", np.nan),
                "local_accuracy": row.get("accuracy", np.nan),
            })
        if not external.empty:
            row = external.iloc[0]
            record.update({
                "strongest_external_row": row.get("method", ""),
                "external_mse": row.get("mse", np.nan),
                "external_accuracy": row.get("accuracy", np.nan),
                "external_input_view": row.get("input_view", ""),
            })
        records.append(record)
    return pd.DataFrame(records).sort_values("dataset")


def plot_cross_dataset(best_rows: pd.DataFrame, output_path: Path) -> Path:
    """Create a compact visual comparison of best local and external rows."""
    if best_rows.empty:
        raise ValueError("The best-row table is empty.")

    figure_rows = best_rows.copy()
    figure_rows["metric_value"] = np.where(
        figure_rows["task"].eq("regression"),
        figure_rows["mse"],
        figure_rows["accuracy"],
    )
    figure_rows["display"] = figure_rows["dataset"].str.title() + "\n" + figure_rows["row_type"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(figure_rows["display"], figure_rows["metric_value"])
    ax.set_title("Cross-dataset strongest reported rows")
    ax.set_ylabel("MSE for regression; accuracy for MNIST")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def create_cross_dataset_summary(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    regression_paths: Optional[Sequence[Path]] = None,
    mnist_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, Path]:
    """Write candidate, best-row, article-style summary, and figure outputs."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_cross_dataset_candidates(regression_paths=regression_paths, mnist_paths=mnist_paths)
    best = select_best_rows(candidates)
    article = make_article_style_summary(best)

    candidate_path = output_dir / "06_cross_dataset_candidate_rows.csv"
    best_path = output_dir / "06_cross_dataset_best_rows.csv"
    article_path = output_dir / "06_cross_dataset_article_summary.csv"
    figure_path = figure_dir / "06_cross_dataset_best_rows.png"

    candidates.to_csv(candidate_path, index=False)
    best.to_csv(best_path, index=False)
    article.to_csv(article_path, index=False)
    plot_cross_dataset(best, figure_path)

    return {
        "candidate_rows": candidate_path,
        "best_rows": best_path,
        "article_summary": article_path,
        "figure": figure_path,
    }


def parse_path_list(text: Optional[str]) -> Optional[List[Path]]:
    """Parse a comma-separated list of optional CSV paths."""
    if text is None or not str(text).strip():
        return None
    return [Path(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cross-dataset summary tables and figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--regression-files", default=None, help="Comma-separated regression result CSV files.")
    parser.add_argument("--mnist-files", default=None, help="Comma-separated MNIST result CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_cross_dataset_summary(
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        regression_paths=parse_path_list(args.regression_files),
        mnist_paths=parse_path_list(args.mnist_files),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

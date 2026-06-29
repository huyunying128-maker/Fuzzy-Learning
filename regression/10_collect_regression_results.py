"""
Regression result collection utilities.

This module collects the concrete and superconductivity result files into a
single long-form table. It also creates compact best-row summaries that are
convenient for checking the strongest interpretable local row and the strongest
external feature-layer row for each regression dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGRESSION_OUTPUT = PROJECT_ROOT / "outputs" / "regression"
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "outputs" / "summary"


STANDARD_COLUMNS = [
    "dataset",
    "task",
    "method_family",
    "method_name",
    "degree",
    "k",
    "f",
    "p",
    "truncation",
    "test_mse",
    "test_rmse",
    "test_mae",
    "test_r2",
    "n_iter",
    "runtime_sec",
]


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists() and path.is_file():
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        return frame
    return None


def find_regression_result_files(root: Path) -> List[Path]:
    """Find result CSV files produced by the regression experiment runners."""
    root = Path(root)
    patterns = [
        "*/concrete_regression_all_results.csv",
        "*/superconductivity_regression_all_results.csv",
        "concrete/concrete_regression_all_results.csv",
        "superconductivity/superconductivity_regression_all_results.csv",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    if not files:
        files.extend(root.rglob("*_regression_all_results.csv"))
    return sorted(set(files))


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep common regression columns while preserving additional metadata."""
    frame = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    front = [column for column in STANDARD_COLUMNS if column in frame.columns]
    rest = [column for column in frame.columns if column not in front]
    return frame[front + rest]


def collect_regression_results(
    input_root: Path = DEFAULT_REGRESSION_OUTPUT,
    output_dir: Path = DEFAULT_SUMMARY_DIR,
    output_name: str = "regression_all_methods_long_summary.csv",
) -> pd.DataFrame:
    """Collect all regression experiment results and save a long-form table."""
    files = find_regression_result_files(input_root)
    if not files:
        raise FileNotFoundError(f"No regression result files were found under {input_root}.")

    frames = []
    for path in files:
        frame = _read_csv_if_exists(path)
        if frame is not None and not frame.empty:
            frames.append(normalize_columns(frame))
    if not frames:
        raise ValueError("Regression result files were found, but they contained no rows.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / output_name, index=False)
    return combined


def _is_interpretable_local(frame: pd.DataFrame) -> pd.Series:
    family = frame.get("method_family", pd.Series("", index=frame.index)).astype(str).str.lower()
    name = frame.get("method_name", pd.Series("", index=frame.index)).astype(str).str.lower()
    return family.str.contains("local") | name.str.contains("local")


def _is_external_feature_row(frame: pd.DataFrame) -> pd.Series:
    family = frame.get("method_family", pd.Series("", index=frame.index)).astype(str).str.lower()
    name = frame.get("method_name", pd.Series("", index=frame.index)).astype(str).str.lower()
    return family.str.contains("external") | name.str.contains("xgboost|random forest|svr|ann|cnn|dl")


def best_rows_by_dataset(frame: pd.DataFrame, mask: pd.Series, metric: str = "test_mse") -> pd.DataFrame:
    """Select the lowest-MSE row for each dataset from a filtered result table."""
    if metric not in frame.columns:
        raise ValueError(f"Metric column {metric!r} is not available.")
    subset = frame.loc[mask].copy()
    subset = subset.dropna(subset=[metric])
    if subset.empty:
        return pd.DataFrame(columns=frame.columns)

    rows = []
    for dataset, group in subset.groupby("dataset", dropna=False):
        index = group[metric].astype(float).idxmin()
        rows.append(group.loc[index])
    return pd.DataFrame(rows).reset_index(drop=True)


def create_regression_summary_tables(
    combined: pd.DataFrame,
    output_dir: Path = DEFAULT_SUMMARY_DIR,
) -> Dict[str, pd.DataFrame]:
    """Create compact best-row summaries from the long-form regression table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_best = best_rows_by_dataset(combined, _is_interpretable_local(combined), metric="test_mse")
    external_best = best_rows_by_dataset(combined, _is_external_feature_row(combined), metric="test_mse")

    local_best.to_csv(output_dir / "regression_best_interpretable_local_rows.csv", index=False)
    external_best.to_csv(output_dir / "regression_best_external_rows.csv", index=False)

    cross_dataset = pd.DataFrame(
        {
            "dataset": sorted(combined["dataset"].dropna().unique()),
        }
    )
    if not local_best.empty:
        cross_dataset = cross_dataset.merge(
            local_best[["dataset", "method_name", "degree", "k", "f", "p", "truncation", "test_mse", "test_mae", "test_r2"]]
            .rename(columns={
                "method_name": "best_local_method",
                "test_mse": "best_local_mse",
                "test_mae": "best_local_mae",
                "test_r2": "best_local_r2",
            }),
            on="dataset",
            how="left",
        )
    if not external_best.empty:
        cross_dataset = cross_dataset.merge(
            external_best[["dataset", "method_name", "test_mse", "test_mae", "test_r2"]]
            .rename(columns={
                "method_name": "best_external_method",
                "test_mse": "best_external_mse",
                "test_mae": "best_external_mae",
                "test_r2": "best_external_r2",
            }),
            on="dataset",
            how="left",
        )
    cross_dataset.to_csv(output_dir / "regression_cross_dataset_summary.csv", index=False)

    return {
        "best_interpretable_local": local_best,
        "best_external": external_best,
        "cross_dataset": cross_dataset,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect regression experiment result tables.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_REGRESSION_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-name", default="regression_all_methods_long_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = collect_regression_results(
        input_root=args.input_root,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
    summaries = create_regression_summary_tables(combined, output_dir=args.output_dir)
    print(f"Saved regression long summary with {combined.shape[0]} rows")
    for name, table in summaries.items():
        print(f"Saved {name}: {table.shape[0]} rows")


if __name__ == "__main__":
    main()

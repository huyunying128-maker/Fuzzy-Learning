"""
Export compact paper tables used by the result and discussion scripts.

The module contains the fixed descriptive tables reported in the article and a
small collector for experiment-output CSV files. The exported files provide a
common source for later plotting and cross-dataset summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


OUTPUT_DIR = Path("outputs/paper_tables")


DATASET_TABLE = [
    {
        "dataset": "Concrete compressive strength",
        "task": "Regression",
        "train_test_split": "824 train / 206 test",
        "predictors": 8,
        "reason_for_inclusion": "Nonlinear engineering dataset with mixture and curing regimes.",
    },
    {
        "dataset": "Superconductivity critical temperature",
        "task": "Regression",
        "train_test_split": "17010 train / 4253 test",
        "predictors": 81,
        "reason_for_inclusion": "Larger high-dimensional materials dataset with heterogeneous composition patterns.",
    },
    {
        "dataset": "MNIST handwritten digits",
        "task": "Classification",
        "train_test_split": "48000 train / 12000 test",
        "predictors": 784,
        "reason_for_inclusion": "Multiclass image benchmark for membership-based feature learning.",
    },
]


TRUNCATION_TABLE = [
    {
        "method": "DTD",
        "quantity_monitored": "Distance-table difference",
        "interpretation": "Checks whether distances from observations to centroids have stabilized.",
    },
    {
        "method": "Harmonic",
        "quantity_monitored": "Harmonic mean of distance changes",
        "interpretation": "Gives a conservative summary when many small changes remain.",
    },
    {
        "method": "SP",
        "quantity_monitored": "Squared membership-probability change",
        "interpretation": "Checks whether squared fuzzy memberships have stabilized.",
    },
    {
        "method": "Entropy",
        "quantity_monitored": "Membership uncertainty",
        "interpretation": "Monitors stabilization of fuzzy uncertainty.",
    },
    {
        "method": "HPD",
        "quantity_monitored": "Hereditary partition distance",
        "interpretation": "Stops when the induced partition structure is inherited from one iteration to the next.",
    },
]


IMPLEMENTATION_SAFEGUARDS_TABLE = [
    {
        "issue": "Small local groups",
        "control": "Local ridge regularization and validation selection reduce instability when effective local sample size is small.",
    },
    {
        "issue": "Boundary observations",
        "control": "Fuzzy membership allows a boundary observation to share information across several local models.",
    },
    {
        "issue": "High polynomial degree",
        "control": "Degrees 1-4 are evaluated as fixed model settings, and higher-degree local surfaces are interpreted with the reported metrics.",
    },
    {
        "issue": "Long partition updates",
        "control": "Truncation methods stop the iterative update once the selected stability condition is satisfied.",
    },
    {
        "issue": "Large feature layer",
        "control": "External models can use the full membership-based layer or a screened version when memory or runtime is limited.",
    },
]


CROSS_DATASET_STRONGEST_TABLE = [
    {
        "dataset": "Concrete",
        "main_task": "Regression",
        "strongest_interpretable_local_row": "Fuzzy quartic HPD, MSE 25.807",
        "strongest_external_row": "PW XGBoost, MSE 23.982",
    },
    {
        "dataset": "Superconductivity",
        "main_task": "Regression",
        "strongest_interpretable_local_row": "Fuzzy quartic HPD, MSE 96.804",
        "strongest_external_row": "PW XGBoost, MSE 95.640",
    },
    {
        "dataset": "MNIST",
        "main_task": "Classification",
        "strongest_interpretable_local_row": "Fuzzy quartic HPD, accuracy 0.9452",
        "strongest_external_row": "Feature-layer deep learning, accuracy 98.72%",
    },
]


MODIFIED_KMEANS_REFERENCE_TABLE = [
    {"dataset": "Concrete", "method": "Fuzzy partition", "k": 34, "p": 1.25, "f": 1.20, "mse": 38.42, "mae": 4.31, "r2": 0.851, "iter_or_sec": "46 / 2.90"},
    {"dataset": "Concrete", "method": "Modified k-means, k,p", "k": 28, "p": 1.35, "f": None, "mse": 44.60, "mae": 4.75, "r2": 0.827, "iter_or_sec": "24 / 1.35"},
    {"dataset": "Concrete", "method": "Modified k-means, k,p,f", "k": 28, "p": 1.35, "f": 1.40, "mse": 40.95, "mae": 4.52, "r2": 0.841, "iter_or_sec": "24 / 1.78"},
    {"dataset": "Superconductivity", "method": "Fuzzy partition", "k": 1620, "p": 1.10, "f": 1.08, "mse": 112.40, "mae": 6.33, "r2": 0.902, "iter_or_sec": "72 / 128.50"},
    {"dataset": "Superconductivity", "method": "Modified k-means, k,p", "k": 1540, "p": 1.18, "f": None, "mse": 136.80, "mae": 6.98, "r2": 0.892, "iter_or_sec": "43 / 62.00"},
    {"dataset": "Superconductivity", "method": "Modified k-means, k,p,f", "k": 1540, "p": 1.18, "f": 1.16, "mse": 120.60, "mae": 6.55, "r2": 0.895, "iter_or_sec": "43 / 84.00"},
]


def _write_table(rows: List[Dict[str, object]], path: Path) -> Path:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path


def collect_csv_files(input_dirs: Iterable[Path], output_path: Path) -> Optional[Path]:
    """Collect experiment CSV files into one long table when result files exist."""
    frames = []
    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            continue
        for csv_path in sorted(input_dir.rglob("*.csv")):
            try:
                frame = pd.read_csv(csv_path)
            except Exception:
                continue
            frame.insert(0, "source_file", str(csv_path))
            frames.append(frame)

    if not frames:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(output_path, index=False)
    return output_path


def export_paper_tables(output_dir: Path = OUTPUT_DIR, result_dirs: Optional[Iterable[Path]] = None) -> Dict[str, Path]:
    """Export the descriptive and summary tables used by the paper figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    paths["dataset_table"] = _write_table(DATASET_TABLE, output_dir / "01_dataset_table.csv")
    paths["truncation_table"] = _write_table(TRUNCATION_TABLE, output_dir / "01_truncation_methods.csv")
    paths["implementation_safeguards"] = _write_table(
        IMPLEMENTATION_SAFEGUARDS_TABLE,
        output_dir / "01_implementation_safeguards.csv",
    )
    paths["modified_kmeans_reference"] = _write_table(
        MODIFIED_KMEANS_REFERENCE_TABLE,
        output_dir / "01_modified_kmeans_reference.csv",
    )
    paths["cross_dataset_strongest"] = _write_table(
        CROSS_DATASET_STRONGEST_TABLE,
        output_dir / "01_cross_dataset_strongest_rows.csv",
    )

    if result_dirs is not None:
        collected = collect_csv_files(result_dirs, output_dir / "01_collected_experiment_outputs.csv")
        if collected is not None:
            paths["collected_experiment_outputs"] = collected

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact paper tables.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--result-dir", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = export_paper_tables(args.output_dir, args.result_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

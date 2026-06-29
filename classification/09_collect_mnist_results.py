"""
Collect MNIST classification outputs into paper-ready result tables.

The collector reads the CSV files produced by the MNIST scripts and writes a
long table, a local-classification table, an external-classifier table, and a
small best-row summary. Missing files are skipped so partial runs can still be
summarized.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification" / "mnist"


LOCAL_FILES = {
    "clustering_only": "01_mnist_clustering_only.csv",
    "raw_logit": "02_raw_logit_baselines.csv",
    "crisp_local_logit": "03_crisp_local_logit.csv",
    "fuzzy_local_logit": "04_fuzzy_local_logit.csv",
}
EXTERNAL_FILE = "06_external_classifiers.csv"
FEATURE_FILE = "05_mnist_feature_layers_summary.csv"
CENTROID_FILE = "07_mnist_centroid_membership.csv"


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not Path(path).exists():
        return None
    table = pd.read_csv(path)
    if table.empty:
        return None
    return table


def _first_existing_column(table: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in table.columns:
            return name
    return None


def normalize_local_table(table: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize local MNIST result columns from one source table."""
    table = table.copy()
    acc_col = _first_existing_column(table, ["test_accuracy", "accuracy", "Accuracy", "acc"])
    ce_col = _first_existing_column(table, ["test_cross_entropy", "cross_entropy", "CE", "ce"])
    iter_col = _first_existing_column(table, ["n_iter", "iteration", "iterations", "Iter."])
    time_col = _first_existing_column(table, ["runtime_sec", "time_sec", "Time", "sec"])

    normalized = pd.DataFrame()
    normalized["source_file"] = source
    normalized["family"] = table.get("family", source)
    normalized["method"] = table.get("method", table.get("family", source))
    normalized["degree"] = table.get("degree", np.nan)
    normalized["truncation"] = table.get("truncation", table.get("truncation_rule", "none"))
    normalized["k"] = table.get("k", np.nan)
    normalized["f"] = table.get("f", np.nan)
    normalized["p"] = table.get("p", np.nan)
    normalized["accuracy"] = table[acc_col] if acc_col is not None else np.nan
    normalized["cross_entropy"] = table[ce_col] if ce_col is not None else np.nan
    normalized["n_iter"] = table[iter_col] if iter_col is not None else np.nan
    normalized["runtime_sec"] = table[time_col] if time_col is not None else np.nan
    return normalized


def normalize_external_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize external-classifier rows into a matched input-view table."""
    table = table.copy()
    acc_col = _first_existing_column(table, ["test_accuracy", "accuracy", "acc"])
    ce_col = _first_existing_column(table, ["test_cross_entropy", "cross_entropy", "ce"])
    time_col = _first_existing_column(table, ["runtime_sec", "time_sec", "sec"])

    normalized = pd.DataFrame()
    normalized["classifier"] = table.get("classifier", table.get("model", table.get("model_name", "external")))
    normalized["input_view"] = table.get("input_view", table.get("view", "unknown"))
    normalized["accuracy"] = table[acc_col] if acc_col is not None else np.nan
    normalized["cross_entropy"] = table[ce_col] if ce_col is not None else np.nan
    normalized["runtime_sec"] = table[time_col] if time_col is not None else np.nan
    return normalized


def best_rows(local_long: pd.DataFrame, external_long: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Select representative best rows for local and external comparisons."""
    records: List[Dict[str, object]] = []
    if not local_long.empty and "accuracy" in local_long.columns:
        valid = local_long.dropna(subset=["accuracy"])
        if not valid.empty:
            local_best = valid.sort_values(["accuracy", "cross_entropy"], ascending=[False, True]).iloc[0]
            records.append(
                {
                    "section": "local_classification",
                    "name": str(local_best.get("method", local_best.get("family", "local"))),
                    "family": str(local_best.get("family", "")),
                    "input_view": "local_logit",
                    "degree": local_best.get("degree", np.nan),
                    "truncation": local_best.get("truncation", np.nan),
                    "k": local_best.get("k", np.nan),
                    "f": local_best.get("f", np.nan),
                    "p": local_best.get("p", np.nan),
                    "accuracy": local_best.get("accuracy", np.nan),
                    "cross_entropy": local_best.get("cross_entropy", np.nan),
                    "runtime_sec": local_best.get("runtime_sec", np.nan),
                }
            )

    if external_long is not None and not external_long.empty:
        valid = external_long.dropna(subset=["accuracy"])
        if not valid.empty:
            external_best = valid.sort_values(["accuracy", "cross_entropy"], ascending=[False, True]).iloc[0]
            records.append(
                {
                    "section": "external_classifier",
                    "name": str(external_best.get("classifier", "external")),
                    "family": "external_classifier",
                    "input_view": str(external_best.get("input_view", "unknown")),
                    "degree": np.nan,
                    "truncation": np.nan,
                    "k": np.nan,
                    "f": np.nan,
                    "p": np.nan,
                    "accuracy": external_best.get("accuracy", np.nan),
                    "cross_entropy": external_best.get("cross_entropy", np.nan),
                    "runtime_sec": external_best.get("runtime_sec", np.nan),
                }
            )
    return pd.DataFrame(records)


def collect_mnist_results(
    input_dir: Path = OUTPUT_DIR,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Collect MNIST output files and write combined summary tables."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    local_tables: List[pd.DataFrame] = []
    for source, file_name in LOCAL_FILES.items():
        table = _read_csv_if_exists(input_dir / file_name)
        if table is not None:
            local_tables.append(normalize_local_table(table, source=source))

    local_long = pd.concat(local_tables, ignore_index=True) if local_tables else pd.DataFrame()
    local_path = output_dir / "09_mnist_local_results_long.csv"
    local_long.to_csv(local_path, index=False)

    external_raw = _read_csv_if_exists(input_dir / EXTERNAL_FILE)
    external_long = normalize_external_table(external_raw) if external_raw is not None else pd.DataFrame()
    external_path = output_dir / "09_mnist_external_results_long.csv"
    external_long.to_csv(external_path, index=False)

    all_blocks = []
    if not local_long.empty:
        local_block = local_long.copy()
        local_block["section"] = "local"
        local_block["name"] = local_block["method"]
        local_block["input_view"] = "local_logit"
        all_blocks.append(local_block)
    if not external_long.empty:
        external_block = external_long.copy()
        external_block["section"] = "external"
        external_block["family"] = "external_classifier"
        external_block["method"] = external_block["classifier"]
        external_block["degree"] = np.nan
        external_block["truncation"] = np.nan
        external_block["k"] = np.nan
        external_block["f"] = np.nan
        external_block["p"] = np.nan
        external_block["n_iter"] = np.nan
        external_block["name"] = external_block["classifier"]
        all_blocks.append(external_block)
    all_long = pd.concat(all_blocks, ignore_index=True, sort=False) if all_blocks else pd.DataFrame()
    all_path = output_dir / "09_mnist_all_results_long.csv"
    all_long.to_csv(all_path, index=False)

    best = best_rows(local_long, external_long)
    best_path = output_dir / "09_mnist_best_rows.csv"
    best.to_csv(best_path, index=False)

    paths = {
        "local_long": local_path,
        "external_long": external_path,
        "all_long": all_path,
        "best_rows": best_path,
    }

    feature_table = _read_csv_if_exists(input_dir / FEATURE_FILE)
    if feature_table is not None:
        feature_path = output_dir / "09_mnist_feature_layer_summary.csv"
        feature_table.to_csv(feature_path, index=False)
        paths["feature_layer_summary"] = feature_path

    centroid_table = _read_csv_if_exists(input_dir / CENTROID_FILE)
    if centroid_table is not None:
        centroid_path = output_dir / "09_mnist_centroid_membership_summary.csv"
        centroid_table.to_csv(centroid_path, index=False)
        paths["centroid_membership"] = centroid_path

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MNIST classification result tables.")
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_mnist_results(input_dir=args.input_dir, output_dir=args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

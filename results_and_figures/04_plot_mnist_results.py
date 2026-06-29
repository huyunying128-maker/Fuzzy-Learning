"""
Create MNIST local-classification figures from saved result tables.

The figures summarize the classification section through the raw-logit path,
the crisp local HPD path, the fuzzy local HPD path, the quartic local family,
and a 6:4 mixed score using accuracy and inverse cross entropy.
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
DEFAULT_INPUTS = [
    PROJECT_ROOT / "outputs" / "classification" / "mnist" / "mnist_all_methods_long_summary.csv",
    PROJECT_ROOT / "outputs" / "summary" / "mnist_all_methods_long_summary.csv",
]
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "outputs" / "classification" / "mnist"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "mnist"

LOCAL_FILES = [
    "01_mnist_clustering_only.csv",
    "02_raw_logit_baselines.csv",
    "03_crisp_local_logit.csv",
    "04_fuzzy_local_logit.csv",
]


def _first_column(table: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in table.columns:
            return name
    return None


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _existing_summary_path(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def _read_local_files(local_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file_name in LOCAL_FILES:
        path = Path(local_dir) / file_name
        if path.exists():
            table = pd.read_csv(path)
            table["source_file"] = file_name
            frames.append(table)
    if not frames:
        raise FileNotFoundError(f"No MNIST local result files were found under {local_dir}.")
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize_mnist_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names used by the MNIST scripts."""
    table = table.copy()
    family_col = _first_column(table, ["family", "method_family"])
    method_col = _first_column(table, ["method", "method_name", "classifier"])
    trunc_col = _first_column(table, ["truncation", "truncation_rule", "condition"])
    degree_col = _first_column(table, ["degree", "Degree"])
    acc_col = _first_column(table, ["accuracy", "test_accuracy", "acc", "Accuracy"])
    ce_col = _first_column(table, ["cross_entropy", "test_cross_entropy", "ce", "CE"])
    k_col = _first_column(table, ["k"])
    f_col = _first_column(table, ["f"])
    p_col = _first_column(table, ["p"])
    iter_col = _first_column(table, ["n_iter", "iteration", "iterations", "Iter."])
    time_col = _first_column(table, ["runtime_sec", "time_sec", "Time", "sec"])

    out = pd.DataFrame(index=table.index)
    out["family"] = table[family_col].astype(str) if family_col else ""
    out["method"] = table[method_col].astype(str) if method_col else out["family"]
    out["truncation"] = table[trunc_col].astype(str).str.lower() if trunc_col else "none"
    out["degree"] = _as_float(table[degree_col]) if degree_col else np.nan
    out["accuracy"] = _as_float(table[acc_col]) if acc_col else np.nan
    out["cross_entropy"] = _as_float(table[ce_col]) if ce_col else np.nan
    out["k"] = _as_float(table[k_col]) if k_col else np.nan
    out["f"] = _as_float(table[f_col]) if f_col else np.nan
    out["p"] = _as_float(table[p_col]) if p_col else np.nan
    out["n_iter"] = _as_float(table[iter_col]) if iter_col else np.nan
    out["runtime_sec"] = _as_float(table[time_col]) if time_col else np.nan

    text = (out["family"] + " " + out["method"]).str.lower()
    out["is_raw_logit"] = text.str.contains("raw") | text.str.contains("global logit")
    out["is_crisp_local"] = text.str.contains("crisp") & text.str.contains("local")
    out["is_fuzzy_local"] = text.str.contains("fuzzy") & text.str.contains("local")
    out["is_clustering_only"] = text.str.contains("clustering") & text.str.contains("only")
    return out.dropna(subset=["accuracy"])


def load_mnist_results(input_path: Optional[Path] = None, local_dir: Path = DEFAULT_LOCAL_DIR) -> pd.DataFrame:
    """Load MNIST local-classification results for plotting."""
    if input_path is not None:
        table = pd.read_csv(input_path)
    else:
        summary_path = _existing_summary_path(DEFAULT_INPUTS)
        table = pd.read_csv(summary_path) if summary_path is not None else _read_local_files(local_dir)
    return normalize_mnist_table(table)


def _select_best_accuracy(table: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    valid = table.dropna(subset=["accuracy"]).copy()
    if valid.empty:
        return valid
    idx = valid.groupby(list(group_cols), dropna=False)["accuracy"].idxmax()
    return valid.loc[idx].sort_values(list(group_cols)).reset_index(drop=True)


def build_degree_curve_table(table: pd.DataFrame) -> pd.DataFrame:
    """Create the degree-wise MNIST accuracy trend table."""
    rows: List[Dict[str, object]] = []
    selections = {
        "Raw logit": table[table["is_raw_logit"]],
        "Crisp local HPD": table[table["is_crisp_local"] & table["truncation"].str.contains("hpd", na=False)],
        "Fuzzy local HPD": table[table["is_fuzzy_local"] & table["truncation"].str.contains("hpd", na=False)],
    }
    for label, subset in selections.items():
        best = _select_best_accuracy(subset, ["degree"])
        for _, row in best.iterrows():
            rows.append({
                "series": label,
                "degree": row["degree"],
                "accuracy": row["accuracy"],
                "cross_entropy": row.get("cross_entropy", np.nan),
            })
    return pd.DataFrame(rows).sort_values(["series", "degree"])


def build_quartic_comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    """Create a compact quartic local table for the MNIST bar plots."""
    quartic = table[table["degree"].eq(4)].copy()
    pieces = []
    selectors = {
        "Raw quartic logit": quartic[quartic["is_raw_logit"]],
        "Crisp quartic HPD": quartic[quartic["is_crisp_local"] & quartic["truncation"].str.contains("hpd", na=False)],
        "Fuzzy quartic DTD": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("dtd", na=False)],
        "Fuzzy quartic harmonic": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("harmonic", na=False)],
        "Fuzzy quartic SP": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("sp", na=False)],
        "Fuzzy quartic entropy": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("entropy", na=False)],
        "Fuzzy quartic HPD": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("hpd", na=False)],
    }
    for label, subset in selectors.items():
        best = _select_best_accuracy(subset, ["degree"])
        if not best.empty:
            row = best.iloc[0].to_dict()
            row["display_method"] = label
            pieces.append(row)
    return pd.DataFrame(pieces)


def add_mixed_score(table: pd.DataFrame) -> pd.DataFrame:
    """Add the 6:4 MNIST mixed score using accuracy and inverse cross entropy."""
    table = table.copy()
    accuracy = table["accuracy"].astype(float)
    ce = table["cross_entropy"].astype(float)
    inv_ce = 1.0 / ce.replace(0, np.nan)

    def normalize(values: pd.Series) -> pd.Series:
        if values.notna().sum() <= 1 or np.isclose(values.max(), values.min()):
            return pd.Series(np.ones(len(values)), index=values.index)
        return (values - values.min()) / (values.max() - values.min())

    table["normalized_accuracy"] = normalize(accuracy)
    table["normalized_inverse_ce"] = normalize(inv_ce)
    table["mixed_score_6_4"] = 0.6 * table["normalized_accuracy"] + 0.4 * table["normalized_inverse_ce"]
    return table


def _plot_degree_curves(curves: pd.DataFrame, output_path: Path) -> Path:
    if curves.empty:
        raise ValueError("The MNIST degree-curve table is empty.")
    fig, ax = plt.subplots(figsize=(8, 5))
    for series, subset in curves.groupby("series"):
        subset = subset.sort_values("degree")
        ax.plot(subset["degree"], subset["accuracy"], marker="o", label=series)
    ax.set_xticks([1, 2, 3, 4], ["1", "2", "3", "4"])
    ax.set_xlabel("Logit degree")
    ax.set_ylabel("Accuracy")
    ax.set_title("MNIST local-classification degree-wise comparison")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_bar(table: pd.DataFrame, value_col: str, ylabel: str, title: str, output_path: Path) -> Path:
    if table.empty:
        raise ValueError(f"The table for {title} is empty.")
    plot_table = table.sort_values(value_col, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(plot_table["display_method"], plot_table[value_col])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def create_mnist_figures(
    input_path: Optional[Path] = None,
    local_dir: Path = DEFAULT_LOCAL_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Path]:
    """Create MNIST CSV summaries and figure files."""
    output_dir = Path(output_dir)
    table = load_mnist_results(input_path, local_dir)
    curves = build_degree_curve_table(table)
    quartic = build_quartic_comparison_table(table)
    quartic_scored = add_mixed_score(quartic) if not quartic.empty else quartic

    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / "04_mnist_degree_curves.csv"
    quartic_path = output_dir / "04_mnist_quartic_comparison.csv"
    curves.to_csv(curves_path, index=False)
    quartic_scored.to_csv(quartic_path, index=False)

    paths = {
        "degree_curve_table": curves_path,
        "quartic_table": quartic_path,
        "degree_curve_figure": _plot_degree_curves(curves, output_dir / "04_mnist_degree_accuracy.png"),
        "quartic_accuracy_figure": _plot_bar(quartic_scored, "accuracy", "Accuracy", "MNIST quartic local comparison", output_dir / "04_mnist_quartic_accuracy.png"),
        "mixed_score_figure": _plot_bar(quartic_scored, "mixed_score_6_4", "Mixed score", "MNIST mixed-score comparison", output_dir / "04_mnist_mixed_score.png"),
    }
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MNIST local-classification figures.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_mnist_figures(args.input, args.local_dir, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

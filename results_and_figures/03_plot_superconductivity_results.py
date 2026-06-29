"""
Create superconductivity regression figures from the saved experiment tables.

The figures mirror the concrete result layout: degree-wise MSE trends, a
quartic-family comparison, and a 6:4 mixed-score summary using normalized
inverse MSE and normalized R2.
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
DEFAULT_INPUTS = [
    PROJECT_ROOT / "outputs" / "regression" / "superconductivity" / "superconductivity_regression_all_results.csv",
    PROJECT_ROOT / "outputs" / "summary" / "regression_all_methods_long_summary.csv",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "superconductivity"
DEGREE_LABELS = {1: "linear", 2: "quadratic", 3: "cubic", 4: "quartic"}


def _first_existing_path(paths: Sequence[Path]) -> Path:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"No superconductivity result table was found among: {joined}")


def _first_column(table: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in table.columns:
            return name
    return None


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_superconductivity_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names used by the regression scripts."""
    table = table.copy()
    dataset_col = _first_column(table, ["dataset", "dataset_name"])
    if dataset_col is not None:
        mask = table[dataset_col].astype(str).str.lower().str.contains("superconduct")
        table = table.loc[mask].copy()

    method_col = _first_column(table, ["method_name", "method", "family"])
    family_col = _first_column(table, ["method_family", "family"])
    trunc_col = _first_column(table, ["truncation", "truncation_rule", "condition"])
    mse_col = _first_column(table, ["test_mse", "mse", "MSE"])
    mae_col = _first_column(table, ["test_mae", "mae", "MAE"])
    r2_col = _first_column(table, ["test_r2", "r2", "R2", "R^2"])
    degree_col = _first_column(table, ["degree", "Degree"])
    k_col = _first_column(table, ["k"])
    f_col = _first_column(table, ["f"])
    p_col = _first_column(table, ["p"])
    iter_col = _first_column(table, ["n_iter", "iteration", "iterations", "Iter."])
    time_col = _first_column(table, ["runtime_sec", "time_sec", "Time", "sec"])

    out = pd.DataFrame(index=table.index)
    out["dataset"] = "superconductivity"
    out["method_family"] = table[family_col].astype(str) if family_col else ""
    out["method_name"] = table[method_col].astype(str) if method_col else ""
    out["truncation"] = table[trunc_col].astype(str).str.lower() if trunc_col else "none"
    out["degree"] = _as_float(table[degree_col]) if degree_col else np.nan
    out["degree_label"] = out["degree"].map(DEGREE_LABELS)
    out["mse"] = _as_float(table[mse_col]) if mse_col else np.nan
    out["mae"] = _as_float(table[mae_col]) if mae_col else np.nan
    out["r2"] = _as_float(table[r2_col]) if r2_col else np.nan
    out["k"] = _as_float(table[k_col]) if k_col else np.nan
    out["f"] = _as_float(table[f_col]) if f_col else np.nan
    out["p"] = _as_float(table[p_col]) if p_col else np.nan
    out["n_iter"] = _as_float(table[iter_col]) if iter_col else np.nan
    out["runtime_sec"] = _as_float(table[time_col]) if time_col else np.nan

    text = (out["method_family"] + " " + out["method_name"]).str.lower()
    out["is_global"] = text.str.contains("global") | text.str.contains("polynomial baseline")
    out["is_crisp_local"] = text.str.contains("crisp") & text.str.contains("local")
    out["is_fuzzy_local"] = text.str.contains("fuzzy") & text.str.contains("local")
    out["is_external"] = text.str.contains("external|xgboost|random forest|svr|ann|cnn|dl")
    return out.dropna(subset=["mse"])


def load_superconductivity_results(input_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the superconductivity result table used by the plotting functions."""
    path = Path(input_path) if input_path is not None else _first_existing_path(DEFAULT_INPUTS)
    return normalize_superconductivity_table(pd.read_csv(path))


def _select_lowest_mse(table: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    valid = table.dropna(subset=["mse"]).copy()
    if valid.empty:
        return valid
    idx = valid.groupby(list(group_cols), dropna=False)["mse"].idxmin()
    return valid.loc[idx].sort_values(list(group_cols)).reset_index(drop=True)


def build_degree_curve_table(table: pd.DataFrame) -> pd.DataFrame:
    """Create the degree-wise data used in the superconductivity trend plot."""
    rows: List[Dict[str, object]] = []
    global_rows = _select_lowest_mse(table[table["is_global"]], ["degree"])
    for _, row in global_rows.iterrows():
        rows.append({"series": "Global polynomial", "degree": row["degree"], "mse": row["mse"]})

    crisp_hpd = table[table["is_crisp_local"] & table["truncation"].str.contains("hpd", na=False)]
    crisp_rows = _select_lowest_mse(crisp_hpd, ["degree"])
    for _, row in crisp_rows.iterrows():
        rows.append({"series": "Crisp local HPD", "degree": row["degree"], "mse": row["mse"]})

    fuzzy = table[table["is_fuzzy_local"]].copy()
    for truncation in ["dtd", "harmonic", "sp", "entropy", "hpd"]:
        subset = fuzzy[fuzzy["truncation"].str.contains(truncation, na=False)]
        subset = _select_lowest_mse(subset, ["degree"])
        label = f"Fuzzy local {truncation.upper() if truncation in {'dtd', 'sp', 'hpd'} else truncation}"
        for _, row in subset.iterrows():
            rows.append({"series": label, "degree": row["degree"], "mse": row["mse"]})
    return pd.DataFrame(rows).sort_values(["series", "degree"])


def build_quartic_comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    """Create a compact quartic-family table for the superconductivity bar plot."""
    quartic = table[table["degree"].eq(4)].copy()
    pieces = []
    selectors = {
        "Global quartic": quartic[quartic["is_global"]],
        "Crisp quartic HPD": quartic[quartic["is_crisp_local"] & quartic["truncation"].str.contains("hpd", na=False)],
        "Fuzzy quartic DTD": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("dtd", na=False)],
        "Fuzzy quartic harmonic": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("harmonic", na=False)],
        "Fuzzy quartic SP": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("sp", na=False)],
        "Fuzzy quartic entropy": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("entropy", na=False)],
        "Fuzzy quartic HPD": quartic[quartic["is_fuzzy_local"] & quartic["truncation"].str.contains("hpd", na=False)],
    }
    for label, subset in selectors.items():
        best = _select_lowest_mse(subset, ["dataset"])
        if not best.empty:
            row = best.iloc[0].to_dict()
            row["display_method"] = label
            pieces.append(row)
    return pd.DataFrame(pieces)


def add_mixed_score(table: pd.DataFrame) -> pd.DataFrame:
    """Add the 6:4 mixed score using inverse MSE and R2."""
    table = table.copy()
    mse = table["mse"].astype(float)
    r2 = table["r2"].astype(float)
    inv_mse = 1.0 / mse.replace(0, np.nan)

    def normalize(values: pd.Series) -> pd.Series:
        if values.notna().sum() <= 1 or np.isclose(values.max(), values.min()):
            return pd.Series(np.ones(len(values)), index=values.index)
        return (values - values.min()) / (values.max() - values.min())

    table["normalized_inverse_mse"] = normalize(inv_mse)
    table["normalized_r2"] = normalize(r2)
    table["mixed_score_6_4"] = 0.6 * table["normalized_inverse_mse"] + 0.4 * table["normalized_r2"]
    return table


def _plot_degree_curves(curves: pd.DataFrame, output_path: Path) -> Path:
    if curves.empty:
        raise ValueError("The superconductivity degree-curve table is empty.")
    fig, ax = plt.subplots(figsize=(9, 5))
    for series, subset in curves.groupby("series"):
        subset = subset.sort_values("degree")
        ax.plot(subset["degree"], subset["mse"], marker="o", label=series)
    ax.set_xticks([1, 2, 3, 4], ["Linear", "Quadratic", "Cubic", "Quartic"])
    ax.set_xlabel("Local polynomial degree")
    ax.set_ylabel("Test MSE")
    ax.set_title("Superconductivity regression degree-wise comparison")
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
    plot_table = table.sort_values(value_col, ascending=value_col != "mixed_score_6_4")
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


def create_superconductivity_figures(input_path: Optional[Path] = None, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Path]:
    """Create the superconductivity CSV summaries and figure files."""
    output_dir = Path(output_dir)
    table = load_superconductivity_results(input_path)
    curves = build_degree_curve_table(table)
    quartic = build_quartic_comparison_table(table)
    quartic_scored = add_mixed_score(quartic) if not quartic.empty else quartic

    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / "03_superconductivity_degree_curves.csv"
    quartic_path = output_dir / "03_superconductivity_quartic_comparison.csv"
    curves.to_csv(curves_path, index=False)
    quartic_scored.to_csv(quartic_path, index=False)

    paths = {
        "degree_curve_table": curves_path,
        "quartic_table": quartic_path,
        "degree_curve_figure": _plot_degree_curves(curves, output_dir / "03_superconductivity_degree_mse.png"),
        "quartic_mse_figure": _plot_bar(quartic_scored, "mse", "Test MSE", "Superconductivity quartic-family comparison", output_dir / "03_superconductivity_quartic_mse.png"),
        "mixed_score_figure": _plot_bar(quartic_scored, "mixed_score_6_4", "Mixed score", "Superconductivity mixed-score comparison", output_dir / "03_superconductivity_mixed_score.png"),
    }
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create superconductivity regression result figures.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_superconductivity_figures(args.input, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

"""Audit regression metric consistency and optionally recompute metrics from predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from _shared import evaluate_predictions, load_raw_dataset


DEFAULT_OUTPUT = Path("outputs/reviewer_response/metric_audit")


def _first_present(columns, choices):
    for choice in choices:
        if choice in columns:
            return choice
    return None


def audit_metric_table(path: Path, relative_tolerance: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = set(frame.columns)
    mse_col = _first_present(columns, ["test_mse", "mse"])
    r2_col = _first_present(columns, ["test_r2", "r2", "test_R2", "R2"])
    if mse_col is None or r2_col is None:
        return pd.DataFrame()

    variance_col = _first_present(columns, ["test_target_variance", "target_variance"])
    dataset_col = _first_present(columns, ["dataset", "dataset_name"])
    n_col = _first_present(columns, ["n_test", "test_n", "n"])
    method_col = _first_present(columns, ["method_name", "method", "model", "variant"])
    split_col = _first_present(columns, ["seed", "random_state", "split_id", "repeat", "outer_split"])

    work = frame.copy()
    work["_audit_dataset"] = work[dataset_col].astype(str) if dataset_col else path.stem
    work["_audit_n"] = work[n_col].astype(str).fillna("<missing>") if n_col else "<unknown>"
    work["_audit_split"] = work[split_col].astype(str).fillna("<missing>") if split_col else "<single_split>"
    work["_audit_mse"] = pd.to_numeric(work[mse_col], errors="coerce")
    work["_audit_r2"] = pd.to_numeric(work[r2_col], errors="coerce")
    work["_audit_implied_variance"] = np.where(
        np.isfinite(work["_audit_r2"]) & (np.abs(1.0 - work["_audit_r2"]) > 1e-10),
        work["_audit_mse"] / (1.0 - work["_audit_r2"]),
        np.nan,
    )
    work["_audit_stored_variance"] = (
        pd.to_numeric(work[variance_col], errors="coerce") if variance_col else np.nan
    )
    group_cols = ["_audit_dataset", "_audit_n", "_audit_split"]
    records: List[Dict[str, Any]] = []

    group_stats = {}
    for key, group in work.groupby(group_cols, dropna=False):
        vals = group["_audit_implied_variance"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if vals.size:
            median = float(np.median(vals))
            rel_spread = float(np.max(np.abs(vals - median)) / max(abs(median), 1e-12))
        else:
            median, rel_spread = np.nan, np.nan
        normalized_key = tuple(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        group_stats[normalized_key] = (median, rel_spread, int(vals.size))

    for row_index, row in work.iterrows():
        key = (str(row["_audit_dataset"]), str(row["_audit_n"]), str(row["_audit_split"]))
        median, rel_spread, n_valid = group_stats.get(key, (np.nan, np.nan, 0))
        implied = row["_audit_implied_variance"]
        stored = row["_audit_stored_variance"]
        stored_mismatch = (
            bool(np.isfinite(stored) and np.isfinite(implied)
                 and abs(stored - implied) / max(abs(stored), 1e-12) > relative_tolerance)
            if variance_col else False
        )
        group_mismatch = bool(np.isfinite(rel_spread) and rel_spread > relative_tolerance and n_valid > 1)
        if not np.isfinite(implied):
            status = "not_checkable"
        elif stored_mismatch or group_mismatch:
            status = "flag_inconsistent"
        else:
            status = "consistent_with_group"
        method = str(row[method_col]) if method_col else f"row_{row_index}"
        records.append({
            "source_file": str(path),
            "row_index": int(row_index),
            "dataset": row["_audit_dataset"],
            "n_test": row["_audit_n"],
            "split_id": row["_audit_split"],
            "method": method,
            "mse_column": mse_col,
            "r2_column": r2_col,
            "mse": row["_audit_mse"],
            "r2": row["_audit_r2"],
            "implied_population_target_variance": implied,
            "stored_target_variance": stored,
            "group_median_implied_variance": median,
            "group_relative_spread": rel_spread,
            "stored_variance_mismatch": stored_mismatch,
            "group_variance_mismatch": group_mismatch,
            "status": status,
        })
    return pd.DataFrame(records)


def recompute_prediction_file(path: Path) -> pd.DataFrame:
    """Recompute metrics from row-level predictions produced by the new runner."""
    frame = pd.read_csv(path)
    required = {"y_true", "y_pred", "method"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    group_cols = [c for c in ["dataset", "task", "seed", "method"] if c in frame.columns]
    records = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = {"source_file": str(path)}
        record.update(dict(zip(group_cols, key)))
        task = str(group["task"].iloc[0]).lower() if "task" in group.columns else "regression"
        y_true = group["y_true"].to_numpy()
        if task == "regression":
            prediction = pd.to_numeric(group["y_pred"], errors="coerce").to_numpy(dtype=float)
            metrics = evaluate_predictions("regression", y_true.astype(float), prediction)
        else:
            probability_cols = sorted(
                [c for c in group.columns if c.startswith("prob_")],
                key=lambda name: int(name.split("_", 1)[1]),
            )
            if probability_cols:
                probs = group[probability_cols].to_numpy(dtype=float)
                metrics = evaluate_predictions("classification", y_true.astype(int), probs)
            else:
                predicted = group["y_pred"].to_numpy(dtype=int)
                metrics = {
                    "n": int(len(group)),
                    "accuracy": float(np.mean(predicted == y_true.astype(int))),
                    "cross_entropy": np.nan,
                }
        record.update(metrics)
        records.append(record)
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"),
                        help="Directory to scan for aggregate regression result CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--relative-tolerance", type=float, default=0.05,
                        help="Relative variance mismatch threshold; default 5 percent.")
    parser.add_argument("--predictions-csv", type=Path, default=None,
                        help="Optional row-level prediction CSV to recompute metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_frames = []
    input_root = Path(args.input_root)
    if input_root.exists():
        for path in sorted(input_root.rglob("*.csv")):
            try:
                if output_dir.resolve() in path.resolve().parents:
                    continue
                table = audit_metric_table(path, args.relative_tolerance)
                if not table.empty:
                    row_frames.append(table)
            except Exception as exc:
                print(f"[warning] Could not audit {path}: {type(exc).__name__}: {exc}")

    row_table = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    row_table.to_csv(output_dir / "aggregate_metric_audit.csv", index=False)

    if not row_table.empty:
        group_table = (
            row_table.groupby(["source_file", "dataset", "n_test", "split_id"], dropna=False, as_index=False)
            .agg(
                n_rows=("row_index", "count"),
                n_flagged=("status", lambda x: int((x == "flag_inconsistent").sum())),
                implied_variance_min=("implied_population_target_variance", "min"),
                implied_variance_median=("implied_population_target_variance", "median"),
                implied_variance_max=("implied_population_target_variance", "max"),
                max_relative_spread=("group_relative_spread", "max"),
            )
        )
    else:
        group_table = pd.DataFrame()
    group_table.to_csv(output_dir / "aggregate_metric_audit_groups.csv", index=False)

    if args.predictions_csv:
        prediction_metrics = recompute_prediction_file(args.predictions_csv)
        prediction_metrics.to_csv(output_dir / "recomputed_prediction_metrics.csv", index=False)
    else:
        prediction_metrics = pd.DataFrame()

    n_flagged = int((row_table["status"] == "flag_inconsistent").sum()) if not row_table.empty else 0
    n_checked = int(len(row_table))
    report_lines = [
        "Regression metric consistency audit",
        f"Input root: {input_root.resolve()}",
        f"Aggregate rows checked: {n_checked}",
        f"Rows flagged for inconsistent implied target variance: {n_flagged}",
        f"Relative tolerance: {args.relative_tolerance:.3f}",
        "",
        "Interpretation:",
        "- For a fixed test target vector, MSE and R2 imply the same population target variance: Var(y) = MSE / (1 - R2).",
        "- Large differences among rows with the same dataset and test size suggest inconsistent splits, target scales, or metric calculations.",
        "- Aggregate-only tables cannot fully verify predictions; use --predictions-csv to recompute metrics from row-level predictions.",
        "- This audit flags discrepancies for investigation; it does not silently overwrite published values.",
    ]
    (output_dir / "audit_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"Saved audit outputs to {output_dir}")


if __name__ == "__main__":
    main()

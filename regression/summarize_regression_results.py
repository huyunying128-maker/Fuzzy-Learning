"""
Regression result summaries for partition-weighted local learning.

The script combines regression result tables, selects the strongest rows by
mean squared error, and stores compact summary tables for reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics_utils import save_results, summarize_by_group

try:
    from regression_config import DATASET_SPECS
except ImportError:  # pragma: no cover
    from .regression_config import DATASET_SPECS


DEFAULT_FILENAMES = (
    "global_polynomial_baselines.csv",
    "local_regression_results.csv",
    "pw_regression_baselines.csv",
)


def existing_result_files(result_dir: str | Path, filenames: Iterable[str] = DEFAULT_FILENAMES) -> list[Path]:
    """Return the available regression result files in a directory."""
    directory = Path(result_dir)
    return [directory / name for name in filenames if (directory / name).exists()]


def load_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and combine CSV result tables."""
    frames: list[pd.DataFrame] = []
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _clean_ok_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with usable MSE values."""
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "status" in out.columns:
        out = out[(out["status"].isna()) | (out["status"].astype(str).str.lower() == "ok")]
    if "mse" in out.columns:
        out = out[pd.to_numeric(out["mse"], errors="coerce").notna()]
    return out.reset_index(drop=True)


def strongest_overall(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest row per dataset by MSE."""
    clean = _clean_ok_rows(frame)
    if clean.empty or "mse" not in clean.columns:
        return pd.DataFrame()
    return summarize_by_group(clean, ["dataset"], metric="mse", higher_is_better=False)


def strongest_by_family(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest row for each dataset and model family."""
    clean = _clean_ok_rows(frame)
    if clean.empty or "mse" not in clean.columns:
        return pd.DataFrame()

    family_col = "method_family" if "method_family" in clean.columns else None
    if family_col is None:
        clean = clean.copy()
        clean["method_family"] = clean.get("source_file", "unknown")
        family_col = "method_family"
    return summarize_by_group(clean, ["dataset", family_col], metric="mse", higher_is_better=False)


def strongest_local_by_degree(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest local-regression row for each dataset and degree."""
    clean = _clean_ok_rows(frame)
    required = {"dataset", "degree", "mse"}
    if clean.empty or not required.issubset(clean.columns):
        return pd.DataFrame()

    local_mask = pd.Series(True, index=clean.index)
    if "method_family" in clean.columns:
        local_mask = clean["method_family"].astype(str).str.contains("local", case=False, na=False)
    elif "source_file" in clean.columns:
        local_mask = clean["source_file"].astype(str).str.contains("local", case=False, na=False)

    local = clean[local_mask].copy()
    if local.empty:
        return pd.DataFrame()
    return summarize_by_group(local, ["dataset", "degree"], metric="mse", higher_is_better=False)


def write_summary_tables(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Save the main regression summary tables."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "combined": out_dir / "regression_combined_results.csv",
        "best_overall": out_dir / "regression_best_overall.csv",
        "best_by_family": out_dir / "regression_best_by_family.csv",
        "best_local_by_degree": out_dir / "regression_best_local_by_degree.csv",
    }

    if frame.empty:
        for path in outputs.values():
            pd.DataFrame().to_csv(path, index=False)
        return outputs

    frame.to_csv(outputs["combined"], index=False)
    strongest_overall(frame).to_csv(outputs["best_overall"], index=False)
    strongest_by_family(frame).to_csv(outputs["best_by_family"], index=False)
    strongest_local_by_degree(frame).to_csv(outputs["best_local_by_degree"], index=False)
    return outputs


def collect_default_results() -> pd.DataFrame:
    """Collect result tables from the default regression result directories."""
    paths: list[Path] = []
    for spec in DATASET_SPECS.values():
        paths.extend(existing_result_files(spec.default_result_path))
    return load_tables(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize regression result tables.")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-dir", default="results/regression/summary")
    args = parser.parse_args()

    if args.inputs:
        frame = load_tables(args.inputs)
    else:
        frame = collect_default_results()

    outputs = write_summary_tables(frame, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

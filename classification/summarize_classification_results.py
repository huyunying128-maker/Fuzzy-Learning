"""
Classification result summaries for MNIST experiments.

The script combines local-logit and feature-layer classifier results, selects
high-accuracy rows, and stores compact tables for reporting.
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

from metrics_utils import summarize_by_group

try:
    from classification_config import MNIST_SPEC
except ImportError:  # pragma: no cover
    from .classification_config import MNIST_SPEC


DEFAULT_FILENAMES = (
    "mnist_local_logit_results.csv",
    "mnist_pw_classifier_results.csv",
)


def existing_result_files(result_dir: str | Path, filenames: Iterable[str] = DEFAULT_FILENAMES) -> list[Path]:
    """Return the available MNIST classification result files in a directory."""
    directory = Path(result_dir)
    return [directory / name for name in filenames if (directory / name).exists()]


def load_tables(paths: Iterable[str | Path]) -> dict[str, pd.DataFrame]:
    """Load local-logit and feature-layer classifier tables."""
    tables: dict[str, pd.DataFrame] = {}
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        lower_name = path.name.lower()
        if "pw_classifier" in lower_name:
            tables["pw_classifiers"] = frame
        elif "local_logit" in lower_name:
            tables["local_logit"] = frame
        else:
            tables[path.stem] = frame
    return tables


def best_local_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest local-logit rows by accuracy."""
    if frame.empty or "accuracy" not in frame.columns:
        return pd.DataFrame()
    clean = frame[pd.to_numeric(frame["accuracy"], errors="coerce").notna()].copy()
    if clean.empty:
        return clean

    if "family" in clean.columns:
        by_family = summarize_by_group(clean, ["family"], metric="accuracy", higher_is_better=True)
    else:
        by_family = summarize_by_group(clean, ["dataset"], metric="accuracy", higher_is_better=True)
    return by_family.reset_index(drop=True)


def best_local_by_degree(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest local-logit row for each degree."""
    required = {"degree", "accuracy"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    clean = frame[pd.to_numeric(frame["accuracy"], errors="coerce").notna()].copy()
    return summarize_by_group(clean, ["degree"], metric="accuracy", higher_is_better=True)


def best_pw_classifiers(frame: pd.DataFrame) -> pd.DataFrame:
    """Select feature-layer classifier rows by classifier name."""
    required = {"classifier", "pw_accuracy"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    clean = frame[pd.to_numeric(frame["pw_accuracy"], errors="coerce").notna()].copy()
    return summarize_by_group(clean, ["classifier"], metric="pw_accuracy", higher_is_better=True)


def write_summary_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    """Save the main MNIST classification summary tables."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    local = tables.get("local_logit", pd.DataFrame())
    pw = tables.get("pw_classifiers", pd.DataFrame())

    outputs["best_local_by_family"] = out_dir / "mnist_best_local_by_family.csv"
    outputs["best_local_by_degree"] = out_dir / "mnist_best_local_by_degree.csv"
    outputs["best_pw_classifiers"] = out_dir / "mnist_best_pw_classifiers.csv"

    best_local_rows(local).to_csv(outputs["best_local_by_family"], index=False)
    best_local_by_degree(local).to_csv(outputs["best_local_by_degree"], index=False)
    best_pw_classifiers(pw).to_csv(outputs["best_pw_classifiers"], index=False)

    if not local.empty:
        local.to_csv(out_dir / "mnist_local_logit_combined.csv", index=False)
        outputs["local_combined"] = out_dir / "mnist_local_logit_combined.csv"
    if not pw.empty:
        pw.to_csv(out_dir / "mnist_pw_classifier_combined.csv", index=False)
        outputs["pw_combined"] = out_dir / "mnist_pw_classifier_combined.csv"

    return outputs


def collect_default_results() -> dict[str, pd.DataFrame]:
    """Collect result tables from the default MNIST result directory."""
    return load_tables(existing_result_files(MNIST_SPEC.default_result_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MNIST classification result tables.")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--output-dir", default="results/classification/summary")
    args = parser.parse_args()

    tables = load_tables(args.inputs) if args.inputs else collect_default_results()
    outputs = write_summary_tables(tables, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

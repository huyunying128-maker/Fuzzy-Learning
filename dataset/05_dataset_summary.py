"""
Dataset summary utilities for the experimental protocol.

The summary records the task type, sample counts, feature counts, and split
sizes used by the regression and classification experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def summarize_csv(input_path: Path, dataset_name: str, task: str, target: Optional[str]) -> Dict[str, object]:
    """Summarize a raw tabular CSV file."""
    df = pd.read_csv(input_path)
    if target is not None and target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found in {input_path}.")

    n_features = df.shape[1] - (1 if target is not None else 0)
    return {
        "dataset": dataset_name,
        "task": task,
        "source_file": str(input_path),
        "n_total": int(df.shape[0]),
        "n_features": int(n_features),
        "target": target,
    }


def summarize_npz_split(input_path: Path, dataset_name: Optional[str] = None, task: Optional[str] = None) -> Dict[str, object]:
    """Summarize a saved train-validation-test split file."""
    data = np.load(input_path)
    required = ("x_train", "x_valid", "x_test", "y_train", "y_valid", "y_test")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing arrays in {input_path}: {missing}")

    n_train = int(data["x_train"].shape[0])
    n_valid = int(data["x_valid"].shape[0])
    n_test = int(data["x_test"].shape[0])
    n_features = int(data["x_train"].shape[1])

    metadata_path = input_path.with_name(input_path.stem.replace("_split", "") + "_split_metadata.json")
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "dataset": dataset_name or metadata.get("dataset", input_path.stem),
        "task": task or metadata.get("task", ""),
        "source_file": str(input_path),
        "n_total": n_train + n_valid + n_test,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "n_features": n_features,
        "target": metadata.get("target"),
    }


def combine_summaries(records: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Combine dataset summary dictionaries into a table."""
    rows: List[Dict[str, object]] = list(records)
    if not rows:
        raise ValueError("At least one summary record is required.")
    return pd.DataFrame(rows)


def save_summary_table(summary: pd.DataFrame, output_path: Path) -> Path:
    """Save the dataset summary table as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset summary tables.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for index, input_path in enumerate(args.inputs):
        name = args.names[index] if args.names and index < len(args.names) else input_path.stem
        task = args.tasks[index] if args.tasks and index < len(args.tasks) else ""
        target = args.targets[index] if args.targets and index < len(args.targets) else None
        if input_path.suffix.lower() == ".npz":
            records.append(summarize_npz_split(input_path, dataset_name=name, task=task or None))
        else:
            records.append(summarize_csv(input_path, dataset_name=name, task=task, target=target))

    output_path = save_summary_table(combine_summaries(records), args.output)
    print(f"Saved dataset summary to {output_path}")


if __name__ == "__main__":
    main()

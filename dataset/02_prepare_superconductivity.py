"""
Prepare the Superconductivity critical-temperature dataset for the
partition-weighted regression experiments.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"
TARGET_COLUMN = "critical_temp"


def snake_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[()\[\],;/]+", " ", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def download_superconduct(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "superconduct.zip"
    urllib.request.urlretrieve(DEFAULT_URL, zip_path)
    return zip_path


def extract_if_needed(path: Path, raw_dir: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path

    extract_dir = raw_dir / "superconduct_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_train_csv(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".csv":
        return path

    candidates = list(path.rglob("train.csv"))
    if not candidates:
        candidates = [p for p in path.rglob("*.csv") if "unique" not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No superconductivity training CSV found under {path}")
    return candidates[0]


def clean_superconductivity_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_name(str(col)) for col in df.columns]

    if TARGET_COLUMN not in df.columns:
        possible_targets = [c for c in df.columns if "critical" in c and "temp" in c]
        if len(possible_targets) == 1:
            df = df.rename(columns={possible_targets[0]: TARGET_COLUMN})
        else:
            raise ValueError(f"Target column not found. Columns: {list(df.columns)}")

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    return df[feature_cols + [TARGET_COLUMN]]


def prepare_superconductivity(
    input_path: Optional[Path] = None,
    raw_dir: Path = Path("data/raw/superconductivity"),
    output_dir: Path = Path("data/processed"),
) -> Path:
    if input_path is None:
        input_path = download_superconduct(raw_dir)

    extracted = extract_if_needed(input_path, raw_dir)
    csv_path = find_train_csv(extracted)
    df = pd.read_csv(csv_path)
    df = clean_superconductivity_dataframe(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "superconductivity.csv"
    df.to_csv(output_path, index=False)

    metadata = {
        "dataset": "superconductivity",
        "task": "regression",
        "target": TARGET_COLUMN,
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "feature_columns": [c for c in df.columns if c != TARGET_COLUMN],
        "output_file": str(output_path),
    }
    with (output_dir / "superconductivity_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the superconductivity regression dataset.")
    parser.add_argument("--input", type=Path, default=None, help="Local ZIP or CSV file.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/superconductivity"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = prepare_superconductivity(args.input, args.raw_dir, args.output_dir)
    print(f"Saved prepared superconductivity data to {output_path}")


if __name__ == "__main__":
    main()

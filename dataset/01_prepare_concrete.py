"""
Prepare the Concrete Compressive Strength dataset for the partition-weighted
regression experiments.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
]

COLUMN_RENAME = {
    "cement_component_1_kg_in_a_m_3_mixture": "cement",
    "blast_furnace_slag_component_2_kg_in_a_m_3_mixture": "blast_furnace_slag",
    "fly_ash_component_3_kg_in_a_m_3_mixture": "fly_ash",
    "water_component_4_kg_in_a_m_3_mixture": "water",
    "superplasticizer_component_5_kg_in_a_m_3_mixture": "superplasticizer",
    "coarse_aggregate_component_6_kg_in_a_m_3_mixture": "coarse_aggregate",
    "fine_aggregate_component_7_kg_in_a_m_3_mixture": "fine_aggregate",
    "age_day": "age",
    "concrete_compressive_strength_mpa_megapascals": "concrete_compressive_strength",
}

TARGET_COLUMN = "concrete_compressive_strength"


def snake_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[()\[\],;/]+", " ", name)
    name = name.replace("^", "_")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def download_first_available(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for url in DEFAULT_URLS:
        suffix = Path(url).suffix or ".data"
        out_path = raw_dir / f"concrete_raw{suffix}"
        try:
            urllib.request.urlretrieve(url, out_path)
            return out_path
        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f"{url}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Concrete dataset download failed. Tried:\n{joined}")


def find_data_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = list(path.rglob("*.xls")) + list(path.rglob("*.xlsx")) + list(path.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No spreadsheet or CSV file found under {path}")
    return candidates[0]


def read_concrete_file(path: Path) -> pd.DataFrame:
    path = find_data_file(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return df


def clean_concrete_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_name(str(col)) for col in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_RENAME.items() if k in df.columns})

    if TARGET_COLUMN not in df.columns:
        possible_targets = [c for c in df.columns if "compressive" in c and "strength" in c]
        if len(possible_targets) == 1:
            df = df.rename(columns={possible_targets[0]: TARGET_COLUMN})
        else:
            raise ValueError(f"Target column not found. Columns: {list(df.columns)}")

    numeric_cols = df.columns.tolist()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    ordered_cols = feature_cols + [TARGET_COLUMN]
    return df[ordered_cols]


def prepare_concrete(
    input_path: Optional[Path] = None,
    raw_dir: Path = Path("data/raw/concrete"),
    output_dir: Path = Path("data/processed"),
) -> Path:
    if input_path is None:
        input_path = download_first_available(raw_dir)

    df = read_concrete_file(input_path)
    df = clean_concrete_dataframe(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "concrete.csv"
    df.to_csv(output_path, index=False)

    metadata = {
        "dataset": "concrete",
        "task": "regression",
        "target": TARGET_COLUMN,
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "feature_columns": [c for c in df.columns if c != TARGET_COLUMN],
        "output_file": str(output_path),
    }
    with (output_dir / "concrete_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the concrete regression dataset.")
    parser.add_argument("--input", type=Path, default=None, help="Local raw spreadsheet or CSV file.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/concrete"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = prepare_concrete(args.input, args.raw_dir, args.output_dir)
    print(f"Saved prepared concrete data to {output_path}")


if __name__ == "__main__":
    main()

"""
Preprocessing utilities for the regression datasets.

The module loads concrete and superconductivity data, creates reproducible
train-test splits, and applies standardization based on the training portion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CONCRETE_TARGET_CANDIDATES = (
    "concrete_compressive_strength",
    "Concrete compressive strength(MPa, megapascals)",
    "Concrete compressive strength",
)
SUPERCONDUCTIVITY_TARGET_CANDIDATES = (
    "critical_temp",
    "critical temperature",
    "tc",
)


@dataclass(frozen=True)
class RegressionDataBundle:
    """
    Container for a standardized regression split.
    """

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def clean_column_names(columns: Sequence[str]) -> list[str]:
    """
    Normalize column names to compact lowercase identifiers.
    """
    cleaned: list[str] = []
    for col in columns:
        value = str(col).strip().lower()
        for token in ["(", ")", "[", "]", ",", ";", "/", "-", ":"]:
            value = value.replace(token, " ")
        value = "_".join(value.split())
        cleaned.append(value)
    return cleaned


def find_target_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    """
    Find a target column using exact and normalized names.
    """
    exact_map = {str(col): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate in exact_map:
            return exact_map[candidate]

    normalized_columns = dict(zip(clean_column_names(frame.columns), frame.columns))
    normalized_candidates = clean_column_names(candidates)
    for candidate in normalized_candidates:
        if candidate in normalized_columns:
            return str(normalized_columns[candidate])

    raise KeyError("Target column was not found in the dataset.")


def read_table(path: str | Path) -> pd.DataFrame:
    """
    Read a CSV, XLS, or XLSX table.
    """
    table_path = Path(path)
    suffix = table_path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(table_path)
    elif suffix in {".xls", ".xlsx"}:
        frame = pd.read_excel(table_path)
    else:
        raise ValueError(f"Unsupported table format: {suffix}")

    frame = frame.copy()
    frame.columns = clean_column_names(frame.columns)
    return frame


def build_regression_bundle(
    frame: pd.DataFrame,
    target_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RegressionDataBundle:
    """
    Split a regression table and standardize the predictors.
    """
    if target_name not in frame.columns:
        raise KeyError(f"Target column not found: {target_name}")

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.dropna(axis=0).reset_index(drop=True)

    y = numeric_frame[target_name].to_numpy(dtype=float)
    x_frame = numeric_frame.drop(columns=[target_name])
    feature_names = list(x_frame.columns)
    x = x_frame.to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return RegressionDataBundle(
        x_train=x_train_scaled,
        x_test=x_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_name=target_name,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )


def prepare_concrete(
    path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RegressionDataBundle:
    """
    Prepare the concrete compressive strength dataset.
    """
    frame = read_table(path)
    target = find_target_column(frame, CONCRETE_TARGET_CANDIDATES)
    return build_regression_bundle(frame, target, test_size=test_size, random_state=random_state)


def prepare_superconductivity(
    path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RegressionDataBundle:
    """
    Prepare the superconductivity critical-temperature dataset.
    """
    frame = read_table(path)
    target = find_target_column(frame, SUPERCONDUCTIVITY_TARGET_CANDIDATES)
    return build_regression_bundle(frame, target, test_size=test_size, random_state=random_state)


def save_bundle(bundle: RegressionDataBundle, output_path: str | Path) -> Path:
    """
    Save a regression bundle as a compressed NumPy file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=bundle.x_train,
        x_test=bundle.x_test,
        y_train=bundle.y_train,
        y_test=bundle.y_test,
        feature_names=np.asarray(bundle.feature_names, dtype=object),
        target_name=np.asarray(bundle.target_name, dtype=object),
        scaler_mean=bundle.scaler_mean,
        scaler_scale=bundle.scaler_scale,
    )
    return path


def load_bundle(path: str | Path) -> RegressionDataBundle:
    """
    Load a compressed regression bundle.
    """
    data = np.load(path, allow_pickle=True)
    return RegressionDataBundle(
        x_train=data["x_train"],
        x_test=data["x_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        feature_names=list(data["feature_names"].astype(str)),
        target_name=str(data["target_name"].item()),
        scaler_mean=data["scaler_mean"],
        scaler_scale=data["scaler_scale"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a regression dataset for the experiments.")
    parser.add_argument("--dataset", choices=["concrete", "superconductivity"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "concrete":
        bundle = prepare_concrete(args.input, test_size=args.test_size, random_state=args.random_state)
    else:
        bundle = prepare_superconductivity(args.input, test_size=args.test_size, random_state=args.random_state)

    output_path = save_bundle(bundle, args.output)
    print(output_path)


if __name__ == "__main__":
    main()

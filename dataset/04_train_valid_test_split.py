"""
Create reproducible train, validation, and test splits for tabular datasets
and MNIST arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def split_arrays(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    valid_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    stratify: bool = False,
) -> Dict[str, np.ndarray]:
    stratify_values = y if stratify else None
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    inner_stratify = y_train_valid if stratify else None
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        test_size=valid_size,
        random_state=random_state,
        stratify=inner_stratify,
    )

    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "x_test": x_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }


def read_tabular_csv(input_path: Path, target: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(input_path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in {input_path}")

    feature_columns = [c for c in df.columns if c != target]
    x = df[feature_columns].to_numpy(dtype="float32")
    y = df[target].to_numpy(dtype="float32")
    return x, y, feature_columns


def read_mnist_npz(input_path: Path) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(input_path)
    x = data["x"].astype("float32")
    y = data["y"].astype("int64")
    feature_columns = [f"pixel_{i}" for i in range(x.shape[1])]
    return x, y, feature_columns


def scale_split(split_data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    scaler = StandardScaler()
    scaled = split_data.copy()
    scaled["x_train"] = scaler.fit_transform(split_data["x_train"])
    scaled["x_valid"] = scaler.transform(split_data["x_valid"])
    scaled["x_test"] = scaler.transform(split_data["x_test"])
    return scaled, scaler


def save_split(
    split_data: Dict[str, np.ndarray],
    output_dir: Path,
    dataset_name: str,
    feature_columns: list[str],
    task: str,
    target: Optional[str],
    scaler: Optional[StandardScaler] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{dataset_name}_split.npz"
    np.savez_compressed(npz_path, **split_data)

    metadata = {
        "dataset": dataset_name,
        "task": task,
        "target": target,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "n_train": int(split_data["x_train"].shape[0]),
        "n_valid": int(split_data["x_valid"].shape[0]),
        "n_test": int(split_data["x_test"].shape[0]),
        "split_file": str(npz_path),
        "scaler_file": None,
    }

    if scaler is not None:
        scaler_path = output_dir / f"{dataset_name}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        metadata["scaler_file"] = str(scaler_path)

    with (output_dir / f"{dataset_name}_split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npz_path


def create_split(
    input_path: Path,
    dataset_name: str,
    task: str,
    target: Optional[str] = None,
    output_dir: Path = Path("data/splits"),
    test_size: float = 0.20,
    valid_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    scale: bool = False,
) -> Path:
    if input_path.suffix.lower() == ".npz":
        x, y, feature_columns = read_mnist_npz(input_path)
    else:
        if target is None:
            raise ValueError("A target column is required for CSV input.")
        x, y, feature_columns = read_tabular_csv(input_path, target)

    split_data = split_arrays(
        x=x,
        y=y,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
        stratify=(task == "classification"),
    )

    scaler = None
    if scale:
        split_data, scaler = scale_split(split_data)

    return save_split(
        split_data=split_data,
        output_dir=output_dir,
        dataset_name=dataset_name,
        feature_columns=feature_columns,
        task=task,
        target=target,
        scaler=scaler,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train, validation, and test splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--scale", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = create_split(
        input_path=args.input,
        dataset_name=args.dataset_name,
        task=args.task,
        target=args.target,
        output_dir=args.output_dir,
        test_size=args.test_size,
        valid_size=args.valid_size,
        random_state=args.random_state,
        scale=args.scale,
    )
    print(f"Saved split arrays to {output_path}")


if __name__ == "__main__":
    main()

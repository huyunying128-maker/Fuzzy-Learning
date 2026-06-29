"""
Prepare the MNIST handwritten-digit data for the partition-weighted
classification experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


IMAGE_SHAPE = (28, 28)
N_PIXELS = 784


def load_mnist_from_keras() -> Tuple[np.ndarray, np.ndarray]:
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), _ = mnist.load_data()
    x = x_train.reshape(x_train.shape[0], N_PIXELS).astype("float32") / 255.0
    y = y_train.astype("int64")
    return x, y


def load_mnist_from_openml() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = data.data.astype("float32") / 255.0
    y = data.target.astype("int64")
    return x, y


def load_mnist(source: str) -> Tuple[np.ndarray, np.ndarray]:
    if source == "keras":
        return load_mnist_from_keras()
    if source == "openml":
        return load_mnist_from_openml()
    if source == "auto":
        try:
            return load_mnist_from_keras()
        except Exception:
            return load_mnist_from_openml()
    raise ValueError(f"Unknown MNIST source: {source}")


def save_optional_csv(x: np.ndarray, y: np.ndarray, output_dir: Path) -> Path:
    import pandas as pd

    columns = [f"pixel_{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=columns)
    df["label"] = y
    csv_path = output_dir / "mnist.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def prepare_mnist(
    source: str = "auto",
    output_dir: Path = Path("data/processed"),
    save_csv: bool = False,
) -> Path:
    x, y = load_mnist(source)

    if x.ndim != 2 or x.shape[1] != N_PIXELS:
        raise ValueError(f"Expected flattened MNIST matrix with 784 columns, got {x.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mnist_full.npz"
    np.savez_compressed(output_path, x=x, y=y)

    metadata = {
        "dataset": "mnist",
        "task": "classification",
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "image_shape": list(IMAGE_SHAPE),
        "n_classes": int(len(np.unique(y))),
        "pixel_scale": "[0, 1]",
        "output_file": str(output_path),
    }

    if save_csv:
        metadata["csv_file"] = str(save_optional_csv(x, y, output_dir))

    with (output_dir / "mnist_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the MNIST classification dataset.")
    parser.add_argument("--source", choices=["auto", "keras", "openml"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--save-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = prepare_mnist(args.source, args.output_dir, args.save_csv)
    print(f"Saved prepared MNIST data to {output_path}")


if __name__ == "__main__":
    main()

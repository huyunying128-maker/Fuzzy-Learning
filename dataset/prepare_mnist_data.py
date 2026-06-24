"""
Preprocessing utilities for the MNIST classification dataset.

The module loads handwritten-digit images, scales pixel values, creates a
reproducible train-test split, and stores the processed arrays in a compact
NumPy format.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class MNISTDataBundle:
    """Container for a processed MNIST train-test split."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    image_shape: tuple[int, int]
    n_classes: int


def _as_uint_or_float_images(x: np.ndarray) -> np.ndarray:
    """Return image data as a finite numeric array."""
    arr = np.asarray(x)
    if arr.ndim not in {2, 3}:
        raise ValueError("MNIST images must be a two- or three-dimensional array.")
    if not np.all(np.isfinite(arr.astype(float))):
        raise ValueError("MNIST images contain non-finite values.")
    return arr


def scale_pixels(x: np.ndarray) -> np.ndarray:
    """Scale pixel values to the interval [0, 1]."""
    arr = _as_uint_or_float_images(x).astype(np.float32)
    max_value = float(np.max(arr)) if arr.size else 0.0
    if max_value > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def flatten_images(x: np.ndarray) -> np.ndarray:
    """Flatten image tensors into two-dimensional feature matrices."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 2:
        return arr
    raise ValueError("Image array must have shape (n, height, width) or (n, features).")


def load_mnist_from_keras() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST using the TensorFlow Keras dataset interface."""
    try:
        from tensorflow.keras.datasets import mnist  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("TensorFlow Keras MNIST loader is unavailable.") from exc

    (x_train, y_train), _ = mnist.load_data()
    return x_train, y_train


def load_mnist_from_openml() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST using the scikit-learn OpenML interface."""
    try:
        from sklearn.datasets import fetch_openml
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn OpenML loader is unavailable.") from exc

    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = np.asarray(data.data)
    y = np.asarray(data.target, dtype=int)
    return x, y


def load_mnist_from_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST arrays from an NPZ file."""
    data = np.load(path, allow_pickle=True)

    if {"x", "y"}.issubset(data.files):
        return np.asarray(data["x"]), np.asarray(data["y"], dtype=int)

    if {"x_train", "y_train"}.issubset(data.files):
        return np.asarray(data["x_train"]), np.asarray(data["y_train"], dtype=int)

    raise KeyError("NPZ file must contain x/y or x_train/y_train arrays.")


def load_mnist_arrays(source: str = "keras", input_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST image and label arrays from a supported source."""
    if input_path is not None:
        return load_mnist_from_npz(input_path)
    if source == "keras":
        return load_mnist_from_keras()
    if source == "openml":
        return load_mnist_from_openml()
    raise ValueError("source must be 'keras' or 'openml'.")


def build_mnist_bundle(
    source: str = "keras",
    input_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    flatten: bool = True,
    n_samples: int | None = None,
) -> MNISTDataBundle:
    """Create a scaled and reproducible MNIST train-test split."""
    x_raw, y_raw = load_mnist_arrays(source=source, input_path=input_path)
    x_scaled = scale_pixels(x_raw)
    y = np.asarray(y_raw, dtype=int).reshape(-1)

    if x_scaled.shape[0] != y.shape[0]:
        raise ValueError("Image and label arrays must have the same number of rows.")

    if x_scaled.ndim == 3:
        image_shape = (int(x_scaled.shape[1]), int(x_scaled.shape[2]))
    else:
        side = int(round(np.sqrt(x_scaled.shape[1])))
        image_shape = (side, side)

    if n_samples is not None:
        if n_samples < 1 or n_samples > x_scaled.shape[0]:
            raise ValueError("n_samples must be between 1 and the number of images.")
        rng = np.random.default_rng(random_state)
        indices = rng.choice(x_scaled.shape[0], size=n_samples, replace=False)
        x_scaled = x_scaled[indices]
        y = y[indices]

    x_features = flatten_images(x_scaled) if flatten else x_scaled.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return MNISTDataBundle(
        x_train=x_train.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        image_shape=image_shape,
        n_classes=int(len(np.unique(y))),
    )


def save_bundle(bundle: MNISTDataBundle, output_path: str | Path) -> Path:
    """Save a processed MNIST bundle as a compressed NumPy file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=bundle.x_train,
        x_test=bundle.x_test,
        y_train=bundle.y_train,
        y_test=bundle.y_test,
        image_shape=np.asarray(bundle.image_shape, dtype=int),
        n_classes=np.asarray(bundle.n_classes, dtype=int),
    )
    return path


def load_bundle(path: str | Path) -> MNISTDataBundle:
    """Load a processed MNIST bundle from a compressed NumPy file."""
    data = np.load(path, allow_pickle=True)
    shape_arr = np.asarray(data["image_shape"], dtype=int)
    return MNISTDataBundle(
        x_train=np.asarray(data["x_train"], dtype=np.float32),
        x_test=np.asarray(data["x_test"], dtype=np.float32),
        y_train=np.asarray(data["y_train"], dtype=int),
        y_test=np.asarray(data["y_test"], dtype=int),
        image_shape=(int(shape_arr[0]), int(shape_arr[1])),
        n_classes=int(np.asarray(data["n_classes"]).item()),
    )


def bundle_summary(bundle: MNISTDataBundle) -> dict[str, Any]:
    """Return a compact summary of a processed MNIST bundle."""
    return {
        "n_train": int(bundle.x_train.shape[0]),
        "n_test": int(bundle.x_test.shape[0]),
        "n_features": int(bundle.x_train.shape[1]) if bundle.x_train.ndim == 2 else int(np.prod(bundle.x_train.shape[1:])),
        "image_shape": tuple(bundle.image_shape),
        "n_classes": int(bundle.n_classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the MNIST dataset for classification experiments.")
    parser.add_argument("--source", choices=["keras", "openml"], default="keras")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--keep-images", action="store_true")
    args = parser.parse_args()

    bundle = build_mnist_bundle(
        source=args.source,
        input_path=args.input,
        test_size=args.test_size,
        random_state=args.random_state,
        flatten=not args.keep_images,
        n_samples=args.n_samples,
    )
    output_path = save_bundle(bundle, args.output)
    print(output_path)


if __name__ == "__main__":
    main()

"""
MNIST membership-centroid standardization.

The visualization connects a handwritten digit to the ten class centroids. A
membership vector is computed from distances to the digit prototypes, and the
standardized output blends the original image, the dominant centroid, and the
membership-weighted centroid.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification" / "mnist"
IMAGE_SHAPE = (28, 28)
N_CLASSES = 10
EPSILON = 1e-12


def _load_core_module(file_name: str, alias: str):
    path = CORE_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_distance = _load_core_module("03_distance_functions.py", "fpw_centroid_distance")
_membership = _load_core_module("04_membership_functions.py", "fpw_centroid_membership")


def load_split_npz(split_path: Path) -> Dict[str, np.ndarray]:
    """Load MNIST split arrays saved by the dataset preparation scripts."""
    data = np.load(Path(split_path), allow_pickle=True)
    required = ("x_train", "y_train", "x_test", "y_test")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing arrays in split file: {missing}")
    return {name: data[name] for name in data.files}


def compute_class_centroids(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int = N_CLASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one average image for each digit label."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional image matrix.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    centroids = np.zeros((n_classes, X.shape[1]), dtype=float)
    counts = np.zeros(n_classes, dtype=int)
    global_mean = np.mean(X, axis=0)
    for label in range(n_classes):
        mask = y == label
        counts[label] = int(np.sum(mask))
        centroids[label] = np.mean(X[mask], axis=0) if np.any(mask) else global_mean
    return centroids, counts


def membership_to_digit_centroids(
    X: np.ndarray,
    centroids: np.ndarray,
    f: float = 1.30,
    p: float = 2.00,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fuzzy memberships from images to the ten digit centroids."""
    distance_table = _distance.minkowski_distance_matrix(X, centroids, p=float(p))
    membership = _membership.fuzzy_membership_from_distances(distance_table, f=float(f))
    return membership, distance_table


def standardize_images(
    X: np.ndarray,
    membership: np.ndarray,
    centroids: np.ndarray,
    f: float = 1.30,
    original_weight: float = 0.50,
    dominant_weight: float = 0.25,
    fuzzy_weight: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend input images with dominant and fuzzy membership centroids."""
    X = np.asarray(X, dtype=float)
    membership = np.asarray(membership, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    if X.ndim != 2 or membership.ndim != 2 or centroids.ndim != 2:
        raise ValueError("X, membership, and centroids must be two-dimensional arrays.")
    if X.shape[0] != membership.shape[0]:
        raise ValueError("X and membership must have the same number of rows.")
    if membership.shape[1] != centroids.shape[0]:
        raise ValueError("The membership columns must match the number of centroids.")

    total_weight = float(original_weight + dominant_weight + fuzzy_weight)
    if total_weight <= 0:
        raise ValueError("At least one blending weight must be positive.")

    aggregation = _membership.aggregation_weights(membership, f=float(f))
    fuzzy_centroids = aggregation @ centroids
    dominant_labels = np.argmax(membership, axis=1).astype(int)
    dominant_centroids = centroids[dominant_labels]
    blended = (
        original_weight * X
        + dominant_weight * dominant_centroids
        + fuzzy_weight * fuzzy_centroids
    ) / total_weight
    return np.clip(blended, 0.0, 1.0), dominant_labels, fuzzy_centroids


def save_centroid_panel(
    input_image: np.ndarray,
    standardized_image: np.ndarray,
    centroids: np.ndarray,
    output_path: Path,
    true_label: Optional[int] = None,
    dominant_label: Optional[int] = None,
) -> Path:
    """Save the input, standardized output, and ten digit centroids in one figure."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [input_image, standardized_image] + [centroids[label] for label in range(N_CLASSES)]
    titles = ["Input", "Standardized"] + [f"Centroid {label}" for label in range(N_CLASSES)]
    if true_label is not None:
        titles[0] = f"Input label {int(true_label)}"
    if dominant_label is not None:
        titles[1] = f"Standardized / dom {int(dominant_label)}"

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(np.asarray(image).reshape(IMAGE_SHAPE), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_mnist_centroid_standardization(
    split_path: Path,
    output_dir: Path = OUTPUT_DIR,
    image_index: int = 0,
    f: float = 1.30,
    p: float = 2.00,
    original_weight: float = 0.50,
    dominant_weight: float = 0.25,
    fuzzy_weight: float = 0.25,
) -> Dict[str, Path]:
    """Create the membership-centroid standardization outputs for MNIST."""
    split = load_split_npz(split_path)
    x_train = np.asarray(split["x_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=int)
    x_test = np.asarray(split["x_test"], dtype=float)
    y_test = np.asarray(split["y_test"], dtype=int)
    if not 0 <= int(image_index) < x_test.shape[0]:
        raise ValueError("image_index is outside the test set.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centroids, counts = compute_class_centroids(x_train, y_train, n_classes=N_CLASSES)
    sample = x_test[[int(image_index)]]
    membership, distance_table = membership_to_digit_centroids(sample, centroids, f=float(f), p=float(p))
    standardized, dominant_labels, fuzzy_centroids = standardize_images(
        sample,
        membership,
        centroids,
        f=float(f),
        original_weight=original_weight,
        dominant_weight=dominant_weight,
        fuzzy_weight=fuzzy_weight,
    )

    npz_path = output_dir / "07_mnist_centroid_standardization.npz"
    np.savez_compressed(
        npz_path,
        input_image=sample[0],
        standardized_image=standardized[0],
        centroids=centroids,
        membership=membership[0],
        distance_table=distance_table[0],
        fuzzy_centroid=fuzzy_centroids[0],
        centroid_counts=counts,
        true_label=np.array([int(y_test[int(image_index)])]),
        dominant_label=np.array([int(dominant_labels[0])]),
        f=np.array([float(f)]),
        p=np.array([float(p)]),
    )

    membership_path = output_dir / "07_mnist_centroid_membership.csv"
    pd.DataFrame(
        {
            "digit": np.arange(N_CLASSES, dtype=int),
            "membership": membership[0],
            "distance": distance_table[0],
            "centroid_count": counts,
        }
    ).to_csv(membership_path, index=False)

    figure_path = output_dir / "07_mnist_centroid_standardization.png"
    save_centroid_panel(
        input_image=sample[0],
        standardized_image=standardized[0],
        centroids=centroids,
        output_path=figure_path,
        true_label=int(y_test[int(image_index)]),
        dominant_label=int(dominant_labels[0]),
    )

    return {"npz": npz_path, "membership_csv": membership_path, "figure": figure_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the MNIST membership-centroid standardization example.")
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--f", type=float, default=1.30)
    parser.add_argument("--p", type=float, default=2.00)
    parser.add_argument("--original-weight", type=float, default=0.50)
    parser.add_argument("--dominant-weight", type=float, default=0.25)
    parser.add_argument("--fuzzy-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_mnist_centroid_standardization(
        split_path=args.split_path,
        output_dir=args.output_dir,
        image_index=args.image_index,
        f=args.f,
        p=args.p,
        original_weight=args.original_weight,
        dominant_weight=args.dominant_weight,
        fuzzy_weight=args.fuzzy_weight,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

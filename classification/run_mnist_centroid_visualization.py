"""
MNIST membership-centroid visualization.

The script builds digit centroids, computes fuzzy prototype weights for a sample
image, and saves a figure showing the input, standardized output, and ten class
centroids.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.prepare_mnist_data import build_mnist_bundle, load_bundle, save_bundle
from fpwl_core import fuzzy_membership, minkowski_distance_table

try:
    from classification_config import MNIST_SPEC, RANDOM_STATE
except ImportError:  # pragma: no cover
    from .classification_config import MNIST_SPEC, RANDOM_STATE


def class_centroids(x: np.ndarray, y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Compute one mean image vector for each digit class."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same number of rows.")

    centroids = np.zeros((n_classes, x_arr.shape[1]), dtype=float)
    global_mean = x_arr.mean(axis=0)
    for label in range(n_classes):
        mask = y_arr == label
        centroids[label] = x_arr[mask].mean(axis=0) if np.any(mask) else global_mean
    return centroids


def membership_to_class_centroids(x: np.ndarray, centroids: np.ndarray, f: float = 1.3, p: float = 2.0) -> np.ndarray:
    """Compute fuzzy memberships from images to digit centroids."""
    distances = minkowski_distance_table(x, centroids, p=p)
    return fuzzy_membership(distances, f=f)


def standardized_digit(x: np.ndarray, centroids: np.ndarray, membership: np.ndarray, f: float = 1.3, blend: float = 0.5) -> np.ndarray:
    """Blend one image with its membership-weighted class centroid."""
    x_vec = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(membership, dtype=float).reshape(-1)
    weights = np.power(np.maximum(u, 1.0e-12), f)
    weights = weights / weights.sum()
    prototype = weights @ centroids
    return (1.0 - blend) * x_vec + blend * prototype


def choose_sample_index(y: np.ndarray, label: int | None = None, sample_index: int | None = None) -> int:
    """Choose an image index for the visualization."""
    if sample_index is not None:
        return int(sample_index)
    if label is None:
        return 0
    matches = np.flatnonzero(np.asarray(y, dtype=int).reshape(-1) == int(label))
    if len(matches) == 0:
        return 0
    return int(matches[0])


def save_centroid_figure(
    input_image: np.ndarray,
    output_image: np.ndarray,
    centroids: np.ndarray,
    image_shape: tuple[int, int],
    output_path: str | Path,
) -> Path:
    """Save the MNIST centroid-standardization figure."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5))
    axes = []
    axes.append(fig.add_subplot(2, 6, 1))
    axes.append(fig.add_subplot(2, 6, 2))
    for pos in range(3, 13):
        axes.append(fig.add_subplot(2, 6, pos))

    images = [input_image, output_image] + [centroids[j] for j in range(10)]
    titles = ["Input", "Standardized"] + [str(j) for j in range(10)]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(np.asarray(image).reshape(image_shape), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def load_or_prepare_bundle(bundle_path: Path, source: str, n_samples: int | None, random_state: int):
    """Load an existing MNIST bundle or create it from the selected source."""
    if bundle_path.exists():
        return load_bundle(bundle_path)
    bundle = build_mnist_bundle(source=source, n_samples=n_samples, random_state=random_state)
    save_bundle(bundle, bundle_path)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an MNIST centroid-standardization figure.")
    parser.add_argument("--bundle", default=str(MNIST_SPEC.default_bundle_path))
    parser.add_argument("--source", choices=["keras", "openml"], default="keras")
    parser.add_argument("--output", default="results/classification/mnist/mnist_centroid_visualization.png")
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--f", type=float, default=1.3)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()

    bundle = load_or_prepare_bundle(Path(args.bundle), args.source, args.n_samples, args.random_state)
    centroids = class_centroids(bundle.x_train, bundle.y_train, n_classes=10)
    idx = choose_sample_index(bundle.y_test, label=args.label, sample_index=args.sample_index)

    x = bundle.x_test[idx]
    membership = membership_to_class_centroids(x.reshape(1, -1), centroids, f=args.f, p=args.p)[0]
    output = standardized_digit(x, centroids, membership, f=args.f, blend=args.blend)
    path = save_centroid_figure(x, output, centroids, bundle.image_shape, args.output)
    print(path)


if __name__ == "__main__":
    main()

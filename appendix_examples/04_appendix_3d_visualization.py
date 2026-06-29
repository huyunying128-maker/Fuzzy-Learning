"""
Three-dimensional visualization for the Appendix A clustering example.

The figure compares the initial points and centroids, the crisp centroid path,
the fuzzy centroid path, and the final crisp-versus-fuzzy centroid positions.
It is a visual companion to the Appendix A numerical tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/appendix_examples")
FIGURE_NAME = "04_appendix_3d_visualization.png"


def _load_peer_module(file_name: str, alias: str):
    path = Path(__file__).resolve().with_name(file_name)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _ensure_appendix_outputs(output_dir: Path) -> None:
    crisp_arrays = output_dir / "02_crisp_appendix_arrays.npz"
    fuzzy_arrays = output_dir / "03_fuzzy_appendix_arrays.npz"
    crisp_path = output_dir / "02_crisp_centroid_path.csv"
    fuzzy_path = output_dir / "03_fuzzy_centroid_path.csv"
    if crisp_arrays.exists() and fuzzy_arrays.exists() and crisp_path.exists() and fuzzy_path.exists():
        return

    crisp_module = _load_peer_module("02_crisp_clustering_example.py", "fpw_appendix_crisp")
    fuzzy_module = _load_peer_module("03_fuzzy_clustering_example.py", "fpw_appendix_fuzzy")
    crisp_module.run_crisp_appendix_example(output_dir=output_dir)
    fuzzy_module.run_fuzzy_appendix_example(output_dir=output_dir)


def _load_centroid_path(path: Path) -> Dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    result: Dict[str, np.ndarray] = {}
    for centroid, group in frame.groupby("centroid"):
        ordered = group.sort_values("iteration")
        result[str(centroid)] = ordered[["feature_1", "feature_2", "feature_3"]].to_numpy(dtype=float)
    return result


def _scatter_points(ax, X: np.ndarray, labels=None, sizes=None, title: str = "") -> None:
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=45)
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=sizes if sizes is not None else 45)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")


def _plot_paths(ax, paths: Dict[str, np.ndarray], marker: str = "x", label_prefix: str = "") -> None:
    for centroid, values in paths.items():
        ax.plot(values[:, 0], values[:, 1], values[:, 2], marker=marker, linewidth=1.5, label=f"{label_prefix}{centroid}")


def create_appendix_3d_visualization(output_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
    """Create the Appendix A four-panel 3D visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_appendix_outputs(output_dir)

    appendix_data = _load_peer_module("01_appendix_points_and_centroids.py", "fpw_appendix_data")
    X, initial_centroids, _ = appendix_data.get_appendix_arrays()

    crisp_arrays = np.load(output_dir / "02_crisp_appendix_arrays.npz")
    fuzzy_arrays = np.load(output_dir / "03_fuzzy_appendix_arrays.npz")
    crisp_labels = crisp_arrays["labels"][-1]
    fuzzy_labels = fuzzy_arrays["labels"][-1]
    fuzzy_membership = fuzzy_arrays["memberships"][-1]
    strongest_membership = np.max(fuzzy_membership, axis=1)

    crisp_paths = _load_centroid_path(output_dir / "02_crisp_centroid_path.csv")
    fuzzy_paths = _load_centroid_path(output_dir / "03_fuzzy_centroid_path.csv")

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _scatter_points(ax1, X, title="Initial observations and centroids")
    ax1.scatter(initial_centroids[:, 0], initial_centroids[:, 1], initial_centroids[:, 2], marker="x", s=100)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _scatter_points(ax2, X, labels=crisp_labels, title="Crisp partition and centroid path")
    _plot_paths(ax2, crisp_paths, marker="o", label_prefix="crisp ")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    sizes = 40.0 + 120.0 * strongest_membership
    _scatter_points(ax3, X, labels=fuzzy_labels, sizes=sizes, title="Fuzzy partition and centroid path")
    _plot_paths(ax3, fuzzy_paths, marker="^", label_prefix="fuzzy ")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    _scatter_points(ax4, X, title="Final crisp and fuzzy centroids")
    for centroid, values in crisp_paths.items():
        ax4.scatter(values[-1, 0], values[-1, 1], values[-1, 2], marker="o", s=90, label=f"crisp {centroid}")
    for centroid, values in fuzzy_paths.items():
        ax4.scatter(values[-1, 0], values[-1, 1], values[-1, 2], marker="^", s=90, label=f"fuzzy {centroid}")

    for ax in (ax1, ax2, ax3, ax4):
        ax.view_init(elev=22, azim=-58)

    ax4.legend(loc="best", fontsize=8)
    fig.tight_layout()

    figure_path = output_dir / FIGURE_NAME
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"figure": figure_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Appendix A 3D visualization.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_appendix_3d_visualization(args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

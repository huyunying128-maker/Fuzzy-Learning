"""
Appendix A crisp clustering computation.

The script reproduces the hard nearest-centroid example in Appendix A. It keeps
one table for each visible step: distance tables, nearest-centroid labels, hard
membership tables, centroid updates, and the truncation check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/appendix_examples")
DISPLAY_ITERATIONS = 3


def _load_peer_module(file_name: str, alias: str):
    path = Path(__file__).resolve().with_name(file_name)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _minkowski_distance_matrix(X: np.ndarray, centroids: np.ndarray, p: float = 2.0) -> np.ndarray:
    diff = np.abs(X[:, None, :] - centroids[None, :, :])
    return np.sum(diff ** p, axis=2) ** (1.0 / p)


def _hard_membership(labels: np.ndarray, k: int) -> np.ndarray:
    membership = np.zeros((labels.size, k), dtype=int)
    membership[np.arange(labels.size), labels] = 1
    return membership


def _update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    previous_centroids: np.ndarray,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    centroids = np.array(previous_centroids, copy=True, dtype=float)
    rows: List[Dict[str, object]] = []
    for j in range(k):
        mask = labels == j
        assigned = np.where(mask)[0]
        if assigned.size > 0:
            centroids[j] = np.mean(X[mask], axis=0)
        rows.append(
            {
                "centroid": f"c{j + 1}",
                "assigned_points": ", ".join(f"x{i + 1}" for i in assigned),
                "feature_1": centroids[j, 0],
                "feature_2": centroids[j, 1],
                "feature_3": centroids[j, 2],
            }
        )
    return centroids, rows


def _wide_iteration_table(
    blocks: Dict[int, np.ndarray],
    point_names: List[str],
    value_prefix: str,
    decimals: int = 2,
) -> pd.DataFrame:
    table = pd.DataFrame({"point": point_names})
    for t, values in blocks.items():
        for j in range(values.shape[1]):
            column = f"t{t}_{value_prefix}{j + 1}"
            table[column] = np.round(values[:, j], decimals)
    return table


def _labels_table(label_blocks: Dict[int, np.ndarray], point_names: List[str]) -> pd.DataFrame:
    table = pd.DataFrame({"point": point_names})
    for t, labels in label_blocks.items():
        table[f"t{t}_label"] = labels.astype(int) + 1
    return table


def run_crisp_appendix_example(
    output_dir: Path = OUTPUT_DIR,
    displayed_iterations: int = DISPLAY_ITERATIONS,
) -> Dict[str, Path]:
    """Run the Appendix A crisp example and export all displayed tables."""
    appendix_data = _load_peer_module("01_appendix_points_and_centroids.py", "fpw_appendix_data")
    X, initial_centroids, parameters = appendix_data.get_appendix_arrays()
    p = float(parameters["distance_order_p"])
    k = int(parameters["groups_k"])
    point_names = list(appendix_data.POINT_NAMES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centroids = np.array(initial_centroids, copy=True, dtype=float)
    distance_blocks: Dict[int, np.ndarray] = {}
    label_blocks: Dict[int, np.ndarray] = {}
    membership_blocks: Dict[int, np.ndarray] = {}
    update_rows: List[Dict[str, object]] = []
    truncation_rows: List[Dict[str, object]] = []
    centroid_path_rows: List[Dict[str, object]] = []

    previous_labels = None
    for t in range(displayed_iterations):
        distances = _minkowski_distance_matrix(X, centroids, p=p)
        labels = np.argmin(distances, axis=1)
        membership = _hard_membership(labels, k)

        distance_blocks[t] = distances
        label_blocks[t] = labels
        membership_blocks[t] = membership

        for j, centroid in enumerate(centroids):
            centroid_path_rows.append(
                {
                    "iteration": t,
                    "centroid": f"c{j + 1}",
                    "feature_1": centroid[0],
                    "feature_2": centroid[1],
                    "feature_3": centroid[2],
                }
            )

        new_centroids, rows = _update_centroids(X, labels, k, centroids)
        for row in rows:
            row = dict(row)
            row["iteration"] = t
            update_rows.append(row)

        if previous_labels is None:
            result = "initial partition"
            changed_points = ""
        else:
            changed = np.where(previous_labels != labels)[0]
            result = "stable" if changed.size == 0 else "not stable"
            changed_points = ", ".join(f"x{i + 1}" for i in changed)
        truncation_rows.append(
            {
                "after_iteration": t,
                "comparison": "labels against previous displayed labels",
                "result": result,
                "changed_points": changed_points,
            }
        )

        previous_labels = labels.copy()
        centroids = new_centroids

    distances_path = output_dir / "02_crisp_distance_tables.csv"
    labels_path = output_dir / "02_crisp_labels.csv"
    membership_path = output_dir / "02_crisp_memberships.csv"
    updates_path = output_dir / "02_crisp_centroid_updates.csv"
    truncation_path = output_dir / "02_crisp_truncation_checks.csv"
    centroid_path = output_dir / "02_crisp_centroid_path.csv"
    npz_path = output_dir / "02_crisp_appendix_arrays.npz"
    metadata_path = output_dir / "02_crisp_metadata.json"

    _wide_iteration_table(distance_blocks, point_names, "d", decimals=2).to_csv(distances_path, index=False)
    _labels_table(label_blocks, point_names).to_csv(labels_path, index=False)
    _wide_iteration_table(membership_blocks, point_names, "u", decimals=0).to_csv(membership_path, index=False)
    pd.DataFrame(update_rows).to_csv(updates_path, index=False)
    pd.DataFrame(truncation_rows).to_csv(truncation_path, index=False)
    pd.DataFrame(centroid_path_rows).to_csv(centroid_path, index=False)
    np.savez_compressed(
        npz_path,
        distances=np.stack([distance_blocks[t] for t in sorted(distance_blocks)]),
        labels=np.stack([label_blocks[t] for t in sorted(label_blocks)]),
        memberships=np.stack([membership_blocks[t] for t in sorted(membership_blocks)]),
    )
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "crisp nearest-centroid partition",
                "displayed_iterations": displayed_iterations,
                "distance_order_p": p,
                "groups_k": k,
            },
            f,
            indent=2,
        )

    return {
        "distance_tables": distances_path,
        "labels": labels_path,
        "memberships": membership_path,
        "centroid_updates": updates_path,
        "truncation_checks": truncation_path,
        "centroid_path": centroid_path,
        "arrays": npz_path,
        "metadata": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Appendix A crisp clustering example.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--displayed-iterations", type=int, default=DISPLAY_ITERATIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_crisp_appendix_example(args.output_dir, args.displayed_iterations)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

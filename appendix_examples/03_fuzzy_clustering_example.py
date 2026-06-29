"""
Appendix A fuzzy clustering computation.

The script reproduces the fuzzy-membership example in Appendix A. It exports the
same visible objects used in the paper: distance tables, reciprocal closeness,
normalized memberships, squared membership weights, centroid updates, and the
truncation check.
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
EPSILON = 1e-12


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


def _reciprocal_closeness(distances: np.ndarray, f: float) -> np.ndarray:
    exponent = 2.0 / (f - 1.0)
    safe_distances = np.maximum(distances, EPSILON)
    return 1.0 / (safe_distances ** exponent)


def _normalize_membership(scores: np.ndarray) -> np.ndarray:
    row_sums = np.sum(scores, axis=1, keepdims=True)
    return scores / np.maximum(row_sums, EPSILON)


def _update_fuzzy_centroids(
    X: np.ndarray,
    membership: np.ndarray,
    f: float,
    previous_centroids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    weights = membership ** f
    centroids = np.array(previous_centroids, copy=True, dtype=float)
    rows: List[Dict[str, object]] = []
    for j in range(membership.shape[1]):
        total_weight = float(np.sum(weights[:, j]))
        if total_weight > EPSILON:
            centroids[j] = np.sum(weights[:, [j]] * X, axis=0) / total_weight
        rows.append(
            {
                "centroid": f"c{j + 1}",
                "sum_weight": total_weight,
                "feature_1": centroids[j, 0],
                "feature_2": centroids[j, 1],
                "feature_3": centroids[j, 2],
            }
        )
    return centroids, weights, rows


def _dominant_labels(membership: np.ndarray) -> np.ndarray:
    return np.argmax(membership, axis=1)


def _wide_iteration_table(
    blocks: Dict[int, np.ndarray],
    point_names: List[str],
    value_prefix: str,
    decimals: int = 3,
) -> pd.DataFrame:
    table = pd.DataFrame({"point": point_names})
    for t, values in blocks.items():
        for j in range(values.shape[1]):
            column = f"t{t}_{value_prefix}{j + 1}"
            table[column] = np.round(values[:, j], decimals)
    return table


def run_fuzzy_appendix_example(
    output_dir: Path = OUTPUT_DIR,
    displayed_iterations: int = DISPLAY_ITERATIONS,
) -> Dict[str, Path]:
    """Run the Appendix A fuzzy example and export all displayed tables."""
    appendix_data = _load_peer_module("01_appendix_points_and_centroids.py", "fpw_appendix_data")
    X, initial_centroids, parameters = appendix_data.get_appendix_arrays()
    p = float(parameters["distance_order_p"])
    f = float(parameters["fuzzy_degree_f"])
    k = int(parameters["groups_k"])
    point_names = list(appendix_data.POINT_NAMES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centroids = np.array(initial_centroids, copy=True, dtype=float)
    distance_blocks: Dict[int, np.ndarray] = {}
    closeness_blocks: Dict[int, np.ndarray] = {}
    membership_blocks: Dict[int, np.ndarray] = {}
    weight_blocks: Dict[int, np.ndarray] = {}
    label_blocks: Dict[int, np.ndarray] = {}
    update_rows: List[Dict[str, object]] = []
    truncation_rows: List[Dict[str, object]] = []
    centroid_path_rows: List[Dict[str, object]] = []

    previous_membership = None
    previous_labels = None
    for t in range(displayed_iterations):
        distances = _minkowski_distance_matrix(X, centroids, p=p)
        closeness = _reciprocal_closeness(distances, f=f)
        membership = _normalize_membership(closeness)
        labels = _dominant_labels(membership)
        new_centroids, weights, rows = _update_fuzzy_centroids(X, membership, f, centroids)

        distance_blocks[t] = distances
        closeness_blocks[t] = 1000.0 * closeness
        membership_blocks[t] = membership
        weight_blocks[t] = weights
        label_blocks[t] = labels

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

        for row in rows:
            row = dict(row)
            row["iteration"] = t
            update_rows.append(row)

        if previous_membership is None:
            result = "initial fuzzy partition"
            max_membership_change = float("nan")
            dominant_label_changes = ""
        else:
            max_membership_change = float(np.max(np.abs(membership - previous_membership)))
            changed = np.where(labels != previous_labels)[0]
            result = "dominant memberships preserved" if changed.size == 0 else "dominant memberships changed"
            dominant_label_changes = ", ".join(f"x{i + 1}" for i in changed)
        truncation_rows.append(
            {
                "after_iteration": t,
                "comparison": "membership and dominant labels against previous displayed iteration",
                "result": result,
                "max_membership_change": max_membership_change,
                "dominant_label_changes": dominant_label_changes,
            }
        )

        previous_membership = membership.copy()
        previous_labels = labels.copy()
        centroids = new_centroids

    distances_path = output_dir / "03_fuzzy_distance_tables.csv"
    closeness_path = output_dir / "03_fuzzy_scaled_reciprocal_closeness.csv"
    membership_path = output_dir / "03_fuzzy_memberships.csv"
    weights_path = output_dir / "03_fuzzy_squared_membership_weights.csv"
    updates_path = output_dir / "03_fuzzy_centroid_updates.csv"
    truncation_path = output_dir / "03_fuzzy_truncation_checks.csv"
    centroid_path = output_dir / "03_fuzzy_centroid_path.csv"
    npz_path = output_dir / "03_fuzzy_appendix_arrays.npz"
    metadata_path = output_dir / "03_fuzzy_metadata.json"

    _wide_iteration_table(distance_blocks, point_names, "d", decimals=2).to_csv(distances_path, index=False)
    _wide_iteration_table(closeness_blocks, point_names, "r1000_", decimals=3).to_csv(closeness_path, index=False)
    _wide_iteration_table(membership_blocks, point_names, "u", decimals=3).to_csv(membership_path, index=False)
    _wide_iteration_table(weight_blocks, point_names, "w", decimals=3).to_csv(weights_path, index=False)
    pd.DataFrame(update_rows).to_csv(updates_path, index=False)
    pd.DataFrame(truncation_rows).to_csv(truncation_path, index=False)
    pd.DataFrame(centroid_path_rows).to_csv(centroid_path, index=False)
    np.savez_compressed(
        npz_path,
        distances=np.stack([distance_blocks[t] for t in sorted(distance_blocks)]),
        closeness_1000=np.stack([closeness_blocks[t] for t in sorted(closeness_blocks)]),
        memberships=np.stack([membership_blocks[t] for t in sorted(membership_blocks)]),
        weights=np.stack([weight_blocks[t] for t in sorted(weight_blocks)]),
        labels=np.stack([label_blocks[t] for t in sorted(label_blocks)]),
    )
    with metadata_path.open("w", encoding="utf-8") as f_meta:
        json.dump(
            {
                "method": "fuzzy membership partition",
                "displayed_iterations": displayed_iterations,
                "distance_order_p": p,
                "fuzzy_degree_f": f,
                "groups_k": k,
            },
            f_meta,
            indent=2,
        )

    return {
        "distance_tables": distances_path,
        "scaled_reciprocal_closeness": closeness_path,
        "memberships": membership_path,
        "squared_membership_weights": weights_path,
        "centroid_updates": updates_path,
        "truncation_checks": truncation_path,
        "centroid_path": centroid_path,
        "arrays": npz_path,
        "metadata": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Appendix A fuzzy clustering example.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--displayed-iterations", type=int, default=DISPLAY_ITERATIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_fuzzy_appendix_example(args.output_dir, args.displayed_iterations)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

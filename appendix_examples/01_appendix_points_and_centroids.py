"""
Appendix A data for the crisp and fuzzy clustering examples.

The example uses twelve three-dimensional observations, three initial centroids,
three groups, Euclidean distance, and fuzzy degree f = 2. The exported CSV files
provide the starting point for the appendix crisp and fuzzy computations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/appendix_examples")


POINT_NAMES = [f"x{i}" for i in range(1, 13)]
CENTROID_NAMES = ["c1_initial", "c2_initial", "c3_initial"]
FEATURE_NAMES = ["feature_1", "feature_2", "feature_3"]


APPENDIX_POINTS = np.array(
    [
        [11.2, 35.4, 18.6],
        [17.8, 29.1, 24.7],
        [26.5, 42.8, 16.9],
        [31.7, 31.2, 29.4],
        [74.1, 88.6, 62.5],
        [89.4, 79.2, 77.8],
        [98.8, 102.3, 68.1],
        [83.5, 115.7, 91.6],
        [146.3, 48.5, 121.7],
        [163.9, 63.4, 139.2],
        [181.5, 39.8, 128.6],
        [155.6, 78.9, 151.3],
    ],
    dtype=float,
)


INITIAL_CENTROIDS = np.array(
    [
        [8.0, 28.0, 12.0],
        [45.0, 120.0, 105.0],
        [115.0, 40.0, 115.0],
    ],
    dtype=float,
)


APPENDIX_PARAMETERS = {
    "dimension": 3,
    "groups_k": 3,
    "distance_order_p": 2.0,
    "fuzzy_degree_f": 2.0,
    "displayed_iterations": [0, 1, 2],
}


def appendix_points_dataframe() -> pd.DataFrame:
    """Return the Appendix A observation table."""
    table = pd.DataFrame(APPENDIX_POINTS, columns=FEATURE_NAMES)
    table.insert(0, "point", POINT_NAMES)
    return table


def initial_centroids_dataframe() -> pd.DataFrame:
    """Return the Appendix A initial-centroid table."""
    table = pd.DataFrame(INITIAL_CENTROIDS, columns=FEATURE_NAMES)
    table.insert(0, "centroid", CENTROID_NAMES)
    return table


def parameters_dataframe() -> pd.DataFrame:
    """Return the Appendix A parameter table."""
    rows = [
        {"quantity": "dimension", "value": APPENDIX_PARAMETERS["dimension"]},
        {"quantity": "groups_k", "value": APPENDIX_PARAMETERS["groups_k"]},
        {"quantity": "distance_order_p", "value": APPENDIX_PARAMETERS["distance_order_p"]},
        {"quantity": "fuzzy_degree_f", "value": APPENDIX_PARAMETERS["fuzzy_degree_f"]},
        {"quantity": "displayed_iterations", "value": "0,1,2"},
    ]
    return pd.DataFrame(rows)


def get_appendix_arrays() -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Return observations, initial centroids, and parameter metadata."""
    return APPENDIX_POINTS.copy(), INITIAL_CENTROIDS.copy(), dict(APPENDIX_PARAMETERS)


def export_appendix_inputs(output_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
    """Write Appendix A observations, centroids, and parameters to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = output_dir / "01_appendix_points.csv"
    centroids_path = output_dir / "01_appendix_initial_centroids.csv"
    parameters_path = output_dir / "01_appendix_parameters.csv"
    metadata_path = output_dir / "01_appendix_metadata.json"
    npz_path = output_dir / "01_appendix_inputs.npz"

    appendix_points_dataframe().to_csv(points_path, index=False)
    initial_centroids_dataframe().to_csv(centroids_path, index=False)
    parameters_dataframe().to_csv(parameters_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(APPENDIX_PARAMETERS, f, indent=2)
    np.savez_compressed(npz_path, points=APPENDIX_POINTS, centroids=INITIAL_CENTROIDS)

    return {
        "points": points_path,
        "initial_centroids": centroids_path,
        "parameters": parameters_path,
        "metadata": metadata_path,
        "npz": npz_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Appendix A starting data.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = export_appendix_inputs(args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

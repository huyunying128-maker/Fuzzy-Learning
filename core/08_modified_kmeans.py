"""
Modified k-means reference used in the comparison branch.

The reference first searches for a hard nearest-centroid partition over k and p.
After this hard partition is fixed, a fuzzy degree can be selected to create a
soft distance-derived feature layer for downstream prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import importlib.util
import sys

import numpy as np


EPSILON = 1e-12


def _load_core_module(file_name: str, alias: str):
    path = Path(__file__).resolve().with_name(file_name)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_dist = _load_core_module("03_distance_functions.py", "fpw_distance_functions")
_membership = _load_core_module("04_membership_functions.py", "fpw_membership_functions")
_crisp = _load_core_module("06_crisp_partition.py", "fpw_crisp_partition")


@dataclass
class ModifiedKMeansResult:
    """Selected hard partition and optional post-partition fuzzy layer."""

    k: int
    p: float
    f: Optional[float]
    centroids: np.ndarray
    labels: np.ndarray
    hard_membership: np.ndarray
    fuzzy_membership: Optional[np.ndarray]
    distance_table: np.ndarray
    objective: float
    n_iter: int
    search_records: List[Dict[str, float]]
    f_search_records: List[Dict[str, float]]


def _as_2d_float(array, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    return value


def _clean_grid(values: Iterable[float], name: str) -> List[float]:
    grid = sorted({float(v) for v in values})
    if not grid:
        raise ValueError(f"{name} must contain at least one value.")
    return grid


def fit_hard_kmeans_reference(
    X,
    k: int,
    p: float,
    max_iter: int = 300,
    tolerance: float = 1e-6,
    random_state: int = 42,
) -> ModifiedKMeansResult:
    """Fit one hard nearest-centroid reference for a fixed k and p."""
    fitted = _crisp.fit_crisp_partition(
        X,
        k=k,
        p=p,
        max_iter=max_iter,
        tolerance=tolerance,
        truncation_rule="hpd",
        random_state=random_state,
    )
    return ModifiedKMeansResult(
        k=int(k),
        p=float(p),
        f=None,
        centroids=fitted.centroids,
        labels=fitted.labels,
        hard_membership=fitted.membership,
        fuzzy_membership=None,
        distance_table=fitted.distance_table,
        objective=float(fitted.objective),
        n_iter=int(fitted.n_iter),
        search_records=[],
        f_search_records=[],
    )


def search_hard_kp_reference(
    X,
    k_values: Sequence[int],
    p_values: Sequence[float],
    max_iter: int = 300,
    tolerance: float = 1e-6,
    random_state: int = 42,
    max_candidates: Optional[int] = None,
) -> ModifiedKMeansResult:
    """Search k and p by the hard nearest-centroid distance objective."""
    X = _as_2d_float(X, "X")
    k_grid = [int(k) for k in k_values if 2 <= int(k) <= X.shape[0]]
    p_grid = _clean_grid(p_values, "p_values")
    if not k_grid:
        raise ValueError("k_values must contain at least one feasible cluster count.")

    candidate_pairs: List[Tuple[int, float]] = [(k, p) for k in k_grid for p in p_grid]
    if max_candidates is not None:
        candidate_pairs = candidate_pairs[: int(max_candidates)]

    best: Optional[ModifiedKMeansResult] = None
    records: List[Dict[str, float]] = []
    for run_index, (k, p) in enumerate(candidate_pairs):
        fitted = fit_hard_kmeans_reference(
            X,
            k=k,
            p=p,
            max_iter=max_iter,
            tolerance=tolerance,
            random_state=random_state + run_index,
        )
        records.append(
            {
                "k": float(k),
                "p": float(p),
                "objective": float(fitted.objective),
                "n_iter": float(fitted.n_iter),
            }
        )
        if best is None or fitted.objective < best.objective:
            best = fitted

    if best is None:
        raise RuntimeError("No modified k-means candidate was fitted.")
    best.search_records = records
    return best


def add_distance_based_fuzzy_layer(
    hard_result: ModifiedKMeansResult,
    f: float,
) -> ModifiedKMeansResult:
    """Attach a fuzzy membership layer to a fixed hard k-means partition."""
    fuzzy_membership = _membership.fuzzy_membership_from_distances(hard_result.distance_table, f=f)
    return ModifiedKMeansResult(
        k=hard_result.k,
        p=hard_result.p,
        f=float(f),
        centroids=hard_result.centroids,
        labels=hard_result.labels,
        hard_membership=hard_result.hard_membership,
        fuzzy_membership=fuzzy_membership,
        distance_table=hard_result.distance_table,
        objective=hard_result.objective,
        n_iter=hard_result.n_iter,
        search_records=list(hard_result.search_records),
        f_search_records=list(hard_result.f_search_records),
    )


def learn_fuzzy_degree_from_callback(
    hard_result: ModifiedKMeansResult,
    f_values: Sequence[float],
    score_callback: Callable[[float, np.ndarray], float],
    greater_is_better: bool = False,
) -> ModifiedKMeansResult:
    """Select f for the fixed hard partition through an external validation score."""
    f_grid = _clean_grid(f_values, "f_values")
    best_f: Optional[float] = None
    best_score: Optional[float] = None
    records: List[Dict[str, float]] = []

    for f in f_grid:
        fuzzy_membership = _membership.fuzzy_membership_from_distances(hard_result.distance_table, f=f)
        score = float(score_callback(f, fuzzy_membership))
        records.append({"f": float(f), "score": score})
        if best_score is None:
            best_f = f
            best_score = score
        elif greater_is_better and score > best_score:
            best_f = f
            best_score = score
        elif not greater_is_better and score < best_score:
            best_f = f
            best_score = score

    if best_f is None:
        raise RuntimeError("No fuzzy-degree candidate was evaluated.")

    result = add_distance_based_fuzzy_layer(hard_result, f=best_f)
    result.f_search_records = records
    return result


def search_modified_kmeans_reference(
    X,
    k_values: Sequence[int],
    p_values: Sequence[float],
    f_values: Optional[Sequence[float]] = None,
    f_score_callback: Optional[Callable[[float, np.ndarray], float]] = None,
    greater_is_better: bool = False,
    max_iter: int = 300,
    tolerance: float = 1e-6,
    random_state: int = 42,
    max_candidates: Optional[int] = None,
) -> ModifiedKMeansResult:
    """Run the full modified k-means reference search."""
    hard_result = search_hard_kp_reference(
        X,
        k_values=k_values,
        p_values=p_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
    )

    if f_values is None or f_score_callback is None:
        return hard_result

    return learn_fuzzy_degree_from_callback(
        hard_result,
        f_values=f_values,
        score_callback=f_score_callback,
        greater_is_better=greater_is_better,
    )


def transform_with_modified_kmeans(X, result: ModifiedKMeansResult, use_fuzzy: bool = True):
    """Compute hard or fuzzy memberships for new observations from fixed centroids."""
    distance_table = _dist.minkowski_distance_matrix(X, result.centroids, p=result.p)
    if use_fuzzy and result.f is not None:
        membership = _membership.fuzzy_membership_from_distances(distance_table, f=result.f)
        labels = _membership.dominant_membership_labels(membership)
    else:
        labels = _dist.nearest_centroid_labels(distance_table)
        membership = _membership.hard_membership_from_labels(labels, result.k)
    return membership, labels, distance_table


if __name__ == "__main__":
    sample = np.array([[1.0, 1.0], [1.2, 0.9], [5.0, 5.1], [5.2, 4.8]])
    result = search_modified_kmeans_reference(sample, k_values=[2], p_values=[1.0, 2.0])
    print("selected:", result.k, result.p, round(result.objective, 6))

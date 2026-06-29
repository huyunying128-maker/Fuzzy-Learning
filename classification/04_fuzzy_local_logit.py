"""
Fuzzy local-logit classification for MNIST.

The fuzzy branch keeps a graded membership vector for every image, builds local
membership-weighted inputs, fits local multinomial logit models, and aggregates
local logits using normalized fuzzy weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
CLASSIFICATION_DIR = PROJECT_ROOT / "classification"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification" / "mnist"


def _load_core_module(file_name: str, alias: str):
    path = CORE_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_classification_module(file_name: str, alias: str):
    path = CLASSIFICATION_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_config = _load_core_module("01_config.py", "fpw_fuzzy_logit_config")
_metrics = _load_core_module("02_metrics.py", "fpw_fuzzy_logit_metrics")
_fuzzy = _load_core_module("07_fuzzy_partition.py", "fpw_fuzzy_logit_partition")
_local = _load_classification_module("03_crisp_local_logit.py", "fpw_shared_local_logit")


def candidate_iterator(
    k_values: Sequence[int],
    p_values: Sequence[float],
    f_values: Sequence[float],
    max_candidates: Optional[int],
):
    """Iterate through candidate k, p, and f triples."""
    count = 0
    for k in k_values:
        for p in p_values:
            for f in f_values:
                yield int(k), float(p), float(f)
                count += 1
                if max_candidates is not None and count >= int(max_candidates):
                    return


def evaluate_fuzzy_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    degree: int,
    k: int,
    p: float,
    f: float,
    truncation: str,
    max_iter_partition: int,
    tolerance: float,
    random_state: int,
    c_value: float,
    max_iter_logit: int,
    solver: str,
    max_base_features: Optional[int],
    interaction_features: int,
) -> Tuple[float, Dict[str, float], object, Sequence[object]]:
    """Fit and validate one fuzzy local-logit candidate."""
    partition = _fuzzy.fit_fuzzy_partition(
        x_train,
        k=int(k),
        f=float(f),
        p=float(p),
        max_iter=max_iter_partition,
        tolerance=tolerance,
        truncation_rule=truncation,
        random_state=random_state,
    )
    valid_membership, _, _ = _fuzzy.predict_fuzzy_partition(
        x_valid,
        partition.centroids,
        f=float(f),
        p=float(p),
    )
    models = _local.fit_local_logit_models(
        x_train,
        y_train,
        partition.membership,
        degree=degree,
        f=float(f),
        c_value=c_value,
        max_iter=max_iter_logit,
        solver=solver,
        max_base_features=max_base_features,
        interaction_features=interaction_features,
    )
    probabilities = _local.predict_local_logit_probabilities(
        models,
        x_valid,
        valid_membership,
        f=float(f),
        n_classes=10,
    )
    metrics = _metrics.classification_metrics(y_valid, probabilities=probabilities)
    return float(metrics["cross_entropy"]), metrics, partition, models


def run_fuzzy_local_logit(
    split_path: Path,
    output_path: Path = OUTPUT_DIR / "04_fuzzy_local_logit.csv",
    degrees: Sequence[int] = _config.LOCAL_DEGREES,
    truncations: Sequence[str] = _config.TRUNCATION_RULES,
    k_values_text: Optional[str] = "840,1200,1560,1920",
    p_values_text: Optional[str] = "1.10,1.15,1.20,1.25",
    f_values_text: Optional[str] = "1.10,1.20,1.30",
    max_candidates: Optional[int] = None,
    max_iter_partition: int = 300,
    tolerance: float = 1e-6,
    random_state: int = 42,
    c_value: float = 1.0,
    max_iter_logit: int = 1000,
    solver: str = "lbfgs",
    max_base_features: Optional[int] = 256,
    interaction_features: int = 0,
    search_record_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the fuzzy local-logit grid and save the selected test rows."""
    split = _local.load_split_npz(split_path)
    x_train = np.asarray(split["x_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=int)
    x_valid = np.asarray(split["x_valid"], dtype=float)
    y_valid = np.asarray(split["y_valid"], dtype=int)
    x_test = np.asarray(split["x_test"], dtype=float)
    y_test = np.asarray(split["y_test"], dtype=int)

    default_k, default_p, default_f = _local.default_partition_grids(x_train.shape[0])
    k_values = _local.parse_numeric_grid(k_values_text, default_k, cast=int)
    p_values = _local.parse_numeric_grid(p_values_text, default_p, cast=float)
    f_values = _local.parse_numeric_grid(f_values_text, default_f, cast=float)

    records: List[Dict[str, float]] = []
    search_records: List[Dict[str, float]] = []

    for degree in degrees:
        for truncation in truncations:
            start = time.perf_counter()
            best = None
            best_loss = float("inf")
            best_metrics: Dict[str, float] = {}
            for run_index, (k, p, f) in enumerate(candidate_iterator(k_values, p_values, f_values, max_candidates)):
                try:
                    loss, valid_metrics, partition, _ = evaluate_fuzzy_candidate(
                        x_train,
                        y_train,
                        x_valid,
                        y_valid,
                        degree=int(degree),
                        k=k,
                        p=p,
                        f=f,
                        truncation=truncation,
                        max_iter_partition=max_iter_partition,
                        tolerance=tolerance,
                        random_state=random_state + run_index,
                        c_value=c_value,
                        max_iter_logit=max_iter_logit,
                        solver=solver,
                        max_base_features=max_base_features,
                        interaction_features=interaction_features,
                    )
                    search_records.append(
                        {
                            "degree": int(degree),
                            "truncation": truncation,
                            "k": int(k),
                            "p": float(p),
                            "f": float(f),
                            "validation_cross_entropy": float(loss),
                            "validation_accuracy": float(valid_metrics["accuracy"]),
                            "status": "ok",
                        }
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best = (int(k), float(p), float(f), partition)
                        best_metrics = valid_metrics
                except Exception as exc:  # keeps long grids auditable
                    search_records.append(
                        {
                            "degree": int(degree),
                            "truncation": truncation,
                            "k": int(k),
                            "p": float(p),
                            "f": float(f),
                            "validation_cross_entropy": np.nan,
                            "validation_accuracy": np.nan,
                            "status": f"failed: {exc}",
                        }
                    )

            if best is None:
                continue

            selected_k, selected_p, selected_f, _ = best
            x_full, y_full = _local.combine_train_valid(split)
            final_partition = _fuzzy.fit_fuzzy_partition(
                x_full,
                k=selected_k,
                f=selected_f,
                p=selected_p,
                max_iter=max_iter_partition,
                tolerance=tolerance,
                truncation_rule=truncation,
                random_state=random_state,
            )
            test_membership, _, _ = _fuzzy.predict_fuzzy_partition(
                x_test,
                final_partition.centroids,
                f=selected_f,
                p=selected_p,
            )
            final_models = _local.fit_local_logit_models(
                x_full,
                y_full,
                final_partition.membership,
                degree=int(degree),
                f=selected_f,
                c_value=c_value,
                max_iter=max_iter_logit,
                solver=solver,
                max_base_features=max_base_features,
                interaction_features=interaction_features,
            )
            probabilities = _local.predict_local_logit_probabilities(
                final_models,
                x_test,
                test_membership,
                f=selected_f,
                n_classes=10,
            )
            test_metrics = _metrics.classification_metrics(y_test, probabilities=probabilities)
            elapsed = _metrics.summarize_elapsed_seconds(start, time.perf_counter())

            records.append(
                {
                    "dataset": "mnist",
                    "task": "classification",
                    "method_family": "fuzzy_local_logit",
                    "method_name": "Fuzzy local logit",
                    "degree": int(degree),
                    "truncation": truncation,
                    "k": int(selected_k),
                    "f": float(selected_f),
                    "p": float(selected_p),
                    "validation_accuracy": float(best_metrics.get("accuracy", np.nan)),
                    "validation_cross_entropy": float(best_loss),
                    "test_accuracy": float(test_metrics["accuracy"]),
                    "test_cross_entropy": float(test_metrics["cross_entropy"]),
                    "n_iter": int(final_partition.n_iter),
                    "runtime_sec": elapsed,
                    "n_design_features": int(final_models[0].spec.n_output_features) if final_models else 0,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output_path, index=False)

    if search_record_path is not None:
        search_record_path = Path(search_record_path)
        search_record_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(search_records).to_csv(search_record_path, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MNIST fuzzy local-logit models.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "04_fuzzy_local_logit.csv")
    parser.add_argument("--search-records", type=Path, default=None)
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--truncations", default="dtd,harmonic,sp,entropy,hpd")
    parser.add_argument("--k-values", default="840,1200,1560,1920")
    parser.add_argument("--p-values", default="1.10,1.15,1.20,1.25")
    parser.add_argument("--f-values", default="1.10,1.20,1.30")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-iter-partition", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter-logit", type=int, default=1000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--max-base-features", type=int, default=256)
    parser.add_argument("--interaction-features", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_fuzzy_local_logit(
        split_path=args.split,
        output_path=args.output,
        degrees=_local.parse_degrees(args.degrees),
        truncations=_local.parse_truncations(args.truncations),
        k_values_text=args.k_values,
        p_values_text=args.p_values,
        f_values_text=args.f_values,
        max_candidates=args.max_candidates,
        max_iter_partition=args.max_iter_partition,
        tolerance=args.tolerance,
        random_state=args.random_state,
        c_value=args.c_value,
        max_iter_logit=args.max_iter_logit,
        solver=args.solver,
        max_base_features=args.max_base_features,
        interaction_features=args.interaction_features,
        search_record_path=args.search_records,
    )
    print(f"Saved fuzzy local-logit table with {table.shape[0]} rows")


if __name__ == "__main__":
    main()

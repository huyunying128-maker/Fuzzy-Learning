"""
Concrete regression experiment runner.

The runner executes the concrete regression block in a modular way and saves a
combined table for the global polynomial, crisp local, fuzzy local, partition-only,
fixed-iteration, modified k-means, and external-regressor comparisons.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_DIR = PROJECT_ROOT / "regression"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "regression" / "concrete"


def _load_regression_module(file_name: str, alias: str):
    path = REGRESSION_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_global = _load_regression_module("01_global_polynomial_baselines.py", "fpw_reg_global")
_crisp = _load_regression_module("02_crisp_local_regression.py", "fpw_reg_crisp")
_fuzzy = _load_regression_module("03_fuzzy_local_regression.py", "fpw_reg_fuzzy")
_partition = _load_regression_module("04_partition_only_regression.py", "fpw_reg_partition_only")
_fixed = _load_regression_module("05_fixed_iteration_regression.py", "fpw_reg_fixed_iteration")
_modified = _load_regression_module("06_modified_kmeans_regression.py", "fpw_reg_modified_kmeans")
_external = _load_regression_module("07_external_regressors.py", "fpw_reg_external")


def parse_degrees(text: str) -> List[int]:
    """Parse polynomial degrees used by the regression branch."""
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one degree is required.")
    return values


def parse_names(text: str) -> List[str]:
    """Parse a comma-separated list of model or truncation names."""
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one name is required.")
    return values


def _read_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def _combine_existing_tables(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = _read_if_exists(path)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def run_concrete_experiment(
    split_path: Path,
    output_dir: Path = OUTPUT_DIR,
    degrees: Optional[List[int]] = None,
    truncations: Optional[List[str]] = None,
    k_values: Optional[str] = None,
    p_values: Optional[str] = None,
    f_values: Optional[str] = None,
    external_models: Optional[List[str]] = None,
    fixed_iterations: Optional[List[int]] = None,
    max_iter: int = 300,
    tolerance: float = 1e-6,
    random_state: int = 42,
    max_candidates: Optional[int] = None,
    run_external: bool = True,
) -> pd.DataFrame:
    """Run the concrete regression experiment and return the combined table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    degrees = degrees if degrees is not None else [1, 2, 3, 4]
    truncations = truncations if truncations is not None else ["dtd", "harmonic", "sp", "entropy", "hpd"]
    external_models = external_models if external_models is not None else [
        "ann",
        "adam_ann",
        "dl",
        "cnn1d",
        "svr",
        "random_forest",
        "xgboost",
    ]
    fixed_iterations = fixed_iterations if fixed_iterations is not None else [8]

    paths = {
        "global": output_dir / "01_global_polynomial_baselines.csv",
        "crisp": output_dir / "02_crisp_local_regression.csv",
        "fuzzy": output_dir / "03_fuzzy_local_regression.csv",
        "partition": output_dir / "04_partition_only_regression.csv",
        "fixed": output_dir / "05_fixed_iteration_regression.csv",
        "modified": output_dir / "06_modified_kmeans_regression.csv",
        "external": output_dir / "07_external_regressors.csv",
        "combined": output_dir / "concrete_regression_all_results.csv",
    }

    _global.run_global_polynomial_baselines(
        split_path=split_path,
        dataset_name="concrete",
        output_path=paths["global"],
        degrees=degrees,
    )
    _crisp.run_crisp_local_regression(
        split_path=split_path,
        dataset_name="concrete",
        output_path=paths["crisp"],
        degrees=degrees,
        truncations=truncations,
        k_values_text=k_values,
        p_values_text=p_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
        search_record_path=output_dir / "02_crisp_local_search_records.csv",
    )
    _fuzzy.run_fuzzy_local_regression(
        split_path=split_path,
        dataset_name="concrete",
        output_path=paths["fuzzy"],
        degrees=degrees,
        truncations=truncations,
        k_values_text=k_values,
        p_values_text=p_values,
        f_values_text=f_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
        search_record_path=output_dir / "03_fuzzy_local_search_records.csv",
    )
    _partition.run_partition_only_regression(
        split_path=split_path,
        dataset_name="concrete",
        output_path=paths["partition"],
        partition_types=["crisp", "fuzzy"],
        truncations=truncations,
        k_values_text=k_values,
        p_values_text=p_values,
        f_values_text=f_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
        search_record_path=output_dir / "04_partition_only_search_records.csv",
    )
    _fixed.run_fixed_iteration_regression(
        split_path=split_path,
        dataset_name="concrete",
        output_path=paths["fixed"],
        fixed_iterations=fixed_iterations,
        k_values_text=k_values,
        p_values_text=p_values,
        f_values_text=f_values,
        random_state=random_state,
        max_candidates=max_candidates,
        search_record_path=output_dir / "05_fixed_iteration_search_records.csv",
    )
    _modified.run_modified_kmeans_regression(
        split_path=split_path,
        dataset_name="concrete",
        output_path=paths["modified"],
        k_values_text=k_values,
        p_values_text=p_values,
        f_values_text=f_values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        max_candidates=max_candidates,
        search_record_path=output_dir / "06_modified_kmeans_search_records.csv",
    )
    if run_external:
        _external.run_external_regressors(
            split_path=split_path,
            dataset_name="concrete",
            output_path=paths["external"],
            model_names=external_models,
            k_values_text=k_values,
            p_values_text=p_values,
            f_values_text=f_values,
            max_iter=max_iter,
            tolerance=tolerance,
            random_state=random_state,
            max_candidates=max_candidates,
        )

    combined_paths = [
        paths["global"],
        paths["crisp"],
        paths["fuzzy"],
        paths["partition"],
        paths["fixed"],
        paths["modified"],
    ]
    if run_external:
        combined_paths.append(paths["external"])
    combined = _combine_existing_tables(combined_paths)
    combined.to_csv(paths["combined"], index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the concrete regression experiment block.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--truncations", default="dtd,harmonic,sp,entropy,hpd")
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--f-values", default=None)
    parser.add_argument("--external-models", default="ann,adam_ann,dl,cnn1d,svr,random_forest,xgboost")
    parser.add_argument("--fixed-iterations", default="8")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--skip-external", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_concrete_experiment(
        split_path=args.split,
        output_dir=args.output_dir,
        degrees=parse_degrees(args.degrees),
        truncations=parse_names(args.truncations),
        k_values=args.k_values,
        p_values=args.p_values,
        f_values=args.f_values,
        external_models=parse_names(args.external_models),
        fixed_iterations=[int(value) for value in parse_names(args.fixed_iterations)],
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        random_state=args.random_state,
        max_candidates=args.max_candidates,
        run_external=not args.skip_external,
    )
    print(f"Saved concrete combined result table with {table.shape[0]} rows")


if __name__ == "__main__":
    main()

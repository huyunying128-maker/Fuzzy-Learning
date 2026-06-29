"""
Run the MNIST classification experiment pipeline.

The pipeline follows the paper order: clustering-only references, raw logit
baselines, crisp local logits, fuzzy local logits, modified k-means and
partition-weighted feature layers, external classifiers, the centroid
standardization visualization, and the final MNIST result collector.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_DIR = PROJECT_ROOT / "classification"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification" / "mnist"


def _load_classification_module(file_name: str, alias: str):
    path = CLASSIFICATION_DIR / file_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _parse_csv_list(text: str) -> Sequence[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("The list cannot be empty.")
    return values


def _selected_steps(step_text: str) -> set[str]:
    if str(step_text).strip().lower() == "all":
        return {
            "clustering",
            "raw_logit",
            "crisp_local",
            "fuzzy_local",
            "features",
            "external",
            "centroid",
            "collect",
        }
    return {part.strip().lower() for part in str(step_text).split(",") if part.strip()}


def run_mnist_experiment(
    split_path: Path,
    output_dir: Path = OUTPUT_DIR,
    steps: str = "all",
    degrees: Sequence[int] = (1, 2, 3, 4),
    truncations: Sequence[str] = ("dtd", "harmonic", "sp", "entropy", "hpd"),
    clustering_k: str = "10",
    clustering_p: str = "2.00",
    clustering_f: str = "1.30",
    crisp_k: str = "760,1020,1400,1750",
    crisp_p: str = "2.00",
    fuzzy_k: str = "840,1200,1560,1920",
    fuzzy_p: str = "1.10,1.15,1.20,1.25",
    fuzzy_f: str = "1.10,1.20,1.30",
    feature_k: str = "10",
    feature_p: str = "2.00",
    feature_f: str = "1.30",
    pw_k: int = 10,
    pw_p: float = 2.00,
    pw_f: float = 1.30,
    external_models: Sequence[str] = ("ann", "adam_ann", "cnn_1d", "deep_learning", "svm", "random_forest", "xgboost"),
    max_candidates: Optional[int] = None,
    max_iter_partition: int = 300,
    max_iter_logit: int = 1000,
    max_base_features: Optional[int] = 256,
    interaction_features: int = 0,
    max_external_train_samples: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run selected MNIST scripts and return their in-memory outputs."""
    selected = _selected_steps(steps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, object] = {}

    if "clustering" in selected:
        module = _load_classification_module("01_mnist_clustering_only.py", "fpw_run_mnist_clustering")
        results["clustering"] = module.run_mnist_clustering_only(
            split_path=split_path,
            output_path=output_dir / "01_mnist_clustering_only.csv",
            k_values_text=clustering_k,
            p_values_text=clustering_p,
            f_values_text=clustering_f,
            truncation="hpd",
            max_iter=max_iter_partition,
            random_state=random_state,
            max_candidates=max_candidates,
            search_record_path=output_dir / "01_mnist_clustering_search.csv",
        )

    if "raw_logit" in selected:
        module = _load_classification_module("02_raw_logit_baselines.py", "fpw_run_raw_logit")
        results["raw_logit"] = module.run_raw_logit_baselines(
            split_path=split_path,
            output_path=output_dir / "02_raw_logit_baselines.csv",
            degrees=degrees,
            max_iter=max_iter_logit,
            max_base_features=784,
            interaction_features=interaction_features,
        )

    if "crisp_local" in selected:
        module = _load_classification_module("03_crisp_local_logit.py", "fpw_run_crisp_logit")
        results["crisp_local"] = module.run_crisp_local_logit(
            split_path=split_path,
            output_path=output_dir / "03_crisp_local_logit.csv",
            degrees=degrees,
            truncations=truncations,
            k_values_text=crisp_k,
            p_values_text=crisp_p,
            max_candidates=max_candidates,
            max_iter_partition=max_iter_partition,
            random_state=random_state,
            max_iter_logit=max_iter_logit,
            max_base_features=max_base_features,
            interaction_features=interaction_features,
            search_record_path=output_dir / "03_crisp_local_logit_search.csv",
        )

    if "fuzzy_local" in selected:
        module = _load_classification_module("04_fuzzy_local_logit.py", "fpw_run_fuzzy_logit")
        results["fuzzy_local"] = module.run_fuzzy_local_logit(
            split_path=split_path,
            output_path=output_dir / "04_fuzzy_local_logit.csv",
            degrees=degrees,
            truncations=truncations,
            k_values_text=fuzzy_k,
            p_values_text=fuzzy_p,
            f_values_text=fuzzy_f,
            max_candidates=max_candidates,
            max_iter_partition=max_iter_partition,
            random_state=random_state,
            max_iter_logit=max_iter_logit,
            max_base_features=max_base_features,
            interaction_features=interaction_features,
            search_record_path=output_dir / "04_fuzzy_local_logit_search.csv",
        )

    feature_path = output_dir / "05_mnist_feature_layers.npz"
    if "features" in selected:
        module = _load_classification_module("05_modified_kmeans_mnist_features.py", "fpw_run_mnist_features")
        results["features"] = module.run_mnist_feature_layers(
            split_path=split_path,
            output_npz=feature_path,
            summary_output=output_dir / "05_mnist_feature_layers_summary.csv",
            k_values_text=feature_k,
            p_values_text=feature_p,
            f_values_text=feature_f,
            pw_k=pw_k,
            pw_p=pw_p,
            pw_f=pw_f,
            max_candidates=max_candidates,
            max_iter_partition=max_iter_partition,
            random_state=random_state,
            max_iter_logit=max_iter_logit,
        )

    if "external" in selected:
        if not feature_path.exists():
            feature_module = _load_classification_module("05_modified_kmeans_mnist_features.py", "fpw_run_mnist_features_for_external")
            feature_module.run_mnist_feature_layers(
                split_path=split_path,
                output_npz=feature_path,
                summary_output=output_dir / "05_mnist_feature_layers_summary.csv",
                k_values_text=feature_k,
                p_values_text=feature_p,
                f_values_text=feature_f,
                pw_k=pw_k,
                pw_p=pw_p,
                pw_f=pw_f,
                max_candidates=max_candidates,
                max_iter_partition=max_iter_partition,
                random_state=random_state,
                max_iter_logit=max_iter_logit,
            )
        module = _load_classification_module("06_external_classifiers.py", "fpw_run_external_classifiers")
        results["external"] = module.run_external_classifiers(
            feature_path=feature_path,
            output_path=output_dir / "06_external_classifiers.csv",
            models=external_models,
            random_state=random_state,
            max_train_samples=max_external_train_samples,
        )

    if "centroid" in selected:
        module = _load_classification_module("07_mnist_centroid_standardization.py", "fpw_run_centroid_standardization")
        results["centroid"] = module.run_mnist_centroid_standardization(
            split_path=split_path,
            output_dir=output_dir,
            f=pw_f,
            p=pw_p,
        )

    if "collect" in selected:
        module = _load_classification_module("09_collect_mnist_results.py", "fpw_run_collect_mnist")
        results["collect"] = module.collect_mnist_results(input_dir=output_dir, output_dir=output_dir)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MNIST classification experiment pipeline.")
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--steps", default="all")
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--truncations", default="dtd,harmonic,sp,entropy,hpd")
    parser.add_argument("--clustering-k", default="10")
    parser.add_argument("--clustering-p", default="2.00")
    parser.add_argument("--clustering-f", default="1.30")
    parser.add_argument("--crisp-k", default="760,1020,1400,1750")
    parser.add_argument("--crisp-p", default="2.00")
    parser.add_argument("--fuzzy-k", default="840,1200,1560,1920")
    parser.add_argument("--fuzzy-p", default="1.10,1.15,1.20,1.25")
    parser.add_argument("--fuzzy-f", default="1.10,1.20,1.30")
    parser.add_argument("--feature-k", default="10")
    parser.add_argument("--feature-p", default="2.00")
    parser.add_argument("--feature-f", default="1.30")
    parser.add_argument("--pw-k", type=int, default=10)
    parser.add_argument("--pw-p", type=float, default=2.00)
    parser.add_argument("--pw-f", type=float, default=1.30)
    parser.add_argument("--external-models", default="ann,adam_ann,cnn_1d,deep_learning,svm,random_forest,xgboost")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-iter-partition", type=int, default=300)
    parser.add_argument("--max-iter-logit", type=int, default=1000)
    parser.add_argument("--max-base-features", type=int, default=256)
    parser.add_argument("--interaction-features", type=int, default=0)
    parser.add_argument("--max-external-train-samples", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_mnist_experiment(
        split_path=args.split_path,
        output_dir=args.output_dir,
        steps=args.steps,
        degrees=[int(value) for value in _parse_csv_list(args.degrees)],
        truncations=_parse_csv_list(args.truncations),
        clustering_k=args.clustering_k,
        clustering_p=args.clustering_p,
        clustering_f=args.clustering_f,
        crisp_k=args.crisp_k,
        crisp_p=args.crisp_p,
        fuzzy_k=args.fuzzy_k,
        fuzzy_p=args.fuzzy_p,
        fuzzy_f=args.fuzzy_f,
        feature_k=args.feature_k,
        feature_p=args.feature_p,
        feature_f=args.feature_f,
        pw_k=args.pw_k,
        pw_p=args.pw_p,
        pw_f=args.pw_f,
        external_models=_parse_csv_list(args.external_models),
        max_candidates=args.max_candidates,
        max_iter_partition=args.max_iter_partition,
        max_iter_logit=args.max_iter_logit,
        max_base_features=args.max_base_features,
        interaction_features=args.interaction_features,
        max_external_train_samples=args.max_external_train_samples,
        random_state=args.random_state,
    )
    print("Completed MNIST steps:", ", ".join(results.keys()))


if __name__ == "__main__":
    main()

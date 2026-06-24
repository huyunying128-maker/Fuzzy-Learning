"""
Top-level experiment runner for partition-weighted local learning.

The runner launches the dataset, regression, classification, and summary scripts
with compact default settings suitable for a quick reproducibility check.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_command(command: list[str], enabled: bool = True) -> None:
    """Run one subprocess command and print it in shell-style form."""
    if not enabled:
        return
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact FPWL experiment workflows.")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--dataset", choices=["concrete", "superconductivity"], default="concrete")
    parser.add_argument("--mnist-samples", type=int, default=2000)
    args = parser.parse_args()

    if not args.skip_data:
        run_command([PYTHON, "dataset/download_datasets.py"])
        raw_path = "data/raw/Concrete_Data.xls" if args.dataset == "concrete" else "data/raw/superconductivity/train.csv"
        bundle_path = f"data/processed/{args.dataset}_regression.npz"
        run_command([
            PYTHON,
            "dataset/prepare_regression_data.py",
            "--dataset",
            args.dataset,
            "--input",
            raw_path,
            "--output",
            bundle_path,
        ])
        run_command([
            PYTHON,
            "dataset/prepare_mnist_data.py",
            "--output",
            "data/processed/mnist_classification.npz",
            "--n-samples",
            str(args.mnist_samples),
        ])

    if not args.skip_regression:
        run_command([
            PYTHON,
            "regression/run_global_polynomial_baselines.py",
            "--dataset",
            args.dataset,
            "--degrees",
            "1,2",
        ])
        run_command([
            PYTHON,
            "regression/run_local_regression_models.py",
            "--dataset",
            args.dataset,
            "--degrees",
            "1",
            "--partitions",
            "crisp,fuzzy",
            "--truncations",
            "hpd",
            "--grid-mode",
            "quick",
        ])
        run_command([
            PYTHON,
            "regression/run_pw_regression_baselines.py",
            "--dataset",
            args.dataset,
            "--models",
            "random_forest,xgboost",
        ])
        run_command([PYTHON, "regression/summarize_regression_results.py"])

    if not args.skip_classification:
        run_command([
            PYTHON,
            "classification/run_mnist_local_logit.py",
            "--degrees",
            "1",
            "--partitions",
            "fuzzy",
            "--truncations",
            "hpd",
            "--k-values",
            "10",
            "--n-samples",
            str(args.mnist_samples),
        ])
        run_command([
            PYTHON,
            "classification/run_mnist_pw_classifiers.py",
            "--classifiers",
            "ann,random_forest",
            "--k",
            "10",
            "--n-samples",
            str(args.mnist_samples),
        ])
        run_command([PYTHON, "classification/run_mnist_centroid_visualization.py"])
        run_command([PYTHON, "classification/summarize_classification_results.py"])


if __name__ == "__main__":
    main()

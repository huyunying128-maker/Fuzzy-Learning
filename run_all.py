"""
Top-level runner for the fuzzy partition-weighted learning repository.

The runner executes selected stages of the repository while preserving the modular
file organization. It is a convenience entry point; every script can also be run
separately from its own folder.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command: Sequence[str], dry_run: bool = False) -> None:
    """Run a Python command and echo it in a readable form."""
    text = " ".join(str(part) for part in command)
    print(text)
    if not dry_run:
        subprocess.run(list(command), check=True)


def script_path(*parts: str) -> Path:
    """Build a repository-local script path."""
    return PROJECT_ROOT.joinpath(*parts)


def py_script(path: Path, *args: object) -> List[str]:
    """Create a command list for a repository-local Python script."""
    return [sys.executable, str(path), *[str(arg) for arg in args]]


def build_prepare_commands(args: argparse.Namespace) -> List[List[str]]:
    """Create data-preparation commands for the available datasets."""
    commands: List[List[str]] = []
    processed = args.processed_dir
    if args.dataset in {"all", "concrete"}:
        command = py_script(script_path("dataset", "01_prepare_concrete.py"), "--output-dir", processed)
        if args.concrete_input:
            command.extend(["--input", str(args.concrete_input)])
        commands.append(command)
    if args.dataset in {"all", "superconductivity"}:
        command = py_script(script_path("dataset", "02_prepare_superconductivity.py"), "--output-dir", processed)
        if args.superconductivity_input:
            command.extend(["--input", str(args.superconductivity_input)])
        commands.append(command)
    if args.dataset in {"all", "mnist"}:
        command = py_script(script_path("dataset", "03_prepare_mnist.py"), "--output-dir", processed, "--source", args.mnist_source)
        commands.append(command)
    return commands


def build_split_commands(args: argparse.Namespace) -> List[List[str]]:
    """Create train-validation-test split commands."""
    split_script = script_path("dataset", "04_train_valid_test_split.py")
    processed = args.processed_dir
    split_dir = args.split_dir
    commands: List[List[str]] = []
    if args.dataset in {"all", "concrete"}:
        commands.append(py_script(
            split_script,
            "--input", processed / "concrete.csv",
            "--dataset-name", "concrete",
            "--task", "regression",
            "--target", "concrete_compressive_strength",
            "--output-dir", split_dir,
            "--scale",
        ))
    if args.dataset in {"all", "superconductivity"}:
        commands.append(py_script(
            split_script,
            "--input", processed / "superconductivity.csv",
            "--dataset-name", "superconductivity",
            "--task", "regression",
            "--target", "critical_temp",
            "--output-dir", split_dir,
            "--scale",
        ))
    if args.dataset in {"all", "mnist"}:
        commands.append(py_script(
            split_script,
            "--input", processed / "mnist_full.npz",
            "--dataset-name", "mnist",
            "--task", "classification",
            "--output-dir", split_dir,
        ))
    return commands


def _shared_grid_args(args: argparse.Namespace) -> List[str]:
    grid_args: List[str] = []
    if args.k_values:
        grid_args.extend(["--k-values", args.k_values])
    if args.p_values:
        grid_args.extend(["--p-values", args.p_values])
    if args.f_values:
        grid_args.extend(["--f-values", args.f_values])
    if args.max_candidates is not None:
        grid_args.extend(["--max-candidates", str(args.max_candidates)])
    return grid_args


def build_regression_commands(args: argparse.Namespace) -> List[List[str]]:
    """Create Concrete and Superconductivity regression experiment commands."""
    commands: List[List[str]] = []
    grid_args = _shared_grid_args(args)
    if args.dataset in {"all", "concrete"}:
        command = py_script(
            script_path("regression", "08_run_concrete_experiment.py"),
            "--split-path", args.split_dir / "concrete_split.npz",
            "--output-dir", args.output_dir / "regression" / "concrete",
            "--degrees", args.degrees,
            "--truncations", args.truncations,
        )
        command.extend(grid_args)
        if args.skip_external:
            command.append("--skip-external")
        commands.append(command)
    if args.dataset in {"all", "superconductivity"}:
        command = py_script(
            script_path("regression", "09_run_superconductivity_experiment.py"),
            "--split-path", args.split_dir / "superconductivity_split.npz",
            "--output-dir", args.output_dir / "regression" / "superconductivity",
            "--degrees", args.degrees,
            "--truncations", args.truncations,
        )
        command.extend(grid_args)
        if args.skip_external:
            command.append("--skip-external")
        commands.append(command)
    if args.dataset in {"all", "concrete", "superconductivity"}:
        commands.append(py_script(
            script_path("regression", "10_collect_regression_results.py"),
            "--input-root", args.output_dir / "regression",
            "--output-dir", args.output_dir / "summary",
        ))
    return commands


def build_classification_commands(args: argparse.Namespace) -> List[List[str]]:
    """Create MNIST classification experiment commands."""
    if args.dataset not in {"all", "mnist"}:
        return []
    command = py_script(
        script_path("classification", "08_run_mnist_experiment.py"),
        "--split-path", args.split_dir / "mnist_split.npz",
        "--output-dir", args.output_dir / "classification" / "mnist",
        "--steps", args.mnist_steps,
        "--degrees", args.degrees,
        "--truncations", args.truncations,
        "--max-iter-partition", args.max_iter_partition,
        "--max-iter-logit", args.max_iter_logit,
    )
    if args.max_candidates is not None:
        command.extend(["--max-candidates", str(args.max_candidates)])
    return [command]


def build_appendix_commands(args: argparse.Namespace) -> List[List[str]]:
    """Create commands for the appendix numerical examples."""
    output_dir = args.output_dir / "appendix_examples"
    return [
        py_script(script_path("appendix_examples", "01_appendix_points_and_centroids.py"), "--output-dir", output_dir),
        py_script(script_path("appendix_examples", "02_crisp_clustering_example.py"), "--output-dir", output_dir),
        py_script(script_path("appendix_examples", "03_fuzzy_clustering_example.py"), "--output-dir", output_dir),
        py_script(script_path("appendix_examples", "04_appendix_3d_visualization.py"), "--output-dir", output_dir),
    ]


def build_figure_commands(args: argparse.Namespace) -> List[List[str]]:
    """Create commands that export tables, plots, and cross-dataset summaries."""
    figure_root = args.output_dir / "figures"
    summary_dir = args.output_dir / "summary"
    return [
        py_script(script_path("results_and_figures", "01_export_paper_tables.py"), "--output-dir", args.output_dir / "paper_tables"),
        py_script(script_path("results_and_figures", "02_plot_concrete_results.py"), "--output-dir", figure_root / "concrete"),
        py_script(script_path("results_and_figures", "03_plot_superconductivity_results.py"), "--output-dir", figure_root / "superconductivity"),
        py_script(script_path("results_and_figures", "04_plot_mnist_results.py"), "--output-dir", figure_root / "mnist"),
        py_script(script_path("results_and_figures", "05_plot_external_model_results.py"), "--output-dir", figure_root / "external_models"),
        py_script(script_path("results_and_figures", "06_cross_dataset_summary.py"), "--output-dir", summary_dir, "--figure-dir", figure_root / "cross_dataset"),
    ]


def selected_stages(stage_text: str) -> List[str]:
    """Resolve the stage list used by the top-level runner."""
    if stage_text == "all":
        return ["prepare", "split", "regression", "classification", "appendix", "figures"]
    return [part.strip().lower() for part in stage_text.split(",") if part.strip()]


def build_commands(args: argparse.Namespace) -> List[List[str]]:
    """Build commands for all selected stages."""
    builders = {
        "prepare": build_prepare_commands,
        "split": build_split_commands,
        "regression": build_regression_commands,
        "classification": build_classification_commands,
        "appendix": build_appendix_commands,
        "figures": build_figure_commands,
    }
    commands: List[List[str]] = []
    for stage in selected_stages(args.stage):
        if stage not in builders:
            raise ValueError(f"Unknown stage: {stage}")
        commands.extend(builders[stage](args))
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected fuzzy partition-weighted learning stages.")
    parser.add_argument("--stage", default="all", help="all or comma-separated stages: prepare,split,regression,classification,appendix,figures")
    parser.add_argument("--dataset", choices=["all", "concrete", "superconductivity", "mnist"], default="all")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--concrete-input", type=Path, default=None)
    parser.add_argument("--superconductivity-input", type=Path, default=None)
    parser.add_argument("--mnist-source", choices=["auto", "keras", "openml"], default="auto")
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--truncations", default="dtd,harmonic,sp,entropy,hpd")
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--f-values", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-iter-partition", type=int, default=300)
    parser.add_argument("--max-iter-logit", type=int, default=1000)
    parser.add_argument("--mnist-steps", default="all")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    for command in commands:
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

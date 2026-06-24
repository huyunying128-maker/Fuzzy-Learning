"""
Configuration values for MNIST classification experiments.

The module stores default paths, search grids, model names, and numerical
settings shared by the local-logit and feature-layer classification scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLASSES = 10
MAX_ITER = 300
TOLERANCE = 1.0e-5
LOGIT_MAX_ITER = 300

DEGREE_GRID = (1, 2, 3, 4)
TRUNCATION_GRID = ("dtd", "harmonic", "sp", "entropy", "hpd")
PARTITION_GRID = ("crisp", "fuzzy")
FUZZIFIER_GRID = (1.10, 1.20, 1.30, 1.60, 2.00)
DISTANCE_ORDER_GRID = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 2.00)

QUICK_K_GRID = (10, 20, 40)
PAPER_K_GRID = (700, 760, 840, 1200, 1560, 1920)

DEFAULT_FEATURE_CAP = 64
DEFAULT_PCA_COMPONENTS = 64
DEFAULT_SAMPLE_SIZE = None

CLASSIFIER_NAMES = (
    "ann",
    "adam_ann",
    "deep_learning",
    "svm",
    "random_forest",
    "xgboost",
)


@dataclass(frozen=True)
class ClassificationDatasetSpec:
    """Metadata and default paths for one classification dataset."""

    name: str
    display_name: str
    task: str
    default_bundle_path: Path
    default_result_path: Path


MNIST_SPEC = ClassificationDatasetSpec(
    name="mnist",
    display_name="MNIST handwritten digits",
    task="classification",
    default_bundle_path=Path("data/processed/mnist_classification.npz"),
    default_result_path=Path("results/classification/mnist"),
)


DEGREE_NAMES = {
    1: "linear",
    2: "quadratic",
    3: "cubic",
    4: "quartic",
}


def degree_name(degree: int) -> str:
    """Return the display name for a local logit degree."""
    return DEGREE_NAMES.get(int(degree), f"degree_{degree}")


def parse_int_list(values: str | Iterable[int] | None, default: Iterable[int]) -> list[int]:
    """Parse a comma-separated integer list."""
    if values is None:
        return list(default)
    if isinstance(values, str):
        return [int(item.strip()) for item in values.split(",") if item.strip()]
    return [int(value) for value in values]


def parse_float_list(values: str | Iterable[float] | None, default: Iterable[float]) -> list[float]:
    """Parse a comma-separated float list."""
    if values is None:
        return list(default)
    if isinstance(values, str):
        return [float(item.strip()) for item in values.split(",") if item.strip()]
    return [float(value) for value in values]


def parse_str_list(values: str | Iterable[str] | None, default: Iterable[str]) -> list[str]:
    """Parse a comma-separated string list."""
    if values is None:
        return list(default)
    if isinstance(values, str):
        return [item.strip().lower() for item in values.split(",") if item.strip()]
    return [str(value).lower().strip() for value in values]


def k_grid(mode: str = "quick") -> tuple[int, ...]:
    """Return a predefined MNIST k grid."""
    key = str(mode).lower().strip()
    if key == "quick":
        return QUICK_K_GRID
    if key == "paper":
        return PAPER_K_GRID
    raise ValueError("mode must be 'quick' or 'paper'.")


def result_file(filename: str) -> Path:
    """Return a result-table path for MNIST classification."""
    return MNIST_SPEC.default_result_path / filename


__all__ = [
    "RANDOM_STATE",
    "TEST_SIZE",
    "N_CLASSES",
    "MAX_ITER",
    "TOLERANCE",
    "LOGIT_MAX_ITER",
    "DEGREE_GRID",
    "TRUNCATION_GRID",
    "PARTITION_GRID",
    "FUZZIFIER_GRID",
    "DISTANCE_ORDER_GRID",
    "QUICK_K_GRID",
    "PAPER_K_GRID",
    "DEFAULT_FEATURE_CAP",
    "DEFAULT_PCA_COMPONENTS",
    "DEFAULT_SAMPLE_SIZE",
    "CLASSIFIER_NAMES",
    "ClassificationDatasetSpec",
    "MNIST_SPEC",
    "DEGREE_NAMES",
    "degree_name",
    "parse_int_list",
    "parse_float_list",
    "parse_str_list",
    "k_grid",
    "result_file",
]

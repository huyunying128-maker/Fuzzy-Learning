"""
Configuration objects for regression experiments.

The module defines dataset names, default paths, model-search grids, and shared
numeric settings for the concrete and superconductivity studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RANDOM_STATE = 42
TEST_SIZE = 0.2
RIDGE_ALPHA = 1.0e-6
MAX_ITER = 300
TOLERANCE = 1.0e-5

DEGREE_GRID = (1, 2, 3, 4)
TRUNCATION_GRID = ("dtd", "harmonic", "sp", "entropy", "hpd")
PARTITION_GRID = ("crisp", "fuzzy")

FUZZIFIER_GRID = (1.05, 1.10, 1.20, 1.30, 1.60, 2.00)
DISTANCE_ORDER_GRID = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 2.00)

QUICK_K_GRID = {
    "concrete": (7, 12, 20, 33, 36),
    "superconductivity": (200, 500, 1000, 1620),
}

PAPER_K_GRID = {
    "concrete": (7, 8, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 46),
    "superconductivity": (1620, 1622, 1623, 1624, 1625, 1627, 1714, 1718),
}


@dataclass(frozen=True)
class RegressionDatasetSpec:
    """Metadata and default paths for one regression dataset."""

    name: str
    display_name: str
    task: str
    default_raw_path: Path
    default_bundle_path: Path
    default_result_path: Path


DATASET_SPECS = {
    "concrete": RegressionDatasetSpec(
        name="concrete",
        display_name="Concrete compressive strength",
        task="regression",
        default_raw_path=Path("data/raw/concrete_data.csv"),
        default_bundle_path=Path("data/processed/concrete_regression.npz"),
        default_result_path=Path("results/regression/concrete"),
    ),
    "superconductivity": RegressionDatasetSpec(
        name="superconductivity",
        display_name="Superconductivity critical temperature",
        task="regression",
        default_raw_path=Path("data/raw/superconductivity.csv"),
        default_bundle_path=Path("data/processed/superconductivity_regression.npz"),
        default_result_path=Path("results/regression/superconductivity"),
    ),
}


DEGREE_NAMES = {
    1: "linear",
    2: "quadratic",
    3: "cubic",
    4: "quartic",
}


def get_dataset_spec(dataset: str) -> RegressionDatasetSpec:
    """Return the configuration for a named regression dataset."""
    key = str(dataset).lower().strip()
    if key not in DATASET_SPECS:
        valid = ", ".join(sorted(DATASET_SPECS))
        raise KeyError(f"Unknown dataset '{dataset}'. Available datasets: {valid}")
    return DATASET_SPECS[key]


def degree_name(degree: int) -> str:
    """Return the display name for a polynomial degree."""
    if degree not in DEGREE_NAMES:
        return f"degree_{degree}"
    return DEGREE_NAMES[degree]


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


def k_grid(dataset: str, mode: str = "quick") -> tuple[int, ...]:
    """Return a predefined k grid for a regression dataset."""
    key = str(dataset).lower().strip()
    if mode == "quick":
        return QUICK_K_GRID[key]
    if mode == "paper":
        return PAPER_K_GRID[key]
    raise ValueError("mode must be 'quick' or 'paper'.")


def result_file(dataset: str, filename: str) -> Path:
    """Return a result-table path for one regression dataset."""
    spec = get_dataset_spec(dataset)
    return spec.default_result_path / filename


__all__ = [
    "RANDOM_STATE",
    "TEST_SIZE",
    "RIDGE_ALPHA",
    "MAX_ITER",
    "TOLERANCE",
    "DEGREE_GRID",
    "TRUNCATION_GRID",
    "PARTITION_GRID",
    "FUZZIFIER_GRID",
    "DISTANCE_ORDER_GRID",
    "QUICK_K_GRID",
    "PAPER_K_GRID",
    "RegressionDatasetSpec",
    "DATASET_SPECS",
    "DEGREE_NAMES",
    "get_dataset_spec",
    "degree_name",
    "parse_int_list",
    "parse_float_list",
    "parse_str_list",
    "k_grid",
    "result_file",
]

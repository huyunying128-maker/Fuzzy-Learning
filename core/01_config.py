"""
Shared configuration values for the partition-weighted learning experiments.

The grid utilities define the numerical search space used by the partition
layer. For a dataset with n observations, the candidate cluster counts run from
2 to floor(n / 10). The distance order p and fuzzy degree f are evaluated from
1.00 to 10.00 with a spacing of 0.05.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
VALIDATION_SIZE: float = 0.20

LOCAL_DEGREES: Sequence[int] = (1, 2, 3, 4)
TRUNCATION_RULES: Sequence[str] = ("dtd", "harmonic", "sp", "entropy", "hpd")

GRID_STEP: float = 0.05
MIN_K: int = 2
K_DIVISOR: int = 10
MIN_DISTANCE_ORDER: float = 1.00
MAX_DISTANCE_ORDER: float = 10.00
MIN_FUZZY_DEGREE: float = 1.00
MAX_FUZZY_DEGREE: float = 10.00

RIDGE_ALPHA: float = 1e-4
MAX_ITERATIONS: int = 300
TOLERANCE: float = 1e-6
EPSILON: float = 1e-12

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"


@dataclass(frozen=True)
class PartitionGrid:
    """Container for the validation grid of the partition layer."""

    k_values: List[int]
    p_values: List[float]
    f_values: List[float]

    def as_dict(self) -> Dict[str, List[float]]:
        """Return the grid in a dictionary form that is convenient for logging."""
        return {
            "k": list(self.k_values),
            "p": list(self.p_values),
            "f": list(self.f_values),
        }


def make_float_grid(start: float, stop: float, step: float = GRID_STEP) -> List[float]:
    """Create a stable decimal grid with values rounded to two digits."""
    start_i = int(round(start * 100))
    stop_i = int(round(stop * 100))
    step_i = int(round(step * 100))
    if step_i <= 0:
        raise ValueError("The grid step must be positive.")
    return [round(v / 100.0, 2) for v in range(start_i, stop_i + 1, step_i)]


def make_k_grid(n_samples: int, min_k: int = MIN_K, divisor: int = K_DIVISOR) -> List[int]:
    """Create the cluster-count grid from 2 to floor(n_samples / 10)."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if divisor < 1:
        raise ValueError("divisor must be positive.")

    max_k = max(min_k, n_samples // divisor)
    max_k = min(max_k, n_samples)
    return list(range(min_k, max_k + 1))


def make_partition_grid(n_samples: int) -> PartitionGrid:
    """Build the full candidate grid for k, p, and f."""
    return PartitionGrid(
        k_values=make_k_grid(n_samples),
        p_values=make_float_grid(MIN_DISTANCE_ORDER, MAX_DISTANCE_ORDER, GRID_STEP),
        f_values=make_float_grid(MIN_FUZZY_DEGREE, MAX_FUZZY_DEGREE, GRID_STEP),
    )


def ensure_output_directories() -> None:
    """Create the standard output folders used by the experiment scripts."""
    for path in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    grid = make_partition_grid(1030)
    print("k range:", grid.k_values[0], "to", grid.k_values[-1], "count", len(grid.k_values))
    print("p range:", grid.p_values[0], "to", grid.p_values[-1], "count", len(grid.p_values))
    print("f range:", grid.f_values[0], "to", grid.f_values[-1], "count", len(grid.f_values))

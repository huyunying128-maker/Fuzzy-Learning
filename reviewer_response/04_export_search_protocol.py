"""Export the theoretical search space and tuning-cost estimate for the legacy pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("outputs/reviewer_response/search_protocol")


def load_config():
    path = PROJECT_ROOT / "core" / "01_config.py"
    spec = importlib.util.spec_from_file_location("reviewer_protocol_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load configuration module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["reviewer_protocol_config"] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, required=True,
                        help="Number of observations available to the legacy candidate search.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--degrees", default="1,2,3,4")
    parser.add_argument("--truncations", default="dtd,harmonic,sp,entropy,hpd")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Legacy scripts cap k/f/p tuples separately for each degree and truncation rule.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_train < 2:
        raise ValueError("--n-train must be at least 2.")
    config = load_config()
    grid = config.make_partition_grid(args.n_train)
    degrees = [int(v.strip()) for v in args.degrees.split(",") if v.strip()]
    truncations = [v.strip().lower() for v in args.truncations.split(",") if v.strip()]

    full_kfp = len(grid.k_values) * len(grid.f_values) * len(grid.p_values)
    actual_kfp = full_kfp if args.max_candidates is None else min(full_kfp, args.max_candidates)
    theoretical = full_kfp * len(truncations) * len(degrees)
    capped = actual_kfp * len(truncations) * len(degrees)
    rows = [
        {
            "parameter": "k",
            "range_or_values": f"{grid.k_values[0]}..{grid.k_values[-1]} inclusive",
            "step": 1,
            "count": len(grid.k_values),
            "selection_role": "searched in candidate grid",
        },
        {
            "parameter": "f",
            "range_or_values": f"{config.MIN_FUZZY_DEGREE:.2f}..{config.MAX_FUZZY_DEGREE:.2f} inclusive",
            "step": config.GRID_STEP,
            "count": len(grid.f_values),
            "selection_role": "searched in candidate grid",
        },
        {
            "parameter": "p",
            "range_or_values": f"{config.MIN_DISTANCE_ORDER:.2f}..{config.MAX_DISTANCE_ORDER:.2f} inclusive",
            "step": config.GRID_STEP,
            "count": len(grid.p_values),
            "selection_role": "searched in candidate grid",
        },
        {
            "parameter": "truncation",
            "range_or_values": ",".join(truncations),
            "step": "categorical",
            "count": len(truncations),
            "selection_role": "compared separately",
        },
        {
            "parameter": "degree",
            "range_or_values": ",".join(str(v) for v in degrees),
            "step": "fixed model setting",
            "count": len(degrees),
            "selection_role": "each degree is evaluated separately",
        },
        {
            "parameter": "ridge_alpha",
            "range_or_values": str(args.alpha),
            "step": "fixed",
            "count": 1,
            "selection_role": "fixed regularization value in legacy regression scripts",
        },
        {
            "parameter": "stopping_tolerance",
            "range_or_values": str(args.tolerance),
            "step": "fixed",
            "count": 1,
            "selection_role": "fixed threshold in legacy scripts",
        },
    ]

    payload = {
        "n_train": int(args.n_train),
        "k_range": [int(grid.k_values[0]), int(grid.k_values[-1])],
        "k_count": len(grid.k_values),
        "f_range": [float(grid.f_values[0]), float(grid.f_values[-1])],
        "f_step": float(config.GRID_STEP),
        "f_count": len(grid.f_values),
        "p_range": [float(grid.p_values[0]), float(grid.p_values[-1])],
        "p_step": float(config.GRID_STEP),
        "p_count": len(grid.p_values),
        "truncations": truncations,
        "degrees": degrees,
        "alpha_fixed": float(args.alpha),
        "tolerance_fixed": float(args.tolerance),
        "kfp_tuples_full": int(full_kfp),
        "kfp_tuples_after_legacy_cap_per_degree_and_rule": int(actual_kfp),
        "theoretical_fits_full_grid": int(theoretical),
        "theoretical_fits_after_legacy_cap": int(capped),
        "max_candidates": args.max_candidates,
        "legacy_selection_protocol": (
            "The legacy scripts evaluate each candidate on one pre-created validation split. "
            "They do not implement K-fold cross-validation. The new reviewer_response/02_repeated_split_ablation.py "
            "runner implements inner K-fold CV inside each repeated outer training split."
        ),
        "legacy_cap_warning": (
            "In the legacy regression scripts, --max-candidates truncates the ordered k/f/p product. "
            "Because the product is ordered, a small cap can overrepresent early low-k/low-f candidates. "
            "For reviewer-response analysis, use explicit candidate lists in the new repeated-split runner."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "legacy_search_ranges.csv", index=False)
    with (output_dir / "legacy_search_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Theoretical full-grid fits: {theoretical:,}")
    print(f"Fits after current candidate cap: {capped:,}")
    print(f"Saved search protocol to {output_dir}")


if __name__ == "__main__":
    main()

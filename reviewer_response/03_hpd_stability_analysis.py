"""Assess HPD stopping-threshold sensitivity and its downstream predictive effects."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

from _shared import (
    _load_core_module,
    evaluate_predictions,
    fit_local_predict,
    fit_partition,
    fit_preprocessor,
    load_raw_dataset,
    membership_for,
    parse_csv_list,
    save_json,
    summarize_numeric,
)


DEFAULT_OUTPUT = Path("outputs/reviewer_response/hpd_stability")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw, y, _, is_pixel_data, _ = load_raw_dataset(args.input, args.task, args.target)
    seeds = parse_csv_list(args.seeds, int)
    tolerances = parse_csv_list(args.tolerances, float)
    if any(value < 0 for value in tolerances):
        raise ValueError("All tolerances must be nonnegative.")

    truncation = _load_core_module("05_truncation_rules.py", "reviewer_hpd_truncation")
    records: List[Dict[str, Any]] = []
    for seed in seeds:
        stratify = y if args.task == "classification" else None
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=args.test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        X_train, X_test, _ = fit_preprocessor(
            X_train_raw, X_test_raw, is_pixel_data=is_pixel_data
        )
        if args.k >= X_train.shape[0]:
            raise ValueError(f"k={args.k} must be smaller than the outer training size {X_train.shape[0]}.")

        reference, reference_fit_n = fit_partition(
            X_train,
            k=args.k,
            f=args.f,
            p=args.p,
            truncation_rule="hpd",
            tolerance=0.0,
            max_iter=args.max_iter,
            random_state=seed,
            partition_sample_size=args.partition_sample_size,
        )
        reference_labels = np.asarray(reference.labels, dtype=int)
        reference_objective = float(reference.objective)
        print(f"[seed {seed}] reference HPD partition: {reference.n_iter} iterations")

        for tolerance in tolerances:
            partition, partition_fit_n = fit_partition(
                X_train,
                k=args.k,
                f=args.f,
                p=args.p,
                truncation_rule="hpd",
                tolerance=tolerance,
                max_iter=args.max_iter,
                random_state=seed,
                partition_sample_size=args.partition_sample_size,
            )
            # Compare labels on the actual partition-fitting rows. If a centroid
            # subsample is used, the comparison remains on that same subsample.
            current_labels = np.asarray(partition.labels, dtype=int)
            if current_labels.shape[0] != reference_labels.shape[0]:
                # This can occur only if a different sample size was used; fit
                # both partitions on the same outer training rows for stability.
                raise RuntimeError("Reference and threshold partitions used different fitting samples.")
            hpd_to_reference = truncation.hereditary_partition_distance(
                reference_labels, current_labels, normalize=True
            )
            ari_to_reference = float(adjusted_rand_score(reference_labels, current_labels))

            U_train = membership_for(X_train, partition, args.f, args.p)
            U_test = membership_for(X_test, partition, args.f, args.p)
            prediction = fit_local_predict(
                X_train,
                y_train,
                U_train,
                X_test,
                U_test,
                task=args.task,
                degree=args.degree,
                regularization=args.regularization,
                f=args.f,
                input_gating=True,
                membership_weighted_fit=True,
                aggregation="fuzzy",
                random_state=seed,
            )
            metrics = evaluate_predictions(args.task, y_test, prediction)
            records.append({
                "dataset": args.dataset_name,
                "task": args.task,
                "seed": seed,
                "k": args.k,
                "f": args.f,
                "p": args.p,
                "degree": args.degree,
                "regularization": args.regularization,
                "truncation_rule": "hpd",
                "tolerance": tolerance,
                "partition_fit_n": int(partition_fit_n),
                "reference_fit_n": int(reference_fit_n),
                "n_iter": int(partition.n_iter),
                "converged": bool(partition.converged),
                "final_hpd_value": float(partition.truncation_value),
                "reference_objective": reference_objective,
                "objective": float(partition.objective),
                "objective_delta_from_reference": float(partition.objective - reference_objective),
                "hpd_to_zero_tolerance_reference": float(hpd_to_reference),
                "ari_to_zero_tolerance_reference": ari_to_reference,
                **{f"test_{key}": value for key, value in metrics.items()},
            })

    result = pd.DataFrame(records)
    result.to_csv(output_dir / "hpd_threshold_run_metrics.csv", index=False)
    metric_cols = ["test_mse", "test_mae", "test_r2"] if args.task == "regression" else ["test_accuracy", "test_cross_entropy"]
    summary_rows = []
    for tolerance, frame in result.groupby("tolerance"):
        row: Dict[str, Any] = {
            "dataset": args.dataset_name,
            "task": args.task,
            "tolerance": tolerance,
            "n_runs": int(len(frame)),
            "mean_n_iter": float(frame["n_iter"].mean()),
            "sd_n_iter": float(frame["n_iter"].std(ddof=1)) if len(frame) > 1 else 0.0,
            "mean_hpd_to_reference": float(frame["hpd_to_zero_tolerance_reference"].mean()),
            "mean_ari_to_reference": float(frame["ari_to_zero_tolerance_reference"].mean()),
            "mean_objective_delta_from_reference": float(frame["objective_delta_from_reference"].mean()),
        }
        for metric in metric_cols:
            stats = summarize_numeric(frame[metric].to_numpy(dtype=float))
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "hpd_threshold_summary.csv", index=False)

    save_json({
        "input": str(Path(args.input).resolve()),
        "dataset": args.dataset_name,
        "task": args.task,
        "seeds": seeds,
        "outer_test_size": args.test_size,
        "k": args.k,
        "f": args.f,
        "p": args.p,
        "degree_fixed": args.degree,
        "regularization": args.regularization,
        "truncation_rule": "hpd",
        "tolerances": tolerances,
        "max_iter": args.max_iter,
        "partition_sample_size": args.partition_sample_size,
        "reference_tolerance": 0.0,
        "reference_definition": "Same seed and outer training data, HPD tolerance 0.0.",
        "purpose": "Threshold sensitivity and association between partition stability and downstream predictive metrics.",
    }, output_dir / "hpd_stability_config.json")
    print(f"Saved threshold-level metrics to {output_dir / 'hpd_threshold_run_metrics.csv'}")
    print(f"Saved summary to {output_dir / 'hpd_threshold_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--target", default=None, help="Target column for CSV input.")
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--f", type=float, default=2.0)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--tolerances", default="0,1e-6,1e-4,1e-3")
    parser.add_argument("--degree", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--regularization", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--partition-sample-size", type=int, default=0,
                        help="Optional centroid-fitting subsample size; 0 uses the full training set.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

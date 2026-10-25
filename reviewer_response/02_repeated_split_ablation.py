"""Repeated outer-split experiments, inner-CV tuning, and component-wise ablations.

The script evaluates a matched set of global, fuzzy local, input-gated, and
feature-layer models. It writes run-level metrics, CV logs, confidence intervals,
paired significance tests, and optional per-observation predictions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from _shared import (
    _membership,
    _make_estimator,
    evaluate_predictions,
    fit_global_predict,
    fit_local_predict,
    fit_partition,
    fit_preprocessor,
    load_raw_dataset,
    make_feature_layer,
    membership_for,
    metric_name,
    parse_csv_list,
    save_json,
    summarize_numeric,
)


DEFAULT_OUTPUT = Path("outputs/reviewer_response/repeated_ablation")


def _candidate_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    k_values = parse_csv_list(args.k_values, int)
    f_values = parse_csv_list(args.f_values, float)
    p_values = parse_csv_list(args.p_values, float)
    regularization_values = parse_csv_list(args.regularization, float)
    tolerance_values = parse_csv_list(args.tolerances, float)
    rules = [item.lower() for item in parse_csv_list(args.truncations, str)]

    if any(k < 2 for k in k_values):
        raise ValueError("All k values must be at least 2.")
    if any(f < 1.0 for f in f_values):
        raise ValueError("All f values must be at least 1.00.")
    if any(p < 1.0 for p in p_values):
        raise ValueError("All p values must be at least 1.00.")
    if any(value < 0 for value in regularization_values):
        raise ValueError("Regularization values must be nonnegative.")
    if any(value < 0 for value in tolerance_values):
        raise ValueError("Stopping tolerances must be nonnegative.")
    supported = {"dtd", "harmonic", "sp", "entropy", "hpd"}
    invalid = sorted(set(rules) - supported)
    if invalid:
        raise ValueError(f"Unsupported truncation rules: {invalid}")

    grid = [
        {
            "k": int(k),
            "f": float(f),
            "p": float(p),
            "truncation": str(rule),
            "tolerance": float(tol),
            "regularization": float(reg),
        }
        for k, f, p, rule, tol, reg in itertools.product(
            k_values, f_values, p_values, rules, tolerance_values, regularization_values
        )
    ]
    if args.max_candidates is not None:
        if args.max_candidates < 1:
            raise ValueError("--max-candidates must be positive.")
        if args.max_candidates < len(grid):
            indices = np.linspace(0, len(grid) - 1, num=args.max_candidates, dtype=int)
            grid = [grid[index] for index in sorted(set(indices.tolist()))]
    if not grid:
        raise ValueError("The candidate grid is empty.")
    return grid


def _cv_splits(y: np.ndarray, task: str, n_splits: int, seed: int):
    if task == "classification":
        counts = np.unique(y, return_counts=True)[1]
        feasible = min(int(n_splits), int(np.min(counts)))
        if feasible < 2:
            raise ValueError("At least two observations per class are needed for inner CV.")
        splitter = StratifiedKFold(n_splits=feasible, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y)), feasible
    splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)))), int(n_splits)


def _cv_candidate(
    X: np.ndarray,
    y: np.ndarray,
    is_pixel_data: bool,
    task: str,
    config: Dict[str, Any],
    degree: int,
    cv_splits,
    seed: int,
    max_iter: int,
    partition_sample_size: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        started = time.perf_counter()
        record = {
            **config,
            "fold": fold_id,
            "degree_fixed": int(degree),
            "status": "ok",
            "cv_loss": np.nan,
            "partition_fit_n": np.nan,
            "partition_iterations": np.nan,
            "partition_converged": False,
            "runtime_sec": np.nan,
            "error": "",
        }
        try:
            X_train, X_valid, _ = fit_preprocessor(
                X[train_idx], X[valid_idx], is_pixel_data=is_pixel_data
            )
            if config["k"] >= X_train.shape[0]:
                raise ValueError(f"k={config['k']} is not smaller than fold training size {X_train.shape[0]}")

            partition, partition_fit_n = fit_partition(
                X_train,
                k=config["k"],
                f=config["f"],
                p=config["p"],
                truncation_rule=config["truncation"],
                tolerance=config["tolerance"],
                max_iter=max_iter,
                random_state=seed + fold_id,
                partition_sample_size=partition_sample_size,
            )
            U_train = membership_for(X_train, partition, config["f"], config["p"])
            U_valid = membership_for(X_valid, partition, config["f"], config["p"])
            prediction = fit_local_predict(
                X_train,
                y[train_idx],
                U_train,
                X_valid,
                U_valid,
                task=task,
                degree=degree,
                regularization=config["regularization"],
                f=config["f"],
                input_gating=True,
                membership_weighted_fit=True,
                aggregation="fuzzy",
                random_state=seed + fold_id,
            )
            metrics = evaluate_predictions(task, y[valid_idx], prediction)
            record["cv_loss"] = float(metrics[metric_name(task)])
            record["partition_fit_n"] = int(partition_fit_n)
            record["partition_iterations"] = int(partition.n_iter)
            record["partition_converged"] = bool(partition.converged)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["runtime_sec"] = float(time.perf_counter() - started)
        records.append(record)
    return records


def _select_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    is_pixel_data: bool,
    task: str,
    grid: List[Dict[str, Any]],
    degree: int,
    cv_folds: int,
    seed: int,
    max_iter: int,
    partition_sample_size: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    splits, actual_folds = _cv_splits(y_train, task, cv_folds, seed)
    fold_records: List[Dict[str, Any]] = []
    for candidate_id, config in enumerate(grid, start=1):
        candidate_records = _cv_candidate(
            X_train, y_train, is_pixel_data, task, config, degree, splits,
            seed + candidate_id * 100, max_iter, partition_sample_size
        )
        for row in candidate_records:
            row["candidate_id"] = candidate_id
        fold_records.extend(candidate_records)

    cv_table = pd.DataFrame(fold_records)
    successful = cv_table[cv_table["status"].eq("ok")].copy()
    if successful.empty:
        errors = cv_table[["k", "f", "p", "truncation", "error"]].head(10).to_dict("records")
        raise RuntimeError(f"Every hyperparameter candidate failed. First errors: {errors}")

    summary = (
        successful.groupby(
            ["candidate_id", "k", "f", "p", "truncation", "tolerance", "regularization"],
            as_index=False,
        )
        .agg(mean_cv_loss=("cv_loss", "mean"), sd_cv_loss=("cv_loss", "std"),
             n_successful_folds=("cv_loss", "count"), mean_fit_runtime_sec=("runtime_sec", "mean"))
        .sort_values(["mean_cv_loss", "candidate_id"], ascending=[True, True])
    )
    # A candidate must succeed on every fold to be eligible for selection.
    summary = summary[summary["n_successful_folds"] == actual_folds]
    if summary.empty:
        raise RuntimeError("No candidate completed every inner-CV fold.")
    best = summary.iloc[0]
    selected = {
        "k": int(best["k"]),
        "f": float(best["f"]),
        "p": float(best["p"]),
        "truncation": str(best["truncation"]),
        "tolerance": float(best["tolerance"]),
        "regularization": float(best["regularization"]),
        "mean_cv_loss": float(best["mean_cv_loss"]),
        "sd_cv_loss": float(best["sd_cv_loss"]) if pd.notna(best["sd_cv_loss"]) else np.nan,
        "n_successful_folds": int(best["n_successful_folds"]),
        "candidate_id": int(best["candidate_id"]),
        "actual_cv_folds": int(actual_folds),
        "cv_total_runtime_sec": float(cv_table["runtime_sec"].sum()),
    }
    cv_table = cv_table.merge(
        summary[["candidate_id", "mean_cv_loss", "sd_cv_loss", "n_successful_folds"]],
        on="candidate_id", how="left", suffixes=("", "_candidate")
    )
    return selected, cv_table


def _run_ablation_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    U_train: np.ndarray,
    U_test: np.ndarray,
    task: str,
    degree: int,
    config: Dict[str, Any],
    seed: int,
    feature_layer_max_elements: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], str]:
    """Fit matched component ablations using one frozen partition."""
    rows: List[Dict[str, Any]] = []
    predictions: Dict[str, np.ndarray] = {}
    methods = [
        ("global_raw", None),
        ("local_raw_unweighted", {"gate": False, "fit_weight": False, "aggregation": "fuzzy", "membership": "fuzzy"}),
        ("ts_style_weighted_local", {"gate": False, "fit_weight": True, "aggregation": "fuzzy", "membership": "fuzzy"}),
        ("gated_unweighted_local", {"gate": True, "fit_weight": False, "aggregation": "fuzzy", "membership": "fuzzy"}),
        ("full_pw_hard_aggregation", {"gate": True, "fit_weight": True, "aggregation": "hard", "membership": "fuzzy"}),
        ("full_pw", {"gate": True, "fit_weight": True, "aggregation": "fuzzy", "membership": "fuzzy"}),
        ("hard_membership_local", {"gate": False, "fit_weight": True, "aggregation": "hard", "membership": "hard"}),
    ]

    hard_train = _membership.hard_membership_from_labels(np.argmax(U_train, axis=1), U_train.shape[1])
    hard_test = _membership.hard_membership_from_labels(np.argmax(U_test, axis=1), U_test.shape[1])

    for method, controls in methods:
        start = time.perf_counter()
        if method == "global_raw":
            pred = fit_global_predict(
                X_train, y_train, X_test, task=task, degree=degree,
                regularization=config["regularization"], random_state=seed
            )
        else:
            if controls["membership"] == "hard":
                Utr, Ute = hard_train, hard_test
            else:
                Utr, Ute = U_train, U_test
            pred = fit_local_predict(
                X_train, y_train, Utr, X_test, Ute,
                task=task,
                degree=degree,
                regularization=config["regularization"],
                f=config["f"],
                input_gating=controls["gate"],
                membership_weighted_fit=controls["fit_weight"],
                aggregation=controls["aggregation"],
                random_state=seed,
            )
        elapsed = float(time.perf_counter() - start)
        metrics = evaluate_predictions(task, y_test, pred)
        predictions[method] = np.asarray(pred)
        rows.append({
            "method": method,
            "status": "ok",
            "runtime_sec": elapsed,
            **{f"test_{key}": value for key, value in metrics.items()},
        })

    # A lightweight mixture-of-experts-inspired control learns a soft gate to
    # reproduce the dominant partition labels, then fits raw-input local experts.
    # It is intentionally not described as a jointly optimized canonical MoE.
    start = time.perf_counter()
    hard_labels = np.argmax(U_train, axis=1)
    active_clusters = np.unique(hard_labels)
    if active_clusters.size >= 2:
        try:
            gate_model = _make_estimator("classification", config["regularization"], random_state=seed)
            gate_model.fit(X_train, hard_labels)
            gate_train = np.zeros_like(U_train, dtype=float)
            gate_test = np.zeros_like(U_test, dtype=float)
            gate_train_probs = gate_model.predict_proba(X_train)
            gate_test_probs = gate_model.predict_proba(X_test)
            for column, cluster_id in enumerate(np.asarray(gate_model.classes_, dtype=int)):
                gate_train[:, cluster_id] = gate_train_probs[:, column]
                gate_test[:, cluster_id] = gate_test_probs[:, column]
            moe_prediction = fit_local_predict(
                X_train, y_train, gate_train, X_test, gate_test,
                task=task,
                degree=degree,
                regularization=config["regularization"],
                f=config["f"],
                input_gating=False,
                membership_weighted_fit=True,
                aggregation="fuzzy",
                random_state=seed,
            )
            elapsed = float(time.perf_counter() - start)
            metrics = evaluate_predictions(task, y_test, moe_prediction)
            predictions["moe_style_learned_gate"] = np.asarray(moe_prediction)
            rows.append({
                "method": "moe_style_learned_gate",
                "status": "ok",
                "runtime_sec": elapsed,
                "note": "Soft gate trained to predict dominant partition labels; raw-input local experts.",
                **{f"test_{key}": value for key, value in metrics.items()},
            })
        except Exception as exc:
            rows.append({
                "method": "moe_style_learned_gate",
                "status": "failed_gate_fit",
                "runtime_sec": float(time.perf_counter() - start),
                "note": f"{type(exc).__name__}: {exc}",
            })
    else:
        rows.append({
            "method": "moe_style_learned_gate",
            "status": "skipped_insufficient_gate_classes",
            "runtime_sec": float(time.perf_counter() - start),
            "note": "The hard partition contained fewer than two active clusters.",
        })

    feature_dim = X_train.shape[1] + U_train.shape[1] + X_train.shape[1] * U_train.shape[1]
    estimated_elements = int(max(X_train.shape[0], X_test.shape[0]) * feature_dim)
    if estimated_elements <= feature_layer_max_elements:
        start = time.perf_counter()
        Z_train = make_feature_layer(X_train, U_train)
        Z_test = make_feature_layer(X_test, U_test)
        pred = fit_global_predict(
            Z_train, y_train, Z_test, task=task, degree=1,
            regularization=config["regularization"], random_state=seed
        )
        elapsed = float(time.perf_counter() - start)
        metrics = evaluate_predictions(task, y_test, pred)
        predictions["membership_feature_layer"] = np.asarray(pred)
        rows.append({
            "method": "membership_feature_layer",
            "status": "ok",
            "runtime_sec": elapsed,
            "feature_dim": int(feature_dim),
            "estimated_matrix_elements": estimated_elements,
            **{f"test_{key}": value for key, value in metrics.items()},
        })
        feature_layer_note = f"evaluated; feature_dim={feature_dim}, estimated_matrix_elements={estimated_elements}"
    else:
        rows.append({
            "method": "membership_feature_layer",
            "status": "skipped_dimension_limit",
            "runtime_sec": 0.0,
            "feature_dim": int(feature_dim),
            "estimated_matrix_elements": estimated_elements,
            "note": f"Skipped because estimated dense matrix elements ({estimated_elements}) exceed limit ({feature_layer_max_elements}).",
        })
        feature_layer_note = f"skipped due to dimension limit; feature_dim={feature_dim}, estimated_matrix_elements={estimated_elements}"
    return rows, predictions, feature_layer_note


def _paired_significance(run_table: pd.DataFrame, task: str) -> pd.DataFrame:
    full = run_table[(run_table["method"] == "full_pw") & (run_table["status"] == "ok")]
    comparators = [
        "global_raw",
        "local_raw_unweighted",
        "ts_style_weighted_local",
        "gated_unweighted_local",
        "full_pw_hard_aggregation",
        "hard_membership_local",
        "membership_feature_layer",
        "moe_style_learned_gate",
    ]
    metrics = ["test_mse", "test_mae", "test_r2"] if task == "regression" else ["test_accuracy", "test_cross_entropy"]
    records: List[Dict[str, Any]] = []
    for metric in metrics:
        p_values = []
        row_indices = []
        for comparator in comparators:
            other = run_table[(run_table["method"] == comparator) & (run_table["status"] == "ok")]
            paired = full[["seed", metric]].merge(other[["seed", metric]], on="seed", suffixes=("_full", "_other"))
            paired = paired.dropna()
            if len(paired) < 2:
                continue
            delta = paired[f"{metric}_full"].to_numpy() - paired[f"{metric}_other"].to_numpy()
            try:
                t_p = float(ttest_rel(paired[f"{metric}_full"], paired[f"{metric}_other"], nan_policy="omit").pvalue)
            except Exception:
                t_p = np.nan
            try:
                if np.allclose(delta, 0):
                    w_p = 1.0
                else:
                    w_p = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
            except Exception:
                w_p = np.nan
            records.append({
                "metric": metric,
                "full_method": "full_pw",
                "comparator": comparator,
                "n_paired_seeds": int(len(paired)),
                "mean_delta_full_minus_comparator": float(np.mean(delta)),
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
                "holm_wilcoxon_p": np.nan,
            })
            row_indices.append(len(records) - 1)
            p_values.append(w_p)
        finite = [(idx, p) for idx, p in zip(row_indices, p_values) if np.isfinite(p)]
        # Holm step-down adjustment within each metric family.
        ordered = sorted(finite, key=lambda item: item[1])
        running_max = 0.0
        m = len(ordered)
        for rank, (idx, pval) in enumerate(ordered):
            adjusted = min(1.0, (m - rank) * pval)
            running_max = max(running_max, adjusted)
            records[idx]["holm_wilcoxon_p"] = running_max
    return pd.DataFrame(records)


def _write_predictions(
    records: List[Dict[str, Any]],
    dataset_name: str,
    task: str,
    seed: int,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> None:
    n = len(y_true)
    for method, pred in predictions.items():
        if task == "regression":
            for i in range(n):
                records.append({
                    "dataset": dataset_name,
                    "task": task,
                    "seed": seed,
                    "method": method,
                    "row_index": i,
                    "y_true": float(y_true[i]),
                    "y_pred": float(np.asarray(pred).reshape(-1)[i]),
                })
        else:
            probs = np.asarray(pred, dtype=float)
            for i in range(n):
                row = {
                    "dataset": dataset_name,
                    "task": task,
                    "seed": seed,
                    "method": method,
                    "row_index": i,
                    "y_true": int(y_true[i]),
                    "y_pred": int(np.argmax(probs[i])),
                }
                for class_id in range(probs.shape[1]):
                    row[f"prob_{class_id}"] = float(probs[i, class_id])
                records.append(row)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw, y, feature_names, is_pixel_data, label_encoder = load_raw_dataset(
        args.input, args.task, args.target
    )
    seeds = parse_csv_list(args.seeds, int)
    grid = _candidate_grid(args)
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2; this runner uses inner cross-validation.")

    config_payload = {
        "input": str(Path(args.input).resolve()),
        "dataset": args.dataset_name,
        "task": args.task,
        "target": args.target,
        "n_observations": int(X_raw.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "feature_names": feature_names,
        "pixel_like_data": bool(is_pixel_data),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "seeds": seeds,
        "outer_test_size": float(args.test_size),
        "inner_cv_folds": int(args.cv_folds),
        "inner_cv_shuffle": True,
        "candidate_grid": grid,
        "candidate_count": len(grid),
        "degree_fixed_for_each_run": int(args.degree),
        "max_partition_iterations": int(args.max_iter),
        "partition_sample_size": int(args.partition_sample_size),
        "feature_layer_max_elements": int(args.feature_layer_max_elements),
        "selection_metric": "MSE" if args.task == "regression" else "cross entropy",
        "regularization_interpretation": (
            "Ridge alpha for regression; inverse C (C=1/value) for logistic regression."
        ),
        "preprocessing": (
            "Median imputation and standardization fit only on each training fold/outer training set."
            if not is_pixel_data
            else "Median imputation fit only on training rows; pixel values retained on [0,1] scale."
        ),
        "note": "This reviewer-response runner is separate from the legacy single-validation-split experiment scripts.",
    }
    save_json(config_payload, output_dir / "run_config.json")

    all_run_rows: List[Dict[str, Any]] = []
    all_cv_rows: List[pd.DataFrame] = []
    all_prediction_rows: List[Dict[str, Any]] = []
    all_selected: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"[seed {seed}] outer split and inner CV")
        stratify = y if args.task == "classification" else None
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=args.test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        selected, cv_table = _select_config(
            X_train_raw,
            y_train,
            is_pixel_data,
            args.task,
            grid,
            args.degree,
            args.cv_folds,
            seed,
            args.max_iter,
            args.partition_sample_size,
        )
        cv_table.insert(0, "seed", seed)
        cv_table.insert(1, "dataset", args.dataset_name)
        cv_table.insert(2, "task", args.task)
        all_cv_rows.append(cv_table)
        all_selected.append({"seed": seed, "dataset": args.dataset_name, "task": args.task, **selected})

        X_train, X_test, _ = fit_preprocessor(
            X_train_raw, X_test_raw, is_pixel_data=is_pixel_data
        )
        partition, partition_fit_n = fit_partition(
            X_train,
            k=selected["k"],
            f=selected["f"],
            p=selected["p"],
            truncation_rule=selected["truncation"],
            tolerance=selected["tolerance"],
            max_iter=args.max_iter,
            random_state=seed,
            partition_sample_size=args.partition_sample_size,
        )
        U_train = membership_for(X_train, partition, selected["f"], selected["p"])
        U_test = membership_for(X_test, partition, selected["f"], selected["p"])

        model_rows, predictions, feature_note = _run_ablation_models(
            X_train, y_train, X_test, y_test, U_train, U_test,
            args.task, args.degree, selected, seed, args.feature_layer_max_elements
        )
        for row in model_rows:
            row.update({
                "dataset": args.dataset_name,
                "task": args.task,
                "seed": seed,
                "selected_k": selected["k"],
                "selected_f": selected["f"],
                "selected_p": selected["p"],
                "selected_truncation": selected["truncation"],
                "selected_tolerance": selected["tolerance"],
                "selected_regularization": selected["regularization"],
                "selected_degree": int(args.degree),
                "selected_mean_cv_loss": selected["mean_cv_loss"],
                "selected_cv_runtime_sec": selected["cv_total_runtime_sec"],
                "partition_fit_n": int(partition_fit_n),
                "partition_iterations": int(partition.n_iter),
                "partition_converged": bool(partition.converged),
                "feature_layer_note": feature_note,
            })
            all_run_rows.append(row)
        if args.save_predictions:
            _write_predictions(
                all_prediction_rows, args.dataset_name, args.task, seed, y_test, predictions
            )

    run_table = pd.DataFrame(all_run_rows)
    cv_all = pd.concat(all_cv_rows, ignore_index=True) if all_cv_rows else pd.DataFrame()
    selected_table = pd.DataFrame(all_selected)
    run_table.to_csv(output_dir / "run_metrics.csv", index=False)
    cv_all.to_csv(output_dir / "inner_cv_candidate_log.csv", index=False)
    selected_table.to_csv(output_dir / "selected_hyperparameters.csv", index=False)

    metric_cols = ["test_mse", "test_mae", "test_r2"] if args.task == "regression" else ["test_accuracy", "test_cross_entropy"]
    summary_rows: List[Dict[str, Any]] = []
    for method, frame in run_table.groupby("method"):
        successful = frame[frame["status"].eq("ok")]
        row: Dict[str, Any] = {
            "dataset": args.dataset_name,
            "task": args.task,
            "method": method,
            "n_successful_runs": int(len(successful)),
        }
        for metric in metric_cols:
            values = successful[metric].to_numpy(dtype=float) if metric in successful else np.array([])
            summary = summarize_numeric(values)
            for stat_name, stat_value in summary.items():
                row[f"{metric}_{stat_name}"] = stat_value
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_mean_sd_ci95.csv", index=False)

    significance = _paired_significance(run_table, args.task)
    significance.to_csv(output_dir / "paired_significance_tests.csv", index=False)

    if args.save_predictions:
        pd.DataFrame(all_prediction_rows).to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"Saved run-level metrics: {output_dir / 'run_metrics.csv'}")
    print(f"Saved mean/SD/95% CI summary: {output_dir / 'summary_mean_sd_ci95.csv'}")
    print(f"Saved paired tests: {output_dir / 'paired_significance_tests.csv'}")
    print(f"Saved CV search log: {output_dir / 'inner_cv_candidate_log.csv'}")
    print(f"Selected hyperparameters: {output_dir / 'selected_hyperparameters.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw numeric CSV or NPZ containing x and y.")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--target", default=None, help="Target column for CSV input.")
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--k-values", default="5,10")
    parser.add_argument("--f-values", default="1.5,2.0")
    parser.add_argument("--p-values", default="2.0")
    parser.add_argument("--truncations", default="hpd")
    parser.add_argument("--tolerances", default="1e-6")
    parser.add_argument("--regularization", default="0.0001,0.01")
    parser.add_argument("--degree", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--partition-sample-size", type=int, default=0,
                        help="Optional centroid-fitting subsample size; 0 uses the full outer/fold training set.")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Optional cap on the listed candidate grid; use small values for pilot runs.")
    parser.add_argument("--feature-layer-max-elements", type=int, default=20000000,
                        help="Skip dense feature-layer fitting when n_rows * feature_dim exceeds this limit.")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

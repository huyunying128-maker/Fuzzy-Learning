"""
Metric utilities for partition-weighted local learning experiments.

The functions in this module compute regression and classification scores,
convert metric dictionaries to tables, and write experiment results to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


_EPS = 1e-12


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute standard regression metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    residual = y_true_arr - y_pred_arr
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + _EPS))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute accuracy and cross entropy for multiclass classification.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)

    if proba is not None:
        proba_arr = np.asarray(proba, dtype=float)
        if proba_arr.ndim != 2:
            raise ValueError("proba must be a two-dimensional array.")
        if proba_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError("y_true and proba must have the same number of rows.")

        proba_arr = np.clip(proba_arr, _EPS, 1.0)
        proba_arr = proba_arr / np.sum(proba_arr, axis=1, keepdims=True)
        pred_arr = np.argmax(proba_arr, axis=1)

        true_idx = y_true_arr.astype(int)
        if np.any(true_idx < 0) or np.any(true_idx >= proba_arr.shape[1]):
            raise ValueError("y_true contains class labels outside the probability table.")
        ce = float(-np.mean(np.log(proba_arr[np.arange(len(true_idx)), true_idx] + _EPS)))
    elif y_pred is not None:
        pred_arr = np.asarray(y_pred).reshape(-1)
        if pred_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        ce = float("nan")
    else:
        raise ValueError("classification_metrics requires y_pred or proba.")

    accuracy = float(np.mean(pred_arr == y_true_arr))
    return {
        "accuracy": accuracy,
        "cross_entropy": ce,
    }


def softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Convert logits to normalized probabilities.
    """
    logits_arr = np.asarray(logits, dtype=float)
    shifted = logits_arr - np.max(logits_arr, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """
    Convert integer class labels to a one-hot matrix.
    """
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if n_classes is None:
        n_classes = int(np.max(y_arr)) + 1
    if np.any(y_arr < 0) or np.any(y_arr >= n_classes):
        raise ValueError("class labels must be in the range [0, n_classes).")

    out = np.zeros((y_arr.shape[0], n_classes), dtype=float)
    out[np.arange(y_arr.shape[0]), y_arr] = 1.0
    return out


def add_run_metadata(metrics: Mapping[str, Any], **metadata: Any) -> dict[str, Any]:
    """
    Combine experiment metadata and metric values in one dictionary.
    """
    row: dict[str, Any] = dict(metadata)
    row.update(dict(metrics))
    return row


def rows_to_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """
    Convert experiment rows to a pandas DataFrame.
    """
    return pd.DataFrame(list(rows))


def save_results(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> pd.DataFrame:
    """
    Save experiment rows as CSV and return the DataFrame.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows_to_frame(rows)
    frame.to_csv(path, index=False)
    return frame


def append_result(row: Mapping[str, Any], output_path: str | Path) -> pd.DataFrame:
    """
    Append one experiment row to a CSV result table.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([dict(row)])

    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, new_row], ignore_index=True)
    else:
        frame = new_row

    frame.to_csv(path, index=False)
    return frame


def save_json(data: Mapping[str, Any], output_path: str | Path) -> None:
    """
    Save a dictionary as formatted JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, ensure_ascii=False), encoding="utf-8")


def load_result_table(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV result table.
    """
    return pd.read_csv(path)


def select_best_row(
    frame: pd.DataFrame,
    metric: str,
    higher_is_better: bool = False,
) -> pd.Series:
    """
    Select the best row according to one metric column.
    """
    if metric not in frame.columns:
        raise KeyError(f"Metric column not found: {metric}")
    idx = frame[metric].idxmax() if higher_is_better else frame[metric].idxmin()
    return frame.loc[idx]


def summarize_by_group(
    frame: pd.DataFrame,
    group_cols: list[str],
    metric: str,
    higher_is_better: bool = False,
) -> pd.DataFrame:
    """
    Select the best row inside each group.
    """
    if not group_cols:
        raise ValueError("group_cols must contain at least one column name.")
    for col in group_cols + [metric]:
        if col not in frame.columns:
            raise KeyError(f"Column not found: {col}")

    idx = frame.groupby(group_cols, dropna=False)[metric].idxmax() if higher_is_better else frame.groupby(group_cols, dropna=False)[metric].idxmin()
    return frame.loc[idx.to_numpy()].reset_index(drop=True)

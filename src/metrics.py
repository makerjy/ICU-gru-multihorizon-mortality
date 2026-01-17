from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def safe_auc_ap(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def stay_level_from_row(
    df_long: pd.DataFrame,
    row_score: np.ndarray,
    id_col: str,
    time_col: str,
    label_col: str,
    cutoff_hours: int,
    agg: str,
) -> tuple[np.ndarray, np.ndarray]:
    row_score = np.asarray(row_score, dtype=np.float32).reshape(-1)
    if len(row_score) != len(df_long):
        raise ValueError("row_score length mismatch with df_long")

    d = df_long.copy()
    d["_row_score"] = row_score
    d = d.sort_values([id_col, time_col])
    d = d.loc[d[time_col] <= cutoff_hours].copy()

    if agg == "max":
        s_stay = d.groupby(id_col)["_row_score"].max()
    elif agg == "mean":
        s_stay = d.groupby(id_col)["_row_score"].mean()
    elif agg == "last":
        s_stay = d.groupby(id_col)["_row_score"].last()
    else:
        raise ValueError(f"Unknown agg: {agg}")

    y_stay = d.groupby(id_col)[label_col].max().astype(int)
    common = s_stay.index.intersection(y_stay.index)
    return y_stay.loc[common].to_numpy(), s_stay.loc[common].to_numpy()


def pick_threshold_by_recall_then_precision(
    y: np.ndarray,
    s: np.ndarray,
    target_recall: float,
) -> tuple[float, float, float]:
    precision, recall, thr = precision_recall_curve(y, s)
    precision = precision[:-1]
    recall = recall[:-1]

    ok = recall >= target_recall
    if not np.any(ok):
        i = int(np.argmax(recall))
        return float(thr[i]), float(precision[i]), float(recall[i])

    cand = np.where(ok)[0]
    best = cand[np.lexsort((-thr[cand], -precision[cand]))][0]
    return float(thr[best]), float(precision[best]), float(recall[best])


def metrics_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    pred = (s >= thr).astype(int)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

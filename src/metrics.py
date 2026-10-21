"""Evaluation metrics for a rare-event classifier.

Convention used everywhere in the project: a row is flagged as a predicted failure when
``score >= threshold``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

from . import config as cfg


def predict_labels(scores: Iterable[float], threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1]
    ).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def f_beta_from_counts(tp: float, fp: float, fn: float, beta: float) -> float:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return _ratio((1 + beta**2) * precision * recall, beta**2 * precision + recall)


def classification_metrics(
    y_true: Iterable[int], scores: Iterable[float], threshold: float
) -> dict[str, float]:
    """Threshold-free and thresholded metrics for the failure class.

    ``precision``/``recall``/``f1`` refer to the *failure* class. ``roc_auc``,
    ``pr_auc`` (average precision) and ``brier`` use the raw scores.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = predict_labels(s, threshold)
    counts = confusion_counts(y, pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    n = len(y)

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    both_classes = 0 < y.sum() < n
    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "n": int(n),
        "n_failures": int(y.sum()),
        **counts,
        "accuracy": _ratio(tp + tn, n),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f_beta_from_counts(tp, fp, fn, 1.0),
        "f2": f_beta_from_counts(tp, fp, fn, 2.0),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 or both_classes else 0.0,
        "false_alarm_rate": _ratio(fp, fp + tn),
        "alert_rate": _ratio(tp + fp, n),
        "roc_auc": float(roc_auc_score(y, s)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if y.sum() else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))),
    }
    return metrics


def threshold_table(
    y_true: Iterable[int], scores: Iterable[float], thresholds: Iterable[float]
) -> pd.DataFrame:
    """Precision / recall / F1 / F2 and alert volume across many thresholds (vectorised)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    t = np.asarray(list(thresholds), dtype=float)
    flagged = s[:, None] >= t[None, :]
    tp = (flagged & (y[:, None] == 1)).sum(axis=0).astype(float)
    fp = (flagged & (y[:, None] == 0)).sum(axis=0).astype(float)
    fn = y.sum() - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        f2 = np.where(
            4 * precision + recall > 0, 5 * precision * recall / (4 * precision + recall), 0.0
        )
    return pd.DataFrame(
        {
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "alerts": (tp + fp).astype(int),
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "tn": (len(y) - tp - fp - fn).astype(int),
        }
    )


def _ranking_order(scores: np.ndarray, seed: int = cfg.RANDOM_STATE) -> np.ndarray:
    """Indices from highest to lowest score; ties are broken with a fixed random jitter."""
    jitter = np.random.default_rng(seed).random(len(scores))
    return np.lexsort((jitter, -scores))


def cumulative_gains(
    y_true: Iterable[int], scores: Iterable[float], seed: int = cfg.RANDOM_STATE
) -> pd.DataFrame:
    """Share of all failures found when inspecting the top-k ranked machines.

    Rows are ranked by score (ties broken randomly with a fixed seed). Columns:
    ``fraction_inspected`` and ``fraction_failures_found`` (both include the origin).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = _ranking_order(s, seed)
    found = np.concatenate([[0], np.cumsum(y[order])])
    inspected = np.arange(len(y) + 1)
    return pd.DataFrame(
        {
            "fraction_inspected": inspected / len(y),
            "fraction_failures_found": found / max(int(y.sum()), 1),
        }
    )


def recall_at_fraction(
    y_true: Iterable[int], scores: Iterable[float], fraction: float, seed: int = cfg.RANDOM_STATE
) -> float:
    """Share of failures contained in the top ``fraction`` of machines by score."""
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0:
        return 0.0
    k = int(np.ceil(fraction * len(y)))
    top = _ranking_order(np.asarray(scores, dtype=float), seed)[:k]
    return float(y[top].sum() / y.sum())


def per_mode_recall(test_df: pd.DataFrame, scores: Iterable[float], threshold: float) -> pd.DataFrame:
    """Recall of confirmed failures broken down by the recorded failure mode.

    The failure-mode flags are used here for *post-hoc analysis only* - they are never
    model inputs. A failure can carry several modes, so rows may count in more than one
    group. ``no_mode_flag`` is a recorded failure for which no mode flag is set. A group
    without any failure in ``test_df`` has an undefined recall (``NaN``), not a recall of 0.
    """
    s = np.asarray(scores, dtype=float)
    flagged = predict_labels(s, threshold).astype(bool)
    failed = test_df[cfg.TARGET].to_numpy().astype(bool)
    rows = []
    for mode in cfg.FAILURE_MODE_COLUMNS:
        mask = failed & test_df[mode].to_numpy().astype(bool)
        rows.append((mode, cfg.FAILURE_MODE_LABELS[mode], mask))
    any_mode = test_df[cfg.FAILURE_MODE_COLUMNS].to_numpy().sum(axis=1) > 0
    rows.append(("no_mode_flag", "Failure without a mode flag", failed & ~any_mode))
    rows.append(("all_failures", "All recorded failures", failed))
    return pd.DataFrame(
        [
            {
                "mode": key,
                "label": label,
                "n_failures": int(mask.sum()),
                "detected": int((flagged & mask).sum()),
                # Recall is undefined (NaN, stored as null in the JSON report) when the group has no failures.
                "recall": float((flagged & mask).sum() / mask.sum()) if mask.sum() else float("nan"),
            }
            for key, label, mask in rows
        ]
    )


def bootstrap_ci(
    y_true: Iterable[int],
    scores: Iterable[float],
    threshold: float,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "pr_auc", "roc_auc"),
    n_boot: int = 1000,
    seed: int = cfg.RANDOM_STATE,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap percentile intervals for hold-out metrics.

    Failures and non-failures are resampled separately so every replicate keeps the
    original class counts. With ~70 hold-out failures the intervals are wide, which is
    the honest message about how precisely the hold-out set pins down each metric.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
        )
        result = classification_metrics(y[idx], s[idx], threshold)
        for name in metrics:
            draws[name].append(result[name])
    alpha = (1.0 - level) / 2.0
    return {
        name: (float(np.nanquantile(v, alpha)), float(np.nanquantile(v, 1.0 - alpha)))
        for name, v in draws.items()
    }

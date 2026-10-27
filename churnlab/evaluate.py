from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def metric_row(y, proba, threshold: float = 0.5, prefix: str = "") -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        f"{prefix}AUC": roc_auc_score(y, proba),
        f"{prefix}PR-AUC": average_precision_score(y, proba),
        f"{prefix}accuracy": accuracy_score(y, pred),
        f"{prefix}F1": f1_score(y, pred, zero_division=0),
        f"{prefix}precision": precision_score(y, pred, zero_division=0),
        f"{prefix}recall": recall_score(y, pred, zero_division=0),
        f"{prefix}MCC": matthews_corrcoef(y, pred),
        f"{prefix}log loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6)),
        f"{prefix}Brier": brier_score_loss(y, proba),
    }


def roc(y, proba) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y, proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


def pr(y, proba) -> pd.DataFrame:
    p, r, thr = precision_recall_curve(y, proba)
    return pd.DataFrame({"precision": p[:-1], "recall": r[:-1], "threshold": thr})


def calibration(y, proba, bins: int = 10) -> pd.DataFrame:
    frac, mean = calibration_curve(y, proba, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"predicted": mean, "observed": frac})


def gains(y, proba, steps: int = 20) -> pd.DataFrame:
    order = np.argsort(-proba)
    y_sorted = np.asarray(y)[order]
    total = y_sorted.sum()
    rows = []
    for k in range(1, steps + 1):
        n = int(round(len(y_sorted) * k / steps))
        captured = y_sorted[:n].sum()
        rows.append({"% contacted": 100 * n / len(y_sorted), "% of churners caught": 100 * captured / total,
                     "lift": (captured / n) / (total / len(y_sorted))})
    return pd.DataFrame(rows)


def confusion(y, proba, threshold: float) -> np.ndarray:
    return confusion_matrix(y, (proba >= threshold).astype(int), labels=[0, 1])


def threshold_table(y, proba) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (proba >= t).astype(int)
        rows.append({"threshold": t, "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
                     "flagged %": 100 * pred.mean()})
    return pd.DataFrame(rows)


@dataclass
class Economics:
    offer_cost: float = 40.0
    success_rate: float = 0.25
    horizon_months: int = 6
    default_value: float = 65.0


def customer_value(X: pd.DataFrame, value_col: str | None, econ: Economics) -> np.ndarray:
    if value_col and value_col in X.columns:
        monthly = X[value_col].fillna(X[value_col].median()).to_numpy(dtype=float)
    else:
        monthly = np.full(len(X), econ.default_value)
    return monthly * econ.horizon_months


def profit_curve(y, proba, value: np.ndarray, econ: Economics) -> pd.DataFrame:
    """Expected campaign profit if we contact everyone scored at or above each threshold."""
    y = np.asarray(y)
    rows = []
    for t in np.round(np.arange(0.02, 0.99, 0.01), 2):
        hit = proba >= t
        saved = (value * y * hit).sum() * econ.success_rate
        cost = hit.sum() * econ.offer_cost
        rows.append({"threshold": t, "contacted": int(hit.sum()), "revenue saved": saved,
                     "campaign cost": cost, "profit": saved - cost})
    return pd.DataFrame(rows)


def benchmark_profits(y, value: np.ndarray, econ: Economics) -> dict:
    y = np.asarray(y)
    everyone = (value * y).sum() * econ.success_rate - len(y) * econ.offer_cost
    perfect = (value * y).sum() * econ.success_rate - y.sum() * econ.offer_cost
    return {"Contact no one": 0.0, "Contact everyone": everyone, "Perfect foresight": perfect}

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(y, p, thr: float = 0.5, n_boot: int = 500, seed: int = 7) -> Dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
           "accuracy": round(float(accuracy_score(y, pred)), 4),
           "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
           "negative_recall": round(float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)), 4)}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            boots.append(f1_score(y[i], pred[i], average="macro"))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (None, None)
    out["macro_f1_ci95"] = [round(float(lo), 4), round(float(hi), 4)] if boots else None
    return out


def best_threshold(y, p) -> float:
    grid = np.linspace(0.2, 0.8, 61)
    return float(max(grid, key=lambda t: f1_score(y, (np.asarray(p) >= t).astype(int), average="macro")))
